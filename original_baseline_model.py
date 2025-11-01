# original_baseline_model.py (Final Version with W&B Logging)
import os
import sys
import torch
import wandb  # <--- 在顶部添加 import
from transformers import BertForSequenceClassification, BertTokenizerFast, TrainingArguments, Trainer
from datasets import load_dataset
from torch.utils.data import DataLoader

# 确保可以从 utils.py 导入所有需要的函数
try:
    from utils import (
        estimate_flops, 
        measure_energy_and_speed,
        filter_dataset_intelligently
    )
except ImportError:
    print("Error: Could not import from utils.py. Make sure it's in the same directory.")
    sys.exit(1)

# --- Configuration ---
MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "glue"
TASK_NAME = "sst2"
MAX_SEQ_LENGTH = 128
DEVICE_BATCH_SIZE = 2
EFFECTIVE_BATCH_SIZE = 4
assert EFFECTIVE_BATCH_SIZE % DEVICE_BATCH_SIZE == 0
GRADIENT_ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // DEVICE_BATCH_SIZE
BASE_OUTPUT_DIR = "./results_baseline"
NUM_EPOCHS_BASELINE = 3

# 设置 W&B 项目名称
os.environ["WANDB_PROJECT"] = "GreenAI-Optimization-Comparison"

if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # --- 1. Loading and Preparing Data ---
    print("--- 1. Loading and Preparing Data ---")
    dataset = load_dataset(DATASET_NAME, TASK_NAME)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    columns_to_keep = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in tokenized_datasets["train"].features:
        columns_to_keep.append("token_type_ids")
    tokenized_datasets.set_format("torch", columns=columns_to_keep)

    # 为了与主实验公平对比，使用验证集进行最终测量
    validation_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=DEVICE_BATCH_SIZE)

    # --- 2. Loading and Fine-tuning Baseline Model ---
    print("\n--- 2. Loading and Fine-tuning Baseline Model ---")
    num_labels = dataset['train'].features['label'].num_classes
    baseline_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).cuda()
    
    # --- Perform Intelligent Data Filtering ---
    print("Applying intelligent data filtering...")
    filtered_train_dataset = filter_dataset_intelligently(
        model=baseline_model, 
        tokenizer=tokenizer,
        dataset=tokenized_datasets["train"],
        keep_ratio=0.8,
        batch_size=DEVICE_BATCH_SIZE,
        device="cuda"
    )

    baseline_training_args = TrainingArguments(
        output_dir=f"{BASE_OUTPUT_DIR}/checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS_BASELINE,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        report_to="wandb",
        run_name="baseline-finetune-standalone", # 区分于主流程中的baseline
        optim="adamw_bnb_8bit"
    )

    trainer = Trainer(
        model=baseline_model,
        args=baseline_training_args,
        train_dataset=filtered_train_dataset
    )
    
    print("Starting baseline training...")
    trainer.train()
    
    final_baseline_save_path = f"{BASE_OUTPUT_DIR}/fine_tuned_model"
    trainer.save_model(final_baseline_save_path)
    print(f"Fine-tuned baseline model saved to {final_baseline_save_path}")

    # --- 3. Measuring Baseline Model Performance ---
    print("\n--- 3. Measuring Baseline Model Performance ---")
    model_for_measurement = BertForSequenceClassification.from_pretrained(final_baseline_save_path).cuda()
    
    flops, params = estimate_flops(model_for_measurement, MAX_SEQ_LENGTH)
    measurement_metrics = measure_energy_and_speed(model_for_measurement, validation_dataloader, "Baseline Model (Validation Set)")
    
    # --- 4. Final Reporting and Logging ---
    print("\n--- 4. Final Reporting and Logging ---")
    
    final_results = {
        "accuracy": measurement_metrics.get('accuracy'),
        "parameters_M": params / 1e6,
        "flops_G": flops / 1e9,
        "fps": measurement_metrics.get('samples_per_second'),
        "energy_per_sample_uWh": measurement_metrics.get('energy_per_sample_uWh'),
        "avg_gpu_power_W": measurement_metrics.get('avg_gpu_power_W'),
    }

    print("\n--- BASELINE MODEL RESULTS ---")
    for key, value in final_results.items():
        if value is not None:
            print(f"  - {key.replace('_', ' ').title()}: {value:,.4f}")
        else:
            print(f"  - {key.replace('_', ' ').title()}: N/A")
    print("-" * 30)

    print("\nLogging baseline results to W&B...")
    try:
        run = wandb.init(
            project=os.environ["WANDB_PROJECT"], 
            name="baseline-standalone-measurement", 
            job_type="measurement",
            reinit=True
        )
        run.summary.update(final_results)
        run.finish()
        print("✅ Successfully logged baseline results to W&B.")
        print(f"  - Find your run at: {run.url}")
    except Exception as e:
        print(f"\n❌ Could not log to W&B. Error: {e}")