# traditional_L1_norm.py (经典流程版 - 无QAT)

import os
import sys
import torch
import gc
import wandb
from transformers import BertForSequenceClassification, BertTokenizerFast, TrainingArguments, Trainer
from datasets import load_dataset
from torch.utils.data import DataLoader

# 确保可以从 utils.py 导入所有需要的函数
try:
    from utils_L1_norm import (
        estimate_flops, 
        measure_energy_and_speed,
        filter_dataset_intelligently,
        calculate_l1_norm_scores,
        perform_structural_pruning
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
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS_PRUNING_FINETUNE = 5
PRUNING_FLOPs_REDUCTION_TARGET = 0.4
BASELINE_MODEL_PATH = "./results_baseline/fine_tuned_model"
OUTPUT_DIR = "./results_l1_norm_classic" # 使用新文件夹名以区分
os.environ["WANDB_PROJECT"] = "GreenAI-Optimization-Comparison"

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*25}\nRunning Classic L1 Norm Pruning Pipeline (No QAT)\n{'='*25}")
    
    # --- 1. Load Data ---
    print("--- 1. Loading and Preparing Data ---")
    # (数据加载和筛选部分保持不变)
    dataset = load_dataset(DATASET_NAME, TASK_NAME)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    def tokenize_function(e): return tokenizer(e["sentence"], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    columns_to_keep = ["input_ids", "attention_mask", "label", "token_type_ids"]
    tokenized_datasets.set_format('torch', columns=[c for c in columns_to_keep if c in dataset['train'].features])
    
    print("Applying the same intelligent data filtering...")
    temp_model = BertForSequenceClassification.from_pretrained(BASELINE_MODEL_PATH).cuda()
    filtered_train_dataset = filter_dataset_intelligently(
        model=temp_model, tokenizer=tokenizer, dataset=tokenized_datasets["train"], 
        keep_ratio=0.8, batch_size=DEVICE_BATCH_SIZE, device="cuda"
    )
    del temp_model; torch.cuda.empty_cache()
    
    validation_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=DEVICE_BATCH_SIZE)

    # --- 2. Pruning using L1 Norm ---
    print("\n--- 2. Pruning Stage (L1 Norm Method) ---")
    model_to_prune = BertForSequenceClassification.from_pretrained(BASELINE_MODEL_PATH).cuda()
    pruning_scores = calculate_l1_norm_scores(model_to_prune)
    pruned_model, _ = perform_structural_pruning(
        model_to_prune, pruning_scores, PRUNING_FLOPs_REDUCTION_TARGET, MAX_SEQ_LENGTH
    )
    del model_to_prune; torch.cuda.empty_cache()

    # --- 3. Fine-tuning Pruned Model ---
    print("\n--- 3. Fine-tuning Pruned Model ---")
    pruning_finetune_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/pruned_checkpoints",
        run_name="prune_finetune_l1_norm_classic",
        learning_rate=1e-5,
        per_device_train_batch_size=DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS_PRUNING_FINETUNE,
        fp16=True,
        report_to="wandb",
        
        # --- 新增的优化参数 ---
        save_strategy="steps",          # 按步数保存
        save_steps=500,                 # 每500步保存一次 (这是默认值，可以明确写出)
        save_total_limit=1,             # 【【【核心修复】】】: 最多只保留1个最新的checkpoint
        logging_steps=500,              # 建议让日志记录步数和保存步数一致
        load_best_model_at_end=False    # 如果不需要在最后加载最好的模型，可以关闭
    )
    pruned_trainer = Trainer(model=pruned_model, args=pruning_finetune_args, train_dataset=filtered_train_dataset)
    pruned_trainer.train()
    
    final_pruned_save_path = f"{OUTPUT_DIR}/fine_tuned_pruned_model"
    pruned_trainer.save_model(final_pruned_save_path)
    print(f"Fine-tuned pruned model (L1-Norm) saved to {final_pruned_save_path}")
    
    print(f"Fine-tuned pruned model (L1-Norm) saved to {final_pruned_save_path}")
    
    # --- 4. Final Measurement (on GPU) ---
    print("\n--- 4. Measuring Final Pruned FP32 Model (L1 Norm) ---")
    
    # 【【【新增修复】】】: 在测量前重置数据集格式，确保 DataLoader 能正确生成批次
    tokenized_datasets.reset_format()
    tokenized_datasets.set_format('torch', columns=[c for c in columns_to_keep if c in tokenized_datasets["validation"].features])
    validation_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=DEVICE_BATCH_SIZE)

    # 我们直接使用训练好的 pruned_model 进行测量
    final_measurement = measure_energy_and_speed(pruned_model, validation_dataloader, "Final Pruned Model (L1 Norm, GPU)", device="cuda")
    flops, params = estimate_flops(pruned_model, MAX_SEQ_LENGTH)
    
    # --- 5. Final Reporting and Logging ---
    print("\n--- 5. Final Reporting and Logging ---")

    final_results = {
        "accuracy": final_measurement.get('accuracy'),
        "parameters_M": params / 1e6,
        "flops_G": flops / 1e9,
        "fps_gpu": final_measurement.get('samples_per_second'),
        "energy_per_sample_uWh": final_measurement.get('energy_per_sample_uWh'),
        "avg_gpu_power_W": final_measurement.get('avg_gpu_power_W'),
    }

    print(f"\n--- FINAL RESULTS FOR: Classic L1 Norm Method (FP32 on GPU) ---")
    for key, value in final_results.items():
        if value is not None:
            print(f"  - {key.replace('_', ' ').title()}: {value:,.4f}")
        else:
            print(f"  - {key.replace('_', ' ').title()}: N/A")
    print("-" * 30)

    print("\nLogging Classic L1 Norm results to W&B...")
    try:
        run = wandb.init(
            project=os.environ["WANDB_PROJECT"], 
            name="final-model-l1-norm-classic", 
            job_type="measurement"
        )
        run.summary.update(final_results)
        run.finish()
        print("✅ Successfully logged L1 Norm (Classic) results to W&B.")
        print(f"  - Find your run at: {run.url}")
    except Exception as e:
        print(f"\n❌ Could not log to W&B. Error: {e}")