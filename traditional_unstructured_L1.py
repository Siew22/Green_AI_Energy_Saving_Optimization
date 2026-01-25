# traditional_unstructured_L1.py
import os
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast, TrainingArguments, Trainer
from datasets import load_dataset
from torch.utils.data import DataLoader
from utils_unstructured import *

# --- 严格对齐超参数 ---
MODEL_NAME = "bert-base-uncased"
PRUNING_RATIO = 0.20
LEARNING_RATE = 1e-5  # 与所有剪枝微调实验对齐
NUM_EPOCHS = 5        # 与所有剪枝微调实验对齐
DEVICE_BATCH_SIZE = 2
OUTPUT_DIR = "./results_unstructured_final"

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 准备 100% 数据 (Baseline 公平性)
    dataset = load_dataset("glue", "sst2")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    tokenized_ds = dataset.map(lambda e: tokenizer(e["sentence"], padding="max_length", truncation=True, max_length=128), batched=True)
    tokenized_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label', 'token_type_ids'])

    # 2. 执行剪枝 (在 GPU 训练前执行)
    print("\n--- Initializing Unstructured Pruning ---")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = perform_unstructured_l1_pruning(model, PRUNING_RATIO)

    # 3. GPU 训练恢复精度
    print("\n--- Fine-tuning Unstructured Model on GPU ---")
    model.cuda()
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=2,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        optim="adamw_torch",
        save_strategy="epoch",
        save_total_limit=1
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized_ds["train"])
    trainer.train()

    # 4. CPU 最终测量
    print("\n--- Final CPU Benchmarking (End-User Scenario) ---")
    val_loader = DataLoader(tokenized_ds["validation"], batch_size=DEVICE_BATCH_SIZE)
    results = measure_performance_cpu(model, val_loader)
    flops, params = estimate_flops(model)

    print("\n" + "="*40)
    print(f"UNSTRUCTURED L1 RESULTS (CPU INFERENCE)")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"FPS:      {results['fps']:.2f}") # 预期：这里的 FPS 会和 Baseline 差不多慢
    print(f"FLOPs:    {flops/1e9:.2f} G")
    print(f"Params:   {params/1e6:.2f} M")
    print("="*40)