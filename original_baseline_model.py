# original_baseline_model.py
import os
import torch
import time
from transformers import BertForSequenceClassification, BertTokenizerFast, TrainingArguments, Trainer
from datasets import load_dataset
from torch.utils.data import DataLoader
# 确保函数名完全一致
from utils_baseline import get_full_metrics_cpu, estimate_flops, MAX_SEQ_LENGTH

# 配置
MODEL_NAME = "bert-base-uncased"
DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
OUTPUT_DIR = "./results_baseline_final"

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 准备数据 (100% 原始数据)
    print("--- 1. Loading 100% Original Dataset (NO FILTERING) ---")
    dataset = load_dataset("glue", "sst2")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    def tokenize_fn(e): return tokenizer(e["sentence"], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)
    tokenized_ds = dataset.map(tokenize_fn, batched=True)
    tokenized_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label', 'token_type_ids'])

    # 2. 加载模型并在 GPU 训练
    print("\n--- 2. Fine-tuning Baseline Model on GPU ---")
    num_labels = dataset['train'].features['label'].num_classes
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).cuda()
    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        optim="adamw_torch",
        # --- 加上下面这两行核心配置 ---
        save_strategy="epoch",    # 每运行完一个 epoch 存一次
        save_total_limit=1,       # 【最重要】只保留最新的一份，旧的会自动删除
        report_to="wandb"
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized_ds["train"])
    trainer.train()

    # 3. 切换到 CPU 进行最终测量
    print("\n--- 3. Performing Full Metrics Benchmarking on CPU ---")
    val_loader = DataLoader(tokenized_ds["validation"], batch_size=DEVICE_BATCH_SIZE)
    
    # 调用一致的函数名
    results = get_full_metrics_cpu(model, val_loader)

    # 4. 打印最终结果表 (直接拿去填论文)
    print("\n" + "="*45)
    print(" FINAL RESULTS: ORIGINAL BASELINE (CPU) ")
    print("="*45)
    print(f" Accuracy:           {results['accuracy']:.4f}")
    print(f" Params_M (M):       {results['params_m']:.2f}")
    print(f" FLOPs_G (G):        {results['flops_g']:.2f}")
    print(f" Samples/sec (FPS):  {results['fps']:.2f}")
    print("="*45)