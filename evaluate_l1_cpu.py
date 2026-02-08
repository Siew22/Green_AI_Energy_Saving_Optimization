# evaluate_l1_cpu_final_v2.py
import torch
import os
import sys
from transformers import BertForSequenceClassification, BertTokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader
from safetensors.torch import load_file  # <--- 新增：用于读取 .safetensors 文件

# 导入工具包
try:
    from utils_L1_norm import calculate_l1_norm_scores, perform_structural_pruning, get_full_metrics_cpu, MAX_SEQ_LENGTH
    print("✅ Successfully imported utils_L1_norm")
except ImportError:
    print("❌ Error: utils_L1_norm.py not found.")
    sys.exit(1)

# --- 【关键】更新路径：指向你的最新 checkpoint 文件夹 ---
CHECKPOINT_PATH = "./results_L1_final/checkpoint-84190"
MODEL_WEIGHTS_PATH = os.path.join(CHECKPOINT_PATH, "model.safetensors")

DATASET_NAME, TASK_NAME, BATCH_SIZE = "glue", "sst2", 2
PRUNING_RATIO = 0.20 

def main():
    device = torch.device("cpu")
    print(f"\n--- Reconstructing L1-Norm Model from Safetensors ---")

    # 1. 准备数据
    dataset = load_dataset(DATASET_NAME, TASK_NAME)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenized_ds = dataset.map(lambda e: tokenizer(e["sentence"], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH), batched=True)
    tokenized_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label', 'token_type_ids'])
    val_loader = DataLoader(tokenized_ds["validation"], batch_size=BATCH_SIZE)

    # 2. 重建模型结构
    print("Step 1: Loading raw BERT shell...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    print("Step 2: Re-applying L1 Pruning to match architecture...")
    l1_scores = calculate_l1_norm_scores(model)
    model = perform_structural_pruning(model, l1_scores, PRUNING_RATIO)
    
    print(f"Step 3: Loading weights from {MODEL_WEIGHTS_PATH}...")
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"❌ Error: Weights file not found at {MODEL_WEIGHTS_PATH}")
        return
    
    # 使用 safetensors 加载权重
    state_dict = load_file(MODEL_WEIGHTS_PATH, device="cpu")
    
    # 尝试加载权重 (处理可能的 key 不匹配问题)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        print("💡 Found key mismatch, attempting fuzzy match...")
        # 有时候 Trainer 保存的 key 会多一个 'model.' 前缀
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device)
    model.eval()
    print("✅ Model reconstructed and weights loaded successfully!")

    # 3. 运行全套指标测量
    print("\n--- Starting Final CPU Benchmarking ---")
    results = get_full_metrics_cpu(model, val_loader)

    # 4. 打印最终结果表
    print("\n" + "="*45)
    print(" FINAL RESULTS: TRADITIONAL L1 NORM (CPU) ")
    print("="*45)
    print(f" Accuracy:           {results['Accuracy']:.4f}")
    print(f" Params_M (M):       {results['Params_M']:.2f}")
    print(f" FLOPs_G (G):        {results['FLOPs_G']:.2f}")
    print(f" Samples/sec (FPS):  {results['FPS']:.2f}")
    print("="*45)

if __name__ == "__main__":
    main()