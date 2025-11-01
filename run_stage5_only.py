# run_stage5_only.py (v16 - 浮点验证版)

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizerFast, AutoConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
import sys
import traceback
from collections import OrderedDict

# --- 导入函数 ---
try:
    from utils import measure_energy_and_speed, MAX_SEQ_LENGTH
except ImportError:
    print("Error: Could not import from utils.py.")
    sys.exit(1)

# --- 核心配置 ---
BASE_OUTPUT_DIR = "./results"
FINAL_PRUNED_SAVE_PATH = f"{BASE_OUTPUT_DIR}/fine_tuned_pruned_model_final"
CONVERTED_STATE_DICT_PATH = f"{BASE_OUTPUT_DIR}/quantized_pruned_model_final.pt"
DATASET_NAME = "glue"
TASK_NAME = "sst2"
EVAL_BATCH_SIZE = 2

if __name__ == "__main__":

    # --- 准备数据集 ---
    print("\n--- Preparing dataset for final evaluation ---")
    dataset = load_dataset(DATASET_NAME, TASK_NAME)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    def tokenize_function(e): return tokenizer(e['sentence'], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    columns_to_keep = ["input_ids", "attention_mask", "label", "token_type_ids"]
    tokenized_datasets.set_format('torch', columns=[c for c in columns_to_keep if c in tokenized_datasets["validation"].features])
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=EVAL_BATCH_SIZE)
    print("DataLoader for evaluation is ready.\n")

    # --- Stage 5: 验证浮点权重加载和推理 ---
    print("--- Stage 5: Verifying FP32 Weight Loading and Inference ---")

    try:
        # Step 1: 加载 state_dict
        print(f"Step 1: Loading the CONVERTED state_dict from {CONVERTED_STATE_DICT_PATH}")
        loaded_state_dict = torch.load(CONVERTED_STATE_DICT_PATH, map_location='cpu', weights_only=True)
        
        # Step 2: 重建一个纯净的【浮点】骨架模型
        print(f"Step 2: Rebuilding pruned FP32 model structure...")
        fp32_model = BertForSequenceClassification.from_pretrained(
            FINAL_PRUNED_SAVE_PATH, ignore_mismatched_sizes=True
        ).cpu()
        fp32_model.eval()
        print("  - Pruned FP32 structure rebuilt.")

        # Step 3: 手动解包权重，并将其转换为 FP32
        print("Step 3: Manually unpacking and dequantizing weights to FP32...")
        unpacked_fp32_state_dict = OrderedDict()
        for key, value in loaded_state_dict.items():
            new_key = key[6:] if key.startswith('model.') else key
            
            if '_packed_params._packed_params' in new_key:
                # 解包并【反量化】权重和偏置
                unpacked_weight, unpacked_bias = value
                weight_key = new_key.replace('_packed_params._packed_params', 'weight')
                bias_key = new_key.replace('_packed_params._packed_params', 'bias')
                unpacked_fp32_state_dict[weight_key] = unpacked_weight.dequantize()
                if unpacked_bias is not None:
                    unpacked_fp32_state_dict[bias_key] = unpacked_bias
            elif '_packed_params' not in new_key and 'scale' not in new_key and 'zero_point' not in new_key:
                 # 只保留非量化参数 (如 LayerNorm, embeddings)
                 unpacked_fp32_state_dict[new_key] = value
        print("  - Unpacking and dequantization complete.")

        # Step 4: 加载解包后的 FP32 权重到 FP32 模型中
        print("Step 4: Loading the unpacked FP32 state_dict into the FP32 model...")
        fp32_model.load_state_dict(unpacked_fp32_state_dict, strict=False)
        print("  - FP32 state_dict loaded successfully!")
        
        fp32_model_loaded_for_measurement = fp32_model
        fp32_model_loaded_for_measurement.eval()
        print("\nSuccessfully reconstructed the FP32 model for measurement (on CPU).")

    except Exception as e:
        print(f"\nAn unexpected error occurred during model reconstruction: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- 最终测量 (在纯浮点模型上) ---
    print("\n--- Starting Final Measurement (on FP32 model) ---")
    try:
        print("\nMeasuring on CPU...")
        measure_energy_and_speed(
            fp32_model_loaded_for_measurement, 
            eval_dataloader, 
            description="DEQUANTIZED FP32 Model Inference (CPU)",
            device="cpu"
        )
    except Exception as e:
        print(f"Error during final measurement: {e}")
        traceback.print_exc()