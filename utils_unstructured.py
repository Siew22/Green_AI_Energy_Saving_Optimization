# utils_unstructured.py
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import time
import copy
from thop import profile
from tqdm import tqdm
from sklearn.metrics import accuracy_score

MAX_SEQ_LENGTH = 128

def estimate_flops(model, max_seq_length=MAX_SEQ_LENGTH):
    """
    注意：非结构化剪枝后，虽然参数变 0，但矩阵形状未变。
    thop 通常会报告原始 FLOPs，这正好能证明为什么它不加速。
    """
    model_copy = copy.deepcopy(model).cpu().eval()
    dummy = torch.ones(1, max_seq_length, dtype=torch.long)
    try:
        flops, params = profile(model_copy, inputs=(dummy, dummy, dummy), verbose=False)
        # 手动计算非零参数，反映真实压缩比
        non_zero_params = sum(torch.sum(p != 0).item() for p in model.parameters())
    except:
        flops, params = 0, sum(torch.sum(p != 0).item() for p in model.parameters())
        non_zero_params = params
    return flops, non_zero_params

def perform_unstructured_l1_pruning(model, ratio=0.20):
    """
    使用 PyTorch 官方 global_unstructured 逻辑。
    """
    print(f"Applying Unstructured Pruning (Ratio: {ratio*100}%)...")
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=ratio,
    )
    # 移除 mask，直接应用到权重
    for module, name in parameters_to_prune:
        prune.remove(module, name)
    return model

def get_full_metrics_cpu(model, dataloader):
    """
    在 CPU 上测量 4 项指标。
    """
    model.to("cpu").eval()
    flops_raw, params_raw = estimate_flops(model)

    # 预热
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 5: break
            model(**{k: v.to("cpu") for k, v in batch.items() if k != 'label'})

    predictions, references, total_samples = [], [], 0
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Unstructured CPU Evaluation"):
            inputs = {k: v.to("cpu") for k, v in batch.items() if k != 'label'}
            outputs = model(**inputs)
            predictions.extend(outputs.logits.argmax(dim=-1).tolist())
            references.extend(batch["label"].tolist())
            total_samples += inputs["input_ids"].size(0)
    
    duration = time.time() - start_time
    return {
        "Accuracy": accuracy_score(references, predictions),
        "Params_M": params_raw / 1e6,
        "FLOPs_G": flops_raw / 1e9,
        "FPS": total_samples / duration
    }