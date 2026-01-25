# utils_baseline.py
import torch
import time
import copy
from thop import profile
from tqdm import tqdm
from sklearn.metrics import accuracy_score

MAX_SEQ_LENGTH = 128

def estimate_flops(model, max_seq_length=MAX_SEQ_LENGTH):
    model_copy = copy.deepcopy(model).cpu().eval()
    dummy_input = torch.ones(1, max_seq_length, dtype=torch.long)
    try:
        # BERT标准三输入：input_ids, attention_mask, token_type_ids
        flops, params = profile(model_copy, inputs=(dummy_input, dummy_input, dummy_input), verbose=False)
    except:
        flops, params = 0, sum(p.numel() for p in model_copy.parameters())
    return flops, params

def get_full_metrics_cpu(model, dataloader):
    """
    专门在 CPU 上测量：Accuracy, Params_M, FLOPs_G, FPS
    """
    print("\n--- Moving Model to CPU for Scientific Benchmarking ---")
    model.to("cpu").eval()
    
    # 1. 理论指标 (Params & FLOPs)
    flops_raw, params_raw = estimate_flops(model)
    
    # 2. 实验指标 (Accuracy & FPS)
    # 预热 (Warm-up)
    print("Warming up CPU...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 5: break
            inputs = {k: v.to("cpu") for k, v in batch.items() if k != 'label'}
            _ = model(**inputs)

    predictions, references, total_samples = [], [], 0
    start_time = time.time()
    
    print("Starting full validation set inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="CPU Benchmarking"):
            inputs = {k: v.to("cpu") for k, v in batch.items() if k != 'label'}
            labels = batch['label'].tolist()
            
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1).tolist()
            
            predictions.extend(preds)
            references.extend(labels)
            total_samples += inputs['input_ids'].size(0)
            
    duration = time.time() - start_time
    
    return {
        "accuracy": accuracy_score(references, predictions),
        "params_m": params_raw / 1e6,
        "flops_g": flops_raw / 1e9,
        "fps": total_samples / duration
    }