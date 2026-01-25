# utils_L1_norm.py
import torch
import time
import copy
from thop import profile
from tqdm import tqdm
import torch_pruning as tp
from sklearn.metrics import accuracy_score

MAX_SEQ_LENGTH = 128

def estimate_flops(model, max_seq_length=MAX_SEQ_LENGTH):
    """与基准测量标准保持 100% 一致"""
    model_copy = copy.deepcopy(model).cpu().eval()
    dummy = torch.ones(1, max_seq_length, dtype=torch.long)
    try:
        flops, params = profile(model_copy, inputs=(dummy, dummy, dummy), verbose=False)
    except:
        flops, params = 0, sum(p.numel() for p in model_copy.parameters())
    return flops, params

def calculate_l1_norm_scores(model):
    """
    核心逻辑：纯权重 L1 评分，不含任何 FLOPs 惩罚项。
    """
    scores = {}
    model.cpu()
    for layer_idx, layer in enumerate(model.bert.encoder.layer):
        # Attention Heads: Sum(|Query_Weights|)
        attn = layer.attention.self
        h_dim = model.config.hidden_size // model.config.num_attention_heads
        q_w = attn.query.weight.data
        for h_idx in range(model.config.num_attention_heads):
            unit_name = f"layer.{layer_idx}.attention.head.{h_idx}"
            scores[unit_name] = q_w[h_idx*h_dim : (h_idx+1)*h_dim, :].abs().sum().item()
        
        # FFN Neurons: Sum(|Intermediate_Weights|)
        ffn_w = layer.intermediate.dense.weight.data
        for n_idx in range(ffn_w.size(0)):
            unit_name = f"layer.{layer_idx}.ffn.neuron.{n_idx}"
            scores[unit_name] = ffn_w[n_idx, :].abs().sum().item()
    return scores

def perform_structural_pruning(model, scores, target_ratio):
    """执行物理剪枝，减掉分数最低的结构单元"""
    model.cpu()
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.ones(1, 128, dtype=torch.long))
    sorted_units = sorted(scores.items(), key=lambda x: x[1])
    num_to_prune = int(len(sorted_units) * target_ratio)
    
    for i in range(num_to_prune):
        unit_name = sorted_units[i][0]
        try:
            parts = unit_name.split('.')
            l_idx, u_type, u_idx = int(parts[1]), parts[2], int(parts[-1])
            if u_type == "attention":
                layer = model.bert.encoder.layer[l_idx].attention.self.query
                h_dim = model.config.hidden_size // model.config.num_attention_heads
                idxs = list(range(u_idx * h_dim, (u_idx + 1) * h_dim))
                group = DG.get_pruning_group(layer, tp.prune_linear_out_channels, idxs=idxs)
            else: # ffn
                layer = model.bert.encoder.layer[l_idx].intermediate.dense
                group = DG.get_pruning_group(layer, tp.prune_linear_out_channels, idxs=[u_idx])
            if DG.check_pruning_group(group): group.prune()
        except: continue
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
        for batch in tqdm(dataloader, desc="L1-Structured CPU Evaluation"):
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