# utils_L1_norm
import torch
import torch.nn as nn
import time
import subprocess
import os
import copy # Needed for deepcopy in estimate_flops and pruning
from thop import profile # Assuming thop works for your model
from tqdm import tqdm # For progress bars
import torch_pruning as tp
from sklearn.metrics import accuracy_score # Needed for accuracy calculation during measurement
import traceback

# --- Global Configuration (Consider moving to main script config) ---
MAX_SEQ_LENGTH = 128 # Example max sequence length for text classification
# Adjust GPU index if you have multiple GPUs
GPU_INDEX = 0

# [New Function for Intelligent Data Filtering]
def filter_dataset_intelligently(model, tokenizer, dataset, keep_ratio=0.8, batch_size=16, device="cuda"):
    """
    Intelligently filters a dataset based on model uncertainty.
    Keeps samples where the model is more uncertain (higher entropy).

    Args:
        model: A pre-trained model (e.g., the baseline model before fine-tuning).
        tokenizer: The tokenizer.
        dataset: The HuggingFace dataset to filter.
        keep_ratio: The fraction of the dataset to keep (e.g., 0.8 for 80%).
        batch_size: Batch size for inference.
        device: The device to run the model on.

    Returns:
        A new dataset containing only the filtered samples.
    """
    print(f"\n--- Starting Intelligent Data Filtering (Keeping top {keep_ratio*100:.0f}% uncertain samples) ---")
    model.to(device)
    model.eval()

    # We need a dataloader to process the data in batches
    # Ensure dataset is formatted for PyTorch and has the necessary columns
    try:
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "label", "token_type_ids"])
    except:
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    scores = []
    indices = []
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating sample scores"):
            # Move batch to device
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
            
            # Get model logits
            outputs = model(**inputs)
            logits = outputs.logits

            # Calculate entropy of the softmax distribution as the uncertainty score
            # Higher entropy means more uncertainty
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1)
            
            scores.extend(entropy.cpu().tolist())
            
            # Keep track of original indices
            batch_indices = list(range(sample_idx, sample_idx + len(entropy)))
            indices.extend(batch_indices)
            sample_idx += len(entropy)
            
    # Sort samples by score (descending) and get the indices of the top samples to keep
    sorted_pairs = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
    num_to_keep = int(len(sorted_pairs) * keep_ratio)
    indices_to_keep = [idx for idx, score in sorted_pairs[:num_to_keep]]
    
    # Select the top samples from the original dataset
    filtered_dataset = dataset.select(indices_to_keep)
    
    print(f"Filtering complete. Original size: {len(dataset)}, New size: {len(filtered_dataset)}")
    
    # Reset dataset format for subsequent processing
    filtered_dataset.reset_format()
    return filtered_dataset

# --- FLOPs Estimation ---
def estimate_flops(model, max_seq_length=MAX_SEQ_LENGTH):
    """
    Estimates FLOPs and parameters of a given model using thop (Robust version).
    """
    model_copy = copy.deepcopy(model).cpu()
    model_copy.eval()

    # 创建独立的 dummy tensor
    dummy_input_ids = torch.ones(1, max_seq_length, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_seq_length, dtype=torch.long)
    dummy_token_type_ids = torch.ones(1, max_seq_length, dtype=torch.long)
    
    print(f"Estimating FLOPs with dummy input shape: {dummy_input_ids.shape}")
    
    try:
        # --- 最终的、最可靠的调用方式 ---
        forward_args = model_copy.forward.__code__.co_varnames
    
        inputs_tuple = ()
        if 'input_ids' in forward_args:
            inputs_tuple += (dummy_input_ids,)
        if 'attention_mask' in forward_args:
            inputs_tuple += (dummy_attention_mask,)
        if 'token_type_ids' in forward_args:
            inputs_tuple += (dummy_token_type_ids,)

        flops, params = profile(model_copy, inputs=inputs_tuple, verbose=False)
    
    except Exception as e:
        print(f"Error during FLOPs estimation: {e}")
        print("Falling back to parameter count only.")
        flops, params = 0, sum(p.numel() for p in model_copy.parameters())

    return flops, params


def estimate_flops_reduction_single_unit(model, unit_name, max_seq_length=MAX_SEQ_LENGTH):
    """
    Estimates the FLOPs reduction if a single attention head or FFN neuron is pruned.
    This is a principled mathematical estimation, not by re-profiling.

    Args:
        model: The current model state.
        unit_name: String identifier like "layer.0.attention.head.0" or "layer.0.ffn.neuron.0".
        max_seq_length: The sequence length for calculation.

    Returns:
        Estimated FLOPs reduction (float).
    """
    config = model.config
    N = max_seq_length
    H = config.hidden_size
    
    try:
        if "attention.head" in unit_name:
            # FLOPs for one attention head
            num_heads = config.num_attention_heads
            head_dim = H // num_heads
            
            # Q, K, V projection FLOPs per head
            qkv_flops = 3 * (2 * N * H * head_dim)
            
            # Attention scores FLOPs per head (Q*K^T)
            attn_scores_flops = 2 * N * N * head_dim
            
            # Value aggregation FLOPs per head (Attn*V)
            value_agg_flops = 2 * N * N * head_dim
            
            # Output projection FLOPs attributable to one head
            output_proj_flops = 2 * N * H * head_dim
            
            # Total FLOPs for one head
            total_head_flops = qkv_flops + attn_scores_flops + value_agg_flops + output_proj_flops
            return total_head_flops

        elif "ffn.neuron" in unit_name:
            # FLOPs for one FFN neuron (in the intermediate layer)
            I = config.intermediate_size
            
            # FLOPs in the first linear layer (d_model -> d_ff) for one output neuron
            ffn1_flops = 2 * N * H
            
            # FLOPs in the second linear layer (d_ff -> d_model) for one input neuron
            ffn2_flops = 2 * N * H
            
            # Total FLOPs for one FFN neuron
            return ffn1_flops + ffn2_flops
            
    except Exception as e:
        # Fallback if config attributes are missing
        # print(f"Could not estimate FLOPs for {unit_name}, returning default. Error: {e}")
        return 1e6 # Return a default small value

    return 0

# --- Helper for Manual Evaluation ---
def manual_evaluate_model(model, dataloader, description="Evaluating Model", device="cuda"):
    model.to(device)
    model.eval()
    predictions, references = [], []
    print(f"\nPerforming manual evaluation on {description}...")
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=description):
                inputs = {}
                batch_size = batch.get('label', batch.get('input_ids', [])).size(0)

                if batch_size > 0:
                    # 【最终修复】: 同时处理 list 和 Tensor 两种情况
                    if 'input_ids' in batch and isinstance(batch['input_ids'], list):
                        inputs['input_ids'] = torch.cat(batch['input_ids']).view(batch_size, -1).to(device)
                    elif 'input_ids' in batch and isinstance(batch['input_ids'], torch.Tensor):
                        inputs['input_ids'] = batch['input_ids'].view(batch_size, -1).to(device)

                    if 'attention_mask' in batch and isinstance(batch['attention_mask'], list):
                        inputs['attention_mask'] = torch.cat(batch['attention_mask']).view(batch_size, -1).to(device)
                    elif 'attention_mask' in batch and isinstance(batch['attention_mask'], torch.Tensor):
                        inputs['attention_mask'] = batch['attention_mask'].view(batch_size, -1).to(device)

                    if 'token_type_ids' in batch and isinstance(batch['token_type_ids'], list):
                        inputs['token_type_ids'] = torch.cat(batch['token_type_ids']).view(batch_size, -1).to(device)
                    elif 'token_type_ids' in batch and isinstance(batch['token_type_ids'], torch.Tensor):
                        inputs['token_type_ids'] = batch['token_type_ids'].view(batch_size, -1).to(device)
                
                if not inputs or 'input_ids' not in inputs:
                    print(f"Warning: Skipping a batch in '{description}' due to missing 'input_ids'.")
                    continue
                
                config_obj = model.model.config if hasattr(model, 'model') else model.config
                if 'token_type_ids' in inputs and not hasattr(config_obj, 'type_vocab_size'):
                    del inputs['token_type_ids']
                    
                outputs = model(**inputs)

                if hasattr(outputs, 'logits'):
                    predictions.extend(outputs.logits.argmax(dim=-1).tolist())
                if "label" in batch and isinstance(batch["label"], torch.Tensor):
                    references.extend(batch["label"].tolist())

        accuracy = None
        if references:
             accuracy = accuracy_score(references, predictions)
             if accuracy is not None:
                print(f"  {description} Accuracy: {accuracy:.4f}")
        return accuracy
    except Exception as e:
        print(f"An unexpected error occurred during {description}: {e}")
        traceback.print_exc()
        return None

# --- Energy and Speed Measurement ---
def measure_energy_and_speed(model, dataloader, description="Measuring Model", device="cuda"):
    """
    Measures inference speed (FPS) and GPU energy consumption.
    Assumes model is already on the correct device (cuda).
    """
    model.to(device)
    model.eval() # Ensure model is in eval mode

    power_readings = []
    stop_power_thread = False
    power_thread = None

    # Function to read GPU power (run in a separate thread)
    def read_gpu_power_continuously(interval=0.5):
        """Continuously reads GPU power draw using nvidia-smi."""
        nonlocal stop_power_thread
        # Ensure this function is robust to nvidia-smi errors and stops cleanly.
        print(f"Starting GPU power monitoring thread (interval={interval}s)...")
        while not stop_power_thread:
            try:
                # Query power.draw (instantaneous power draw, W)
                # Query utilization.gpu (percentage of time over past second during which GPU was actively computing) - useful for debugging load
                # Query utilization.memory (percentage of time over past second during which global memory was being read/written)
                cmd = ['nvidia-smi', f'--id={GPU_INDEX}', '--query-gpu=power.draw', '--format=csv,noheader,nounits']
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                power = float(result.stdout.strip())
                power_readings.append(power)
                time.sleep(interval)
            except FileNotFoundError:
                 print("Error: nvidia-smi command not found. Cannot measure GPU power.")
                 stop_power_thread = True # Stop the thread
            except subprocess.CalledProcessError as e:
                 print(f"Error calling nvidia-smi: {e}")
                 stop_power_thread = True # Stop the thread
            except Exception as e:
                # print(f"Error in GPU power monitoring thread: {e}") # Optional: log other errors
                pass # Keep trying or add specific checks


        print("GPU power monitoring thread stopped.")


    # Start the power monitoring thread before inference
    if device == "cuda" and torch.cuda.is_available() and torch.cuda.device_count() > GPU_INDEX:
         import threading
         stop_power_thread = False
         power_thread = threading.Thread(target=read_gpu_power_continuously)
         power_thread.daemon = True # Allow thread to exit when main program exits
         power_thread.start()
         # Give the monitoring thread a moment to start collecting data
         time.sleep(1.0) # Adjust sleep time if needed
         if not power_thread.is_alive():
             print("Warning: GPU power monitoring thread failed to start. Power data will be unavailable.")
             power_thread = None # Reset if failed


    start_time = time.time()
    num_samples = 0
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=description):
            # 【最终修复】: 同时处理 list 和 Tensor 两种情况，并确保 batch_size > 0
            inputs = {}
            batch_size = batch.get('label', batch.get('input_ids', [])).size(0)
            
            if batch_size > 0:
                if 'input_ids' in batch and isinstance(batch['input_ids'], list):
                    inputs['input_ids'] = torch.cat(batch['input_ids']).view(batch_size, -1).to(device)
                elif 'input_ids' in batch and isinstance(batch['input_ids'], torch.Tensor):
                    inputs['input_ids'] = batch['input_ids'].view(batch_size, -1).to(device)

                if 'attention_mask' in batch and isinstance(batch['attention_mask'], list):
                    inputs['attention_mask'] = torch.cat(batch['attention_mask']).view(batch_size, -1).to(device)
                elif 'attention_mask' in batch and isinstance(batch['attention_mask'], torch.Tensor):
                    inputs['attention_mask'] = batch['attention_mask'].view(batch_size, -1).to(device)
            
                if 'token_type_ids' in batch and isinstance(batch['token_type_ids'], list):
                    inputs['token_type_ids'] = torch.cat(batch['token_type_ids']).view(batch_size, -1).to(device)
                elif 'token_type_ids' in batch and isinstance(batch['token_type_ids'], torch.Tensor):
                    inputs['token_type_ids'] = batch['token_type_ids'].view(batch_size, -1).to(device)

            if not inputs or 'input_ids' not in inputs:
                print(f"Warning: Skipping a batch in '{description}' due to missing 'input_ids'.")
                continue

            # 移除模型不需要的 token_type_ids
            config_obj = model.config
            if 'token_type_ids' in inputs and not hasattr(config_obj, 'type_vocab_size'):
                del inputs['token_type_ids']
                
            outputs = model(**inputs)
            if hasattr(outputs, 'logits'): predictions.extend(outputs.logits.argmax(dim=-1).tolist())
            if "label" in batch: references.extend(batch["label"].tolist())
            if 'input_ids' in inputs: num_samples += inputs["input_ids"].size(0)
    if device == "cuda": torch.cuda.synchronize()
    end_time = time.time()
    if power_thread is not None:
        stop_power_thread = True
        power_thread.join(timeout=2.0) # Wait for the thread to finish, with a timeout
        if power_thread.is_alive():
             print("Warning: Power monitoring thread did not stop cleanly.")


    duration = end_time - start_time

    # Calculate energy consumption
    avg_power = sum(power_readings) / len(power_readings) if power_readings else 0
    total_energy_joules = avg_power * duration
    total_energy_wh = total_energy_joules / 3600 # Convert Joules to Watt-hours

    # Calculate speed
    samples_per_second = num_samples / duration if duration > 0 else 0

    # Calculate energy per sample (convert Wh to uWh for smaller numbers)
    energy_per_sample_wh = total_energy_wh / num_samples if num_samples > 0 else 0
    energy_per_sample_uwh = energy_per_sample_wh * 1e6


    # Calculate accuracy if labels were collected
    accuracy = None
    if references and len(predictions) == len(references): # Ensure we have matching labels and predictions
         try:
             accuracy = accuracy_score(references, predictions)
         except Exception as e:
             print(f"Error calculating accuracy: {e}")
             accuracy = None


    metrics = {
        "num_samples": num_samples,
        "duration_s": duration,
        "avg_gpu_power_W": avg_power,
        "total_energy_Wh": total_energy_wh,
        "samples_per_second": samples_per_second,
        "energy_per_sample_uWh": energy_per_sample_uwh,
        "accuracy": accuracy # None if labels weren't collected or error occurred
    }

    print(f"\n--- {description} Results ---")
    print(f"  Processed {metrics['num_samples']} samples in {metrics['duration_s']:.2f} seconds")
    print(f"  Avg GPU Power: {metrics['avg_gpu_power_W']:.2f} W (Note: This is the whole GPU package power)")
    print(f"  Total Energy: {metrics['total_energy_Wh']:.6f} Wh")
    print(f"  Samples per Second (FPS): {metrics['samples_per_second']:.2f}")
    print(f"  Energy per Sample: {metrics['energy_per_sample_uWh']:.3f} uWh/sample")
    if accuracy is not None:
         print(f"  Accuracy: {accuracy:.4f}")
    else:
         print("  Accuracy: N/A (Could not calculate)")

    print("-" * (len(description) + 9)) # Match top border length

    return metrics


# --- Structural Pruning Helper (Conceptual/Placeholder) ---
def calculate_energy_aware_pruning_scores(model, alpha=0.1):
    """
    Calculates pruning scores for attention heads and FFN neurons based on an energy-aware criterion.
    The score is defined as: Importance / (FLOPs_Reduction ^ alpha)
    A lower score means higher pruning priority.

    Args:
        model: The model to analyze (BertForSequenceClassification).
        alpha: Weight for the energy-aware component.

    Returns:
        A dictionary mapping unit names to their scores.
    """
    scores = {}
    print(f"Calculating energy-aware pruning scores with alpha = {alpha}...")

    if not hasattr(model, 'bert'):
        print("Model structure does not match expected BERT structure.")
        return scores

    for layer_idx, layer in enumerate(tqdm(model.bert.encoder.layer, desc="Analyzing Layers")):
        # --- Attention Head Scores ---
        if hasattr(layer, 'attention'):
            attention_module = layer.attention.self
            num_heads = attention_module.num_attention_heads
            head_dim = attention_module.attention_head_size
            
            # Get weights for all heads
            q_w = attention_module.query.weight.data
            k_w = attention_module.key.weight.data
            v_w = attention_module.value.weight.data
            o_w = layer.attention.output.dense.weight.data
            
            for head_idx in range(num_heads):
                unit_name = f"layer.{layer_idx}.attention.head.{head_idx}"
                
                start, end = head_idx * head_dim, (head_idx + 1) * head_dim
                
                # Importance Metric: L1 norm of weights related to this head
                importance = q_w[start:end, :].abs().sum() + \
                             k_w[start:end, :].abs().sum() + \
                             v_w[start:end, :].abs().sum() + \
                             o_w[:, start:end].abs().sum()

                # Energy Proxy Metric: Estimated FLOPs reduction
                energy_proxy = estimate_flops_reduction_single_unit(model, unit_name)
                
                # Energy-Aware Score formula
                # Add a small epsilon to avoid division by zero
                score = importance / ((energy_proxy ** alpha) + 1e-9)
                scores[unit_name] = score.item()

        # --- FFN Neuron Scores ---
        if hasattr(layer, 'intermediate'):
            intermediate_dense = layer.intermediate.dense
            output_dense = layer.output.dense
            d_ff = intermediate_dense.out_features

            inter_w = intermediate_dense.weight.data
            out_w = output_dense.weight.data
            
            for neuron_idx in range(d_ff):
                unit_name = f"layer.{layer_idx}.ffn.neuron.{neuron_idx}"

                # Importance Metric: L1 norm of weights
                # Weight from d_model -> neuron_idx and neuron_idx -> d_model
                importance = inter_w[neuron_idx, :].abs().sum() + \
                             out_w[:, neuron_idx].abs().sum()

                # Energy Proxy Metric: Estimated FLOPs reduction
                energy_proxy = estimate_flops_reduction_single_unit(model, unit_name)
                
                # Energy-Aware Score formula
                score = importance / ((energy_proxy ** alpha) + 1e-9)
                scores[unit_name] = score.item()
    
    print(f"Finished calculating scores for {len(scores)} units.")
    return scores

def calculate_l1_norm_scores(model):
    """
    Calculates pruning scores for attention heads and FFN neurons based on L1 norm.
    This is the "traditional" importance-based method for comparison.
    A lower score means higher pruning priority.

    Args:
        model: The model to analyze (BertForSequenceClassification).

    Returns:
        A dictionary mapping unit names to their scores.
    """
    scores = {}
    print("Calculating traditional L1 norm pruning scores...")

    if not hasattr(model, 'bert'):
        print("Model structure does not match expected BERT structure.")
        return scores

    # 为了让分数更稳定，我们让模型在CPU上
    model.cpu() 
    
    for layer_idx, layer in enumerate(tqdm(model.bert.encoder.layer, desc="Analyzing Layers for L1 Norm")):
        # --- Attention Head Scores ---
        if hasattr(layer, 'attention'):
            attention_module = layer.attention.self
            num_heads = attention_module.num_attention_heads
            head_dim = attention_module.attention_head_size
            
            q_w = attention_module.query.weight.data
            k_w = attention_module.key.weight.data
            v_w = attention_module.value.weight.data
            o_w = layer.attention.output.dense.weight.data
            
            for head_idx in range(num_heads):
                unit_name = f"layer.{layer_idx}.attention.head.{head_idx}"
                start, end = head_idx * head_dim, (head_idx + 1) * head_dim
                
                # Importance Metric: L1 norm of weights related to this head
                # We return the importance directly. In the pruning function, we'll sort by this value.
                importance = (
                    q_w[start:end, :].abs().sum() + 
                    k_w[start:end, :].abs().sum() + 
                    v_w[start:end, :].abs().sum() + 
                    o_w[:, start:end].abs().sum()
                )
                scores[unit_name] = importance.item() # 使用.item()获取纯数值

        # --- FFN Neuron Scores ---
        if hasattr(layer, 'intermediate'):
            intermediate_dense = layer.intermediate.dense
            output_dense = layer.output.dense
            d_ff = intermediate_dense.out_features

            inter_w = intermediate_dense.weight.data
            out_w = output_dense.weight.data
            
            for neuron_idx in range(d_ff):
                unit_name = f"layer.{layer_idx}.ffn.neuron.{neuron_idx}"

                # Importance Metric: L1 norm of weights
                importance = (
                    inter_w[neuron_idx, :].abs().sum() + 
                    out_w[:, neuron_idx].abs().sum()
                )
                scores[unit_name] = importance.item()
    
    print(f"Finished calculating L1 norm scores for {len(scores)} units.")
    # 将模型移回GPU，以防后续需要
    model.cuda()
    return scores


def perform_structural_pruning(model, scores, target_flops_reduction_ratio, max_seq_length=MAX_SEQ_LENGTH):
    """
    Performs structural pruning on the model based on scores and target FLOPs reduction.
    (Robust version using DependencyGraph)
    """
    pruned_model = model
    pruned_model.cpu()

    initial_flops, initial_params = estimate_flops(pruned_model, max_seq_length)
    if initial_flops == 0:
        print("Error: Initial FLOPs are zero. Cannot perform pruning.")
        return pruned_model.cuda(), {}
        
    target_flops = initial_flops * (1 - target_flops_reduction_ratio)
    print(f"Initial FLOPs: {initial_flops/1e9:.2f} GFLOPs, Target FLOPs: {target_flops/1e9:.2f} GFLOPs")

    if not scores:
        print("No pruning scores provided. Skipping pruning.")
        return pruned_model.cuda(), {}

    sorted_units = sorted(scores.items(), key=lambda item: item[1])

    # --- 关键修改：使用更底层的DependencyGraph ---
    # 1. 创建虚拟输入
    dummy_input_ids = torch.ones(1, max_seq_length, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_seq_length, dtype=torch.long)
    dummy_token_type_ids = torch.ones(1, max_seq_length, dtype=torch.long)
    example_inputs = (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids)

    # 2. 构建依赖图
    DG = tp.DependencyGraph()
    DG.build_dependency(pruned_model, example_inputs=example_inputs)
    # -----------------------------------------------
    
    current_flops = initial_flops
    pruned_units = {}

    for unit_name, score in tqdm(sorted_units, desc="Performing Structural Pruning"):
        if current_flops <= target_flops:
            print("\nTarget FLOPs reduction reached. Stopping pruning.")
            break
            
        try:
            parts = unit_name.split('.')
            layer_idx, unit_type, unit_idx = int(parts[1]), parts[2], int(parts[-1])
            
            pruning_indices = []
            layer_to_prune = None

            if unit_type == "attention":
                layer_to_prune = pruned_model.bert.encoder.layer[layer_idx].attention.self.query
                head_dim = pruned_model.config.hidden_size // pruned_model.config.num_attention_heads
                pruning_indices = list(range(unit_idx * head_dim, (unit_idx + 1) * head_dim))
                
            elif unit_type == "ffn":
                layer_to_prune = pruned_model.bert.encoder.layer[layer_idx].intermediate.dense
                pruning_indices = [unit_idx]

            if layer_to_prune is not None and len(pruning_indices) > 0:
                # --- 关键修改：使用DG进行剪枝 ---
                group = DG.get_pruning_group(layer_to_prune, tp.prune_linear_out_channels, idxs=pruning_indices)
                if DG.check_pruning_group(group):
                    group.prune()
                # -----------------------------------
            else:
                continue

            pruned_units[unit_name] = score
            flops_reduction = estimate_flops_reduction_single_unit(pruned_model, unit_name, max_seq_length)
            current_flops -= flops_reduction

        except Exception as e:
            continue
            
    final_flops, final_params = estimate_flops(pruned_model, max_seq_length)
    print("\nStructural pruning finished.")
    print(f"Final Estimated FLOPs: {final_flops/1e9:.2f} GFLOPs, Final Params: {final_params:,}")
    reduction_pct = (initial_flops - final_flops) / initial_flops * 100 if initial_flops > 0 else 0
    print(f"Achieved FLOPs Reduction: {reduction_pct:.2f}%")

    return pruned_model.cuda(), pruned_units


def define_mixed_precision_qconfig_strategy(model):
    """
    Defines the quantization configuration for a standard HuggingFace model.
    """
    print("Applying mixed precision quantization strategy...")
    model_with_qconfig = copy.deepcopy(model)
    qconfig_int8 = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # 直接遍历模型即可
    for name, module in model_with_qconfig.named_modules():
         if isinstance(module, nn.Linear):
              if 'bert.encoder.layer' in name or 'classifier' in name:
                  module.qconfig = qconfig_int8
              else:
                   module.qconfig = None # Keep pooler as FP32 for safety
         elif isinstance(module, (nn.Embedding, nn.LayerNorm)):
              module.qconfig = None

    print("Qconfig applied successfully.")
    return model_with_qconfig

# Note: Fusion of modules is often done before prepare_qat for better results,
# but the fusion patterns can be complex for BERT. Skipping for simplicity in framework.
# If you need fusion, refer to PyTorch quantization tutorials.
# torch.quantization.fuse_modules(model_prepared_for_qat, bert_modules_to_fuse, inplace=True)

# Function to convert QAT prepared model to quantized model (from PyTorch docs)
def convert_model_to_quantized(model_prepared_for_qat):
    """
    Converts a QAT-prepared model to a fully quantized model.
    Usually done after QAT training.
    """
    # Ensure model is in eval mode before conversion
    model_prepared_for_qat.eval()

    # Recommended: Convert on CPU first, then move to GPU if needed for measurement/deployment
    # This avoids potential issues with GPU conversion backends in some versions.
    # However, for GPU measurement, we need it on GPU.
    # Let's try converting on GPU if the prepared model is on GPU.
    device = next(model_prepared_for_qat.parameters()).device
    if device.type == 'cuda':
        print("Attempting conversion on GPU...")
        # Convert can sometimes be tricky directly on GPU depending on PyTorch version and backend.
        # If this fails, try: model_prepared_for_qat.cpu(); quantized_model = torch.quantization.convert(model_prepared_for_qat); quantized_model.cuda()
        quantized_model = torch.quantization.convert(model_prepared_for_qat, inplace=False)

    else: # Model is on CPU
        print("Converting on CPU...")
        quantized_model = torch.quantization.convert(model_prepared_for_qat, inplace=False)
        # If measuring on GPU, remember to move it: quantized_model.cuda()

    return quantized_model