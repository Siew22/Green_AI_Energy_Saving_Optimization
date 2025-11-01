# main_research_scripts.py
import sys # Import sys at the very top
import os
os.environ["WANDB_PROJECT"] = "GreenAI-Optimization-FYP" # 在脚本顶部设置项目名
import json # Needed for saving/loading pruning info
import copy # Needed for deepcopy
import time # Needed for time measurement helpers

import torch
import torch.nn as nn
# These imports are where the TypeError happened. We will check their version/path just after import.
from transformers import BertForSequenceClassification, BertTokenizerFast, TrainingArguments, Trainer
from datasets import load_dataset
from tqdm import tqdm
from torch.ao.quantization.fake_quantize import FakeQuantize
import contextlib

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
# No need for copy or time here if imported from utils

# Import your utility functions
# Ensure utils.py exists and contains the necessary function definitions (even with TODOs)
try:
    from utils import estimate_flops, measure_energy_and_speed, \
                      calculate_energy_aware_pruning_scores, perform_structural_pruning, \
                      define_mixed_precision_qconfig_strategy, convert_model_to_quantized, \
                      MAX_SEQ_LENGTH # Import MAX_SEQ_LENGTH from utils
except ImportError as e:
    print(f"Error importing from utils.py: {e}")
    print("Please ensure utils.py exists in the same directory and contains all required functions.")
    print("If utils.py is empty or incomplete, copy the framework code into it.")
    sys.exit(1)


# --- DIAGNOSTIC PRINTS ---
print("--- Script execution started ---")
print(f"Running from Python executable: {sys.executable}") # Print the Python executable path
print(f"Working directory: {os.getcwd()}")
print(f"Python search path (sys.path):")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}") # Print with index

try:
    import transformers
    print(f"\nTransformers Version (in script): {transformers.__version__}")
    print(f"Transformers Path (in script): {os.path.dirname(transformers.__file__)}")
except ImportError as e:
    print(f"\nError importing transformers: {e}")
    print("Please ensure transformers is installed in your environment.")
    sys.exit(1) # Exit if essential library can't be imported


try:
    import torch
    print(f"Torch Version (in script): {torch.__version__}")
    print(f"Torch Path (in script): {os.path.dirname(torch.__file__)}")
    print(f"CUDA Available (in script): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
         # Corrected f-string syntax for GPU Name
         print(f"GPU Name (in script): {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
except ImportError as e:
    print(f"Error importing torch: {e}")
    print("Please ensure PyTorch is installed in your environment.")
    sys.exit(1) # Exit if essential library can't be imported


try:
     import datasets
     print(f"Datasets Version (in script): {datasets.__version__}")
     print(f"Datasets Path (in script): {os.path.dirname(datasets.__file__)}")
except ImportError as e:
     print(f"Error importing datasets: {e}")
     print("Please ensure datasets is installed in your environment.")
     sys.exit(1) # Exit if essential library can't be imported


print("-" * 30) # Separator
# --- END DIAGNOSTIC PRINTS ---


# --- Configuration ---
MODEL_NAME = "bert-base-uncased"
DEVICE_BATCH_SIZE = 2  # 这是你的硬件一次能处理的最大批量
EFFECTIVE_BATCH_SIZE = 4 # 这是我们希望在数学上达到的批量大小
assert EFFECTIVE_BATCH_SIZE % DEVICE_BATCH_SIZE == 0
GRADIENT_ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // DEVICE_BATCH_SIZE
DATASET_NAME = "glue"
TASK_NAME = "sst2" # Example task
# MAX_SEQ_LENGTH is imported from utils
BATCH_SIZE = 2 # Adjusted batch size for RTX 4050 (6GB) - May need further adjustment. Reduce further if OOM.
BASE_OUTPUT_DIR = "./results"
NUM_EPOCHS_BASELINE = 3 # Fine-tune baseline for a few epochs
NUM_EPOCHS_PRUNING_FINETUNE = 5 # More epochs for recovery after pruning
NUM_EPOCHS_QAT = 3 # Fewer epochs for QAT
PRUNING_FLOPs_REDUCTION_TARGET = 0.4 # Example: Target 40% FLOPs reduction
ENERGY_AWARE_ALPHA = 0.1 # Hyperparameter for your pruning score (Adjust this during research)

# --- Helper for Manual Evaluation ---
def manual_evaluate_model(model, dataloader, description="Evaluating Model"):
    """Manually evaluates model accuracy on a given dataloader."""
    model.eval()
    predictions = []
    references = []
    print(f"\nPerforming manual evaluation on {description}...")
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=description):
                # Move batch to GPU. Handle potential missing keys.
                batch = {k: v.cuda() for k, v in batch.items() if isinstance(k, str) and isinstance(v, torch.Tensor)}

                # Perform inference. Adjust arguments based on model's forward method.
                model_inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
                if 'token_type_ids' in batch:
                     model_inputs['token_type_ids'] = batch['token_type_ids']

                outputs = model(**model_inputs)

                # Collect predictions and references
                if hasattr(outputs, 'logits'):
                     predictions.extend(outputs.logits.argmax(dim=-1).tolist())
                if "label" in batch:
                    references.extend(batch["label"].tolist())

        # Calculate accuracy
        accuracy = None
        if references and len(predictions) == len(references):
             try:
                 accuracy = accuracy_score(references, predictions)
                 print(f"  {description} Accuracy: {accuracy:.4f}")
             except Exception as e:
                 print(f"Error calculating accuracy for {description}: {e}")

        return accuracy

    except Exception as e:
        print(f"Error during manual evaluation of {description}: {e}")
        return None # Return None if evaluation fails

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

@contextlib.contextmanager
def force_fp32_context():
    """A context manager to force FP32 operations."""
    with torch.amp.autocast('cuda', enabled=False):
        yield

class UltimateFp32Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        fp32_inputs = self._ensure_fp32_inputs(inputs)
        
        with force_fp32_context():
            is_gc_enabled = getattr(model.config, "use_cache", False) is False and getattr(model, "supports_gradient_checkpointing", False)
            if is_gc_enabled:
                model.gradient_checkpointing_disable()

            try:
                outputs = model(**fp32_inputs)
            finally:
                if is_gc_enabled:
                    model.gradient_checkpointing_enable()

        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def _ensure_fp32_inputs(self, inputs):
        fp32_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                fp32_inputs[k] = v.float()
            else:
                fp32_inputs[k] = v
        
        if "labels" not in fp32_inputs and "label" in fp32_inputs:
            fp32_inputs["labels"] = fp32_inputs.pop("label")
        return fp32_inputs
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        fp32_inputs = self._ensure_fp32_inputs(inputs)
        with force_fp32_context():
            return super().prediction_step(model, fp32_inputs, prediction_loss_only, ignore_keys)

def prepare_ultimate_qat_model(model):
    print("Preparing model with ultimate QAT FP32 control...")
    qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model_prepared = copy.deepcopy(model)
    model_prepared.train()
    
    def set_qconfig(mod):
        if isinstance(mod, nn.Linear):
            mod.qconfig = qconfig
    
    model_prepared.apply(set_qconfig)
    
    print("Preparing model for QAT...")
    torch.quantization.prepare_qat(model_prepared, inplace=True)
    
    print("Disabling gradient checkpointing during QAT to ensure stability.")
    if hasattr(model_prepared, 'gradient_checkpointing_enable'):
        model_prepared.gradient_checkpointing_disable()

    print("Forcing model to FP32...")
    model_prepared.float()
    model_prepared.cuda()
    print("Ultimate QAT model preparation completed successfully!")
    return model_prepared

def create_ultimate_training_args(base_output_dir, num_epochs, grad_accum_steps):
    return TrainingArguments(
        output_dir=f"{base_output_dir}/qat_pruned",
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=False, bf16=False, tf32=False,
        remove_unused_columns=False,
        report_to="wandb",
        run_name="ultimate-qat-finetune",
        optim="adamw_torch"
    )

# 2. 修复的define_mixed_precision_qconfig_strategy函数
def define_mixed_precision_qconfig_strategy(model):
    """定义明确指定FP32的QConfig策略"""
    print("Defining mixed precision QAT strategy with explicit FP32 QConfig...")
    
    # This qconfig definition is for demonstration. PyTorch's default QAT qconfig
    # already works in FP32. The main issue is the interaction with AMP,
    # which the Fp32InputsTrainer solves. We will use the standard default qconfig
    # as it's well-tested, and rely on the trainer to enforce FP32.
    qconfig_int8 = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    model_with_qconfig = copy.deepcopy(model)

    for name, module in model_with_qconfig.named_modules():
         if isinstance(module, nn.Linear):
              if 'bert.encoder.layer' in name or 'classifier' in name:
                  module.qconfig = qconfig_int8
              else:
                   module.qconfig = None
         elif isinstance(module, (nn.Embedding, nn.LayerNorm, nn.GELU, nn.Dropout)):
              module.qconfig = None
              
    return model_with_qconfig

# 3. 深度FP32转换和验证函数 (合并为一个)
def force_and_verify_fp32_conversion(model, stage=""):
    """深度转换模型为FP32并验证"""
    print(f"--- Performing and Verifying Deep FP32 Conversion ({stage}) ---")
    
    model.float()
    
    def _convert_observer(observer):
        try:
            if hasattr(observer, 'dtype'):
                observer.dtype = torch.float32
            for attr_name in dir(observer):
                if not attr_name.startswith('_'):
                    attr_val = getattr(observer, attr_name)
                    if isinstance(attr_val, torch.Tensor) and attr_val.is_floating_point():
                        setattr(observer, attr_name, attr_val.float())
        except Exception:
            pass
            
    for module in model.modules():
        if hasattr(module, 'weight') and module.weight is not None and module.weight.is_floating_point():
            module.weight.data = module.weight.data.float()
        if hasattr(module, 'bias') and module.bias is not None and module.bias.is_floating_point():
            module.bias.data = module.bias.data.float()
        if hasattr(module, 'activation_post_process') and module.activation_post_process is not None:
            _convert_observer(module.activation_post_process)
        if 'FakeQuantize' in str(type(module)) or 'Observer' in str(type(module)):
            _convert_observer(module)
            
    # Verification
    issues_found = 0
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            print(f"  WARNING: Parameter '{name}' is {param.dtype}, not FP32")
            issues_found += 1
    for name, buf in model.named_buffers():
        if buf.is_floating_point() and buf.dtype != torch.float32:
            print(f"  WARNING: Buffer '{name}' is {buf.dtype}, not FP32")
            issues_found += 1
    
    if issues_found == 0:
        print("  ✓ Deep FP32 conversion verification successful!")
        return True
    else:
        print(f"  ⚠ Found {issues_found} issues in FP32 conversion")
        return False

# 4. 替换的QAT准备代码
def prepare_qat_model_robust(model):
    """更健壮的QAT模型准备函数"""
    print("Preparing model for QAT with robust FP32 configuration...")
    
    model.train()
    
    # 1. 应用QConfig策略
    model_with_qconfig = define_mixed_precision_qconfig_strategy(model)
    
    # 2. 准备QAT
    model_prepared = torch.quantization.prepare_qat(model_with_qconfig, inplace=False)
    
    # 3. 启用梯度检查点（如果需要）
    try:
        model_prepared.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled to save VRAM.")
    except Exception:
        print("Warning: Model does not support gradient checkpointing, proceeding without it.")

    # 4. 深度FP32转换和验证
    force_and_verify_fp32_conversion(model_prepared, stage="After QAT Prepare")
    
    # 5. 移动到GPU
    model_prepared = model_prepared.cuda()
    torch.cuda.empty_cache()
    
    print("QAT model preparation completed with FP32 enforcement.")
    return model_prepared

# ============================================================================
# ===== END OF YOUR PROPOSED FIX: ADVANCED QAT & FP32 HANDLING FUNCTIONS =====
# ============================================================================


# --- Helper for Manual Evaluation ---
# (Your existing manual_evaluate_model and compute_metrics functions go here, no changes needed)
def manual_evaluate_model(model, dataloader, description="Evaluating Model"):
    model.eval()
    predictions, references = [], []
    print(f"\nPerforming manual evaluation on {description}...")
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=description):
                batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                model_inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
                if 'token_type_ids' in batch:
                     model_inputs['token_type_ids'] = batch['token_type_ids']
                outputs = model(**model_inputs)
                if hasattr(outputs, 'logits'):
                     predictions.extend(outputs.logits.argmax(dim=-1).tolist())
                if "label" in batch:
                    references.extend(batch["label"].tolist())
        accuracy = accuracy_score(references, predictions) if references else None
        if accuracy is not None:
            print(f"  {description} Accuracy: {accuracy:.4f}")
        return accuracy
    except Exception as e:
        print(f"Error during manual evaluation of {description}: {e}")
        return None

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# --- Main Research Workflow ---
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # --- Stage 1: Environment Setup & Baseline Establishment ---
    print("\n--- Stage 1: Setup & Baseline ---")
    # Load dataset and tokenizer
    try:
        dataset = load_dataset(DATASET_NAME, TASK_NAME)
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
        print("Dataset and tokenizer loaded.")
    except Exception as e:
        print(f"Error loading dataset or tokenizer: {e}")
        print("Please check dataset name, task name, and internet connection.")
        sys.exit(1)

    def tokenize_function(examples):
         # Handle potential missing fields if dataset structure varies
         text_col = "sentence" if "sentence" in examples else ("text" if "text" in examples else None)
         if text_col is None:
             raise ValueError("Dataset does not contain 'sentence' or 'text' column.")
         return tokenizer(examples[text_col], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)


    print("Tokenizing datasets...")
    try:
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        # Ensure the column names match what the Trainer expects and your model forward expects
        # For BertForSequenceClassification, it's typically input_ids, attention_mask, token_type_ids, and label
        # Check if 'token_type_ids' is present in the tokenized data, add it to columns if so.
        columns_to_keep = ["input_ids", "attention_mask", "label"]
        if "token_type_ids" in tokenized_datasets["train"].features:
            columns_to_keep.append("token_type_ids")

        tokenized_datasets.set_format("torch", columns=columns_to_keep)
        print("Dataset tokenization and formatting complete.")
    except Exception as e:
        print(f"Error tokenizing or formatting datasets: {e}")
        sys.exit(1)


    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=DEVICE_BATCH_SIZE)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=BATCH_SIZE)
    #test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=BATCH_SIZE) # Use test for final measurement

    # Load baseline model
    print(f"\nLoading baseline model: {MODEL_NAME}")
    try:
        num_labels = dataset['train'].features['label'].num_classes
        baseline_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
        baseline_model.cuda() # Move to GPU
        print("Baseline model loaded and moved to GPU.")
    except Exception as e:
        print(f"Error loading or moving baseline model to GPU: {e}")
        print("Please ensure PyTorch GPU is correctly installed and GPU is available.")
        sys.exit(1) # Exit if baseline setup fails

    # --- NEW: Perform Intelligent Data Filtering on the training set ---
    # We use the raw pre-trained model to score the data before fine-tuning
    # This assumes the pre-trained model has some general knowledge to assess sample difficulty
    from utils import filter_dataset_intelligently # Make sure to import
    original_train_dataset = tokenized_datasets["train"]
    filtered_train_dataset = filter_dataset_intelligently(
        model=baseline_model, 
        tokenizer=tokenizer,
        dataset=original_train_dataset,
        keep_ratio=0.8, # Example: keep 80% of the most valuable data
        batch_size=BATCH_SIZE,
        device="cuda"
        )
    tokenized_datasets["train"] = filtered_train_dataset # Replace the original training set
    
    # Now, the 'train_dataloader' and 'Trainer' will use the smaller, more efficient dataset
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=BATCH_SIZE)

    # Fine-tune baseline model - MANUAL EVAL/SAVE
    print("\nFine-tuning baseline model (Manual Eval/Save)...")
    # *** TrainingArguments Initialization - Manual Eval/Save ***
    # Removed all evaluation, save, report, load_best parameters due to persistent TypeError
    try:
        baseline_training_args = TrainingArguments(
            output_dir=f"{BASE_OUTPUT_DIR}/baseline",
            learning_rate=2e-5,
            per_device_train_batch_size=DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            # per_device_eval_batch_size is not strictly needed in TrainingArguments if not using Trainer.evaluate internally,
            # but keeping it for potential future use or consistency with Trainer.
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS_BASELINE,
            weight_decay=0.01,
            save_strategy="epoch",
            save_total_limit=1,
            fp16=True,
            report_to="wandb",
            run_name="baseline-finetune",
            optim="adamw_bnb_8bit"
            # Removed all eval/save/report parameters due to persistent errors in this environment
        )
        print("TrainingArguments initialized successfully (Manual Eval/Save).")
    except TypeError as e:
        print(f"\n--- CRITICAL ERROR: TypeError during TrainingArguments initialization ---")
        print(f"Error details: {e}")
        print("TrainingArguments still failing even after removing eval/save/report/load_best parameters.")
        print("This indicates a very deep and unresolvable environment conflict with the transformers library.")
        print("Further remote debugging is not possible for this specific environment issue.")
        sys.exit(1) # Exit if TrainingArguments still fails

    # If TrainingArguments was initialized, proceed with training
    # Trainer will only perform training steps now.
    baseline_trainer = Trainer(
        model=baseline_model,
        args=baseline_training_args,
        train_dataset=tokenized_datasets["train"],
        # eval_dataset is not needed in Trainer if not using evaluation_strategy
        # eval_dataset=tokenized_datasets["validation"], # Not used by trainer with manual eval
        compute_metrics=compute_metrics, # Not used by trainer with manual eval
        # data_collator=DataCollatorWithPadding(tokenizer=tokenizer), # Use if needed
    )

    try:
        print("Starting baseline model training...")
        baseline_trainer.train()
        print("Baseline model training complete.")

        # --- Manual Evaluation and Saving After Training ---
        print("\nPerforming manual evaluation and saving after baseline training...")
        # Evaluate on validation set manually
        final_baseline_val_accuracy = manual_evaluate_model(
            baseline_model, eval_dataloader, description="Baseline Model Final Validation"
        )

        # Save the final model after last epoch (Trainer already saves checkpoints, save_model saves final)
        final_baseline_save_path = f"{BASE_OUTPUT_DIR}/fine_tuned_baseline_model_final"
        baseline_trainer.save_model(final_baseline_save_path) # Saves the model currently loaded in trainer
        print(f"Final baseline model saved to {final_baseline_save_path}")

        # Store baseline metrics for reporting later
        baseline_metrics = {}
        baseline_metrics["accuracy"] = final_baseline_val_accuracy # Use validation accuracy for comparison
        # We'll measure energy/speed separately below

    except Exception as e:
        print(f"Error during baseline model training or manual eval/save: {e}")
        print("Common causes: CUDA out of memory (reduce BATCH_SIZE), model/data issues.")
        sys.exit(1)


    # Measure baseline performance and energy AFTER training and saving
    print("\nMeasuring baseline model...")
    # Reload the *saved* final model for consistent measurement
    try:
        baseline_model_loaded = BertForSequenceClassification.from_pretrained(final_baseline_save_path).cuda()
        print("Baseline model reloaded for measurement.")
    except Exception as e:
        print(f"Error reloading baseline model for measurement: {e}")
        sys.exit(1)

    # Ensure test_dataloader has correct format for measurement function
    try:
        baseline_metrics_measurement = measure_energy_and_speed(baseline_model_loaded, eval_dataloader, description="Baseline Model Inference on Test Set")
        # Add measurement metrics to baseline_metrics
        baseline_metrics.update(baseline_metrics_measurement)

    except Exception as e:
        print(f"Error during baseline energy/speed measurement: {e}")
        # Populate with default values if measurement fails
        measurement_defaults = {
            "num_samples": len(eval_dataloader.dataset), "duration_s": 0, "avg_gpu_power_W": 0,
            "total_energy_Wh": 0, "samples_per_second": 0, "energy_per_sample_uWh": 0,
        }
        baseline_metrics.update(measurement_defaults)
        print("Baseline Measurement failed. Metrics set to default/None.")


    # Estimate FLOPs and params
    try:
        baseline_flops, baseline_params = estimate_flops(baseline_model_loaded, MAX_SEQ_LENGTH)
        baseline_metrics["flops"] = baseline_flops
        baseline_metrics["params"] = baseline_params
        print(f"Baseline Params: {baseline_metrics['params']:,}")
        # Convert FLOPs to GFLOPs for display
        print(f"Baseline FLOPs: {baseline_metrics['flops']/1e9:.2f} GFLOPs (Theoretical)")
    except Exception as e:
        print(f"Error estimating baseline FLOPs/Params: {e}")
        baseline_metrics["flops"] = None
        baseline_metrics["params"] = None


    # --- Stage 2: Energy-Aware Pruning ---
    print("\n--- Stage 2: Energy-Aware Pruning ---")
    # Load the fine-tuned baseline model for pruning
    try:
        model_to_prune = BertForSequenceClassification.from_pretrained(final_baseline_save_path).cuda()
        print("Model loaded for pruning.")
    except Exception as e:
        print(f"Error loading model for pruning: {e}")
        sys.exit(1)


    # *** TODO: Implement calculate_energy_aware_pruning_scores in utils.py ***
    # This is where you define your custom criterion. Ensure it returns a dictionary of scores.
    print("\nCalculating energy-aware pruning scores (Requires utils.py implementation)...")
    pruning_scores = {} # Initialize outside try block
    try:
        pruning_scores = calculate_energy_aware_pruning_scores(model_to_prune, alpha=ENERGY_AWARE_ALPHA)
        print(f"Calculated scores for {len(pruning_scores)} units.")
        if not pruning_scores:
            print("Warning: calculate_energy_aware_pruning_scores returned no scores. Pruning will be skipped.")
    except Exception as e:
        print(f"Error calculating pruning scores: {e}")
        print("Please check your implementation of calculate_energy_aware_pruning_scores.")
        pruning_scores = {} # Proceed with empty scores (no pruning)


    # *** TODO: Implement perform_structural_pruning in utils.py ***
    # This is the complex structural modification step. Ensure it returns the modified model and info.
    print(f"\nPerforming structural pruning to target {PRUNING_FLOPs_REDUCTION_TARGET * 100:.0f}% FLOPs reduction (Requires utils.py implementation)...")
    pruned_model = None # Initialize outside try block
    pruned_units_info = {} # Initialize outside try block
    try:
        # Passing model_to_prune; perform_structural_pruning should make a copy
        pruned_model, pruned_units_info = perform_structural_pruning(model_to_prune, pruning_scores, PRUNING_FLOPs_REDUCTION_TARGET, MAX_SEQ_LENGTH)
        print("Structural pruning step finished.")
        if not pruned_units_info:
            print("Warning: perform_structural_pruning did not report any units pruned.")
            # If no units were pruned, pruned_model is likely a copy of model_to_prune.
    except Exception as e:
        print(f"Error during structural pruning: {e}")
        print("Please check your implementation of perform_structural_pruning.")
        # If pruning fails, maybe fallback to using the unpruned model for subsequent stages
        # or exit? For now, let's exit as pruning is a core step.
        sys.exit(1)


    # Save the pruned model structure and weights
    # This requires custom logic if your pruning modified the architecture in a way
    # that BertForSequenceClassification.from_pretrained cannot load directly.
    # Saving state_dict and knowing how to rebuild the modified structure when loading is crucial.
    pruned_model_save_path = f"{BASE_OUTPUT_DIR}/pruned_model"
    pruned_model_state_dict_path = f"{BASE_OUTPUT_DIR}/pruned_model.pt"
    try:
        # Attempt saving with HuggingFace method first
        pruned_model.save_pretrained(pruned_model_save_path)
        print(f"Pruned model saved using save_pretrained to {pruned_model_save_path}")
        # Save state_dict as a fallback/alternative
        torch.save(pruned_model.state_dict(), pruned_model_state_dict_path)
        print(f"Pruned model state_dict saved to {pruned_model_state_dict_path}")
        # TODO: Save info about WHICH units were pruned (pruned_units_info)
        # This is needed to rebuild the structure if save_pretrained fails later.
        pruned_units_info_path = f"{BASE_OUTPUT_DIR}/pruned_units_info.json"
        try:
            with open(pruned_units_info_path, 'w') as f:
                json.dump(pruned_units_info, f, indent=4) # Use indent for readability
            print(f"Pruned units info saved to {pruned_units_info_path}")
        except Exception as json_e:
            print(f"Warning: Could not save pruned_units_info.json: {json_e}")
            # <--- 内存清理的最佳位置 ---
            print("\nCleaning up memory after pruning...")
            if 'model_to_prune' in locals():
                del model_to_prune, pruned_model
            if 'pruned_model' in locals():
                del pruned_model
            torch.cuda.empty_cache()
            print("Memory cleaned.")
            # --- 内存清理结束 ---

    except Exception as e:
        print(f"Warning: Could not save pruned model using save_pretrained: {e}")
        print("You likely need a custom save/load logic for structural pruning.")
        print(f"State dict was saved to {pruned_model_state_dict_path}")


    # --- Stage 3: Fine-tuning Pruned Model (Manual Eval/Save) ---
    print("\n--- Stage 3: Fine-tuning Pruned Model (Manual Eval/Save) ---")
    
    pruned_model_for_finetuning = None
    try:
        print(f"Loading pruned model for fine-tuning from {pruned_model_save_path}...")
        
        # --- 关键的自定义加载逻辑 ---
        # 1. 首先，只加载被修改后的配置文件
        from transformers import AutoConfig
        pruned_config = AutoConfig.from_pretrained(pruned_model_save_path)
        
        # 2. 然后，使用这个新的配置来初始化一个结构正确的“空壳”模型
        # from_pretrained会先下载原始bert-base，然后再用你的新config覆盖它
        print("Initializing model with the new pruned configuration...")
        pruned_model_for_finetuning = BertForSequenceClassification.from_pretrained(
            MODEL_NAME, # 仍然从原始模型开始
            config=pruned_config # <--- 但强制使用我们剪枝后的新配置
        )
        
        # 3. 最后，将剪枝后的权重加载到这个结构正确的模型中
        # (这一步通常在from_pretrained内部自动完成，但我们可以手动再确认一次)
        # 如果上一步失败，可以尝试手动加载state_dict
        print("Loading pruned weights into the new structure...")
        # (通常上一步已经完成了加载，如果仍然失败，可以取消下面的注释)
        # state_dict = torch.load(os.path.join(pruned_model_save_path, 'pytorch_model.bin'))
        # pruned_model_for_finetuning.load_state_dict(state_dict)

        pruned_model_for_finetuning.cuda()
        print("Pruned model loaded successfully with custom logic.")

    except Exception as e:
        print(f"FATAL: Could not load pruned model even with custom logic. Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    print("Fine-tuning pruned model...")
    # *** TrainingArguments Initialization - Manual Eval/Save ***
    # Removed all evaluation, save, report, load_best parameters due to persistent TypeError
    pruned_finetuning_args = TrainingArguments(
        output_dir=f"{BASE_OUTPUT_DIR}/pruned_finetuned",
        learning_rate=1e-5, # Usually smaller LR for finetuning
        per_device_train_batch_size=DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        per_device_eval_batch_size=DEVICE_BATCH_SIZE, # Needed for manual eval dataloader
        num_train_epochs=NUM_EPOCHS_PRUNING_FINETUNE,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        report_to="wandb",
        run_name="pruned-finetune",
        optim="adamw_bnb_8bit"
        # Removed all eval/save/report parameters
    )
    print("Pruned Fine-tuning TrainingArguments initialized successfully (Manual Eval/Save).")


    pruned_trainer = Trainer(
        model=pruned_model_for_finetuning,
        args=pruned_finetuning_args,
        train_dataset=tokenized_datasets["train"],
        # eval_dataset=tokenized_datasets["validation"], # Not used by trainer
        compute_metrics=compute_metrics, # Not used by trainer
        # data_collator=DataCollatorWithPadding(tokenizer=tokenizer), # Use if needed
    )
    try:
        print("Starting pruned model fine-tuning...")
        pruned_trainer.train()
        print("Pruned model fine-tuning complete.")

        # --- Manual Evaluation and Saving ---
        print("\nPerforming manual evaluation and saving after pruned fine-tuning...")
        # Evaluate on validation set manually
        final_pruned_val_accuracy = manual_evaluate_model(
            pruned_model_for_finetuning, eval_dataloader, description="Pruned Model Final Validation"
        )

        # Save the final model after last epoch
        final_pruned_save_path = f"{BASE_OUTPUT_DIR}/fine_tuned_pruned_model_final"
        pruned_trainer.save_model(final_pruned_save_path)
        print(f"Final pruned model saved to {final_pruned_save_path}")

        # Store pruned metrics for reporting later
        pruned_metrics = {} # Initialize here or outside try/except
        pruned_metrics["accuracy"] = final_pruned_val_accuracy # Use validation accuracy

    except Exception as e:
        print(f"Error during pruned model fine-tuning or manual eval/save: {e}")
        print("Common causes: CUDA out of memory (reduce BATCH_SIZE), model/data issues.")
        sys.exit(1)


    # Measure Pruned Model energy and speed AFTER fine-tuning and saving
    print("\nMeasuring fine-tuned pruned model...")
    # Reload the *saved* final model for consistent measurement
    try:
        pruned_model_loaded_for_measurement = BertForSequenceClassification.from_pretrained(final_pruned_save_path).cuda()
        print("Fine-tuned pruned model reloaded for measurement.")
    except Exception as e:
        print(f"Error reloading fine-tuned pruned model for measurement: {e}")
        sys.exit(1)

    try:
        pruned_metrics_measurement = measure_energy_and_speed(pruned_model_loaded_for_measurement, eval_dataloader, description="Pruned Model Inference on Test Set")
        # Add measurement metrics to pruned_metrics
        pruned_metrics.update(pruned_metrics_measurement)

    except Exception as e:
        print(f"Error during pruned model energy/speed measurement: {e}")
        measurement_defaults = {
             "num_samples": len(eval_dataloader.dataset), "duration_s": 0, "avg_gpu_power_W": 0,
             "total_energy_Wh": 0, "samples_per_second": 0, "energy_per_sample_uWh": 0,
        }
        pruned_metrics.update(measurement_defaults)
        print("Pruned Measurement failed. Metrics set to default/None.")


    # Estimate FLOPs and params
    try:
        pruned_flops, pruned_params = estimate_flops(pruned_model_loaded_for_measurement, MAX_SEQ_LENGTH)
        pruned_metrics["flops"] = pruned_flops
        pruned_metrics["params"] = pruned_params
        print(f"Pruned Model Params: {pruned_metrics['params']:,}")
        print(f"Pruned Model FLOPs: {pruned_metrics['flops']/1e9:.2f} GFLOPs (Theoretical)")
    except Exception as e:
        print(f"Error estimating pruned FLOPs/Params: {e}")
        pruned_metrics["flops"] = None
        pruned_metrics["params"] = None
    
    # Load dataset again for QAT part
    dataset = load_dataset(DATASET_NAME, TASK_NAME)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    filtered_train_dataset = tokenized_datasets["train"] # Assuming no filtering for this standalone part
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=BATCH_SIZE)
    # --- End of placeholder setup ---


        # --- Stage 4: Ultimate Mixed Precision Quantization (QAT) Fix ---
    print("\n--- Stage 4: Ultimate Mixed Precision Quantization (QAT) ---")
    
    model_to_quantize = None
    try:
        print(f"Loading fine-tuned pruned model for QAT from {final_pruned_save_path}...")
        model_to_quantize = BertForSequenceClassification.from_pretrained(final_pruned_save_path)
        print("Model loaded for QAT.")
    except Exception as e:
        print(f"Error loading model for QAT: {e}")
        sys.exit(1)

    print("\nRe-preparing dataset for QAT...")
    if 'filtered_train_dataset' in locals():
         tokenized_datasets["train"] = filtered_train_dataset # Use the filtered one
    columns_to_keep = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in tokenized_datasets["train"].features:
        columns_to_keep.append("token_type_ids")
    tokenized_datasets["train"].set_format("torch", columns=columns_to_keep)
    print("Dataset re-prepared for QAT.")

    try:
        model_prepared_for_qat = prepare_ultimate_qat_model(model_to_quantize)
    except Exception as e:
        print(f"CRITICAL ERROR in ultimate QAT preparation: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\nCreating ultimate training configuration...")
    qat_grad_accum = EFFECTIVE_BATCH_SIZE // 1 
    qat_training_args = create_ultimate_training_args(BASE_OUTPUT_DIR, NUM_EPOCHS_QAT, qat_grad_accum)
    print("Ultimate training configuration created.")

    print("\n" + "="*50)
    print("STARTING QAT TRAINING WITH PURE PYTORCH LOOP")
    print("="*50)

    # 1. 从TrainingArguments获取参数
    qat_training_args = create_ultimate_training_args(BASE_OUTPUT_DIR, NUM_EPOCHS_QAT, GRADIENT_ACCUMULATION_STEPS)
    
    # 2. 准备数据加载器 (DataLoader)
    train_dataloader_qat = DataLoader(
        tokenized_datasets["train"],
        batch_size=qat_training_args.per_device_train_batch_size,
        shuffle=True
    )
    
    # 3. 准备优化器 (Optimizer) 和学习率调度器 (Scheduler)
    optimizer = torch.optim.AdamW(model_prepared_for_qat.parameters(), lr=qat_training_args.learning_rate)
    num_training_steps = len(train_dataloader_qat) * NUM_EPOCHS_QAT
    # 简单的线性学习率衰减
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)

    model_prepared_for_qat.train() # 确保模型在训练模式

    # 4. 手动训练循环
    for epoch in range(NUM_EPOCHS_QAT):
        print(f"\n--- QAT Epoch {epoch + 1}/{NUM_EPOCHS_QAT} ---")
        progress_bar = tqdm(train_dataloader_qat, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # 确保输入是 FP32 并移动到GPU
            inputs = {k: v.cuda().float() if v.dtype == torch.float16 else v.cuda() for k, v in batch.items()}
            if "labels" not in inputs and "label" in inputs:
                inputs["labels"] = inputs.pop("label")

            # 在强制FP32的上下文中进行前向传播
            # 因为我们不再使用Trainer，所以accelerate的autocast不会被触发
            outputs = model_prepared_for_qat(**inputs)
            loss = outputs.loss
            
            # 标准的反向传播
            loss.backward()
            
            # 梯度累积
            if (step + 1) % qat_training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({"loss": loss.item()})

    print("\nPure PyTorch QAT training completed successfully!")
    
    # 训练完成后的处理逻辑保持不变
    model_after_qat = model_prepared_for_qat # 现在模型就是训练好的模型
    
    try:
        print("\nPerforming final evaluation...")
        final_qat_val_accuracy = manual_evaluate_model(
            model_after_qat, eval_dataloader, description="Ultimate QAT Model Final Validation"
        )
        
        qat_prepared_state_dict_path = f"{BASE_OUTPUT_DIR}/ultimate_qat_prepared_model.pt"
        torch.save(model_after_qat.state_dict(), qat_prepared_state_dict_path)
        print(f"Ultimate QAT model saved to {qat_prepared_state_dict_path}")

        print("\nConverting to quantized format...")
        model_after_qat.eval()
        quantized_model = convert_model_to_quantized(model_after_qat.cpu())
        
        quantized_model_path = f"{BASE_OUTPUT_DIR}/ultimate_quantized_model.pt"
        torch.save(quantized_model.state_dict(), quantized_model_path)
        print(f"Ultimate quantized model saved to {quantized_model_path}")
        
        qat_metrics = {"accuracy": final_qat_val_accuracy}
        
        print("="*50)
        print("ULTIMATE QAT PROCESS COMPLETED SUCCESSFULLY!")
        if final_qat_val_accuracy is not None:
            print(f"Final QAT Accuracy: {final_qat_val_accuracy:.4f}")
        print("="*50)

    except Exception as e:
        print(f"\nCRITICAL ERROR during post-QAT processing: {e}")
        traceback.print_exc()
        sys.exit(1)


    # --- Stage 5: Measure Quantized Pruned Model ---
print("\n--- Stage 5: Measure Quantized Pruned Model ---")

# --- Custom Loading Logic for Quantized Pruned Model ---
quantized_model_loaded_for_measurement = None
quantized_model_state_dict_path = f"{BASE_OUTPUT_DIR}/quantized_pruned_model_final.pt"
# We need the path to the *pruned* model's configuration to rebuild its structure
final_pruned_save_path = f"{BASE_OUTPUT_DIR}/fine_tuned_pruned_model_final"

try:
    print(f"Loading quantized model for measurement from state_dict: {quantized_model_state_dict_path}")
    
    # Step 1: Rebuild the pruned model structure from its saved configuration.
    # The .from_pretrained() method will read the config.json from the saved directory,
    # which contains the modified number of attention heads and FFN dimensions after pruning.
    # This creates a "shell" model with the correct, smaller architecture.
    print(f"Step 1: Rebuilding pruned model structure from config at '{final_pruned_save_path}'")
    pruned_model_structure = BertForSequenceClassification.from_pretrained(final_pruned_save_path)
    pruned_model_structure.cpu() # It's safer to perform these steps on the CPU
    print("  - Pruned structure rebuilt successfully.")

    # Step 2: Apply the SAME quantization strategy to this clean, pruned structure.
    # This attaches the necessary qconfig attributes to the layers that will be quantized.
    print("Step 2: Applying the mixed-precision quantization strategy.")
    model_with_qconfig = define_mixed_precision_qconfig_strategy(pruned_model_structure)
    print("  - Qconfig applied.")

    # Step 3: Prepare and convert the model to get the final quantized architecture.
    # We are NOT training, so we use `torch.quantization.prepare` for post-training quantization setup.
    # This adds FakeQuantize modules. Then `convert` replaces them with actual quantized modules.
    # The resulting architecture will match the one whose state_dict we saved.
    print("Step 3: Preparing and converting the structure to its final quantized format.")
    model_with_qconfig.eval() # Must be in eval mode for conversion
    
    # Note: When we saved the model in Stage 4, it came from a QAT process.
    # To load it, we still need to mimic the QAT preparation and conversion steps.
    # So we use `prepare_qat` to match the architecture.
    prepared_model = torch.quantization.prepare_qat(model_with_qconfig, inplace=False)
    quantized_model_shell = convert_model_to_quantized(prepared_model)
    print("  - Structure converted to quantized format.")

    # Step 4: Load the saved quantized state_dict into our newly created quantized shell.
    # The keys in the state_dict should now perfectly match the modules in our shell.
    print(f"Step 4: Loading the saved state_dict.")
    quantized_model_shell.load_state_dict(torch.load(quantized_model_state_dict_path))
    print("  - State_dict loaded successfully.")

    # Step 5: The model is now fully restored. Move it to the GPU for measurement.
    quantized_model_loaded_for_measurement = quantized_model_shell.cuda()
    quantized_model_loaded_for_measurement.eval() # Ensure it's in eval mode for inference
    print("\nSuccessfully loaded the custom quantized and pruned model for measurement!")

except FileNotFoundError:
    print(f"Error loading quantized model: State dict or pruned model config not found.")
    print(f"  - Check for state_dict at: {quantized_model_state_dict_path}")
    print(f"  - Check for pruned model at: {final_pruned_save_path}")
    print("Ensure Stage 3 and 4 completed successfully.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during custom model loading: {e}")
    import traceback
    traceback.print_exc()
    print("This might be due to a mismatch between the saved model and the loading logic.")
    sys.exit(1)
    
    # If loading succeeded, proceed with measurement
    # This part of the code remains the same.
    # The `quantized_model_loaded_for_measurement` variable is now correctly populated.
    print("\nMeasuring quantized pruned model...")
    quantized_metrics = {} # Initialize outside try block
    try:
        quantized_metrics_measurement = measure_energy_and_speed(quantized_model_loaded_for_measurement, eval_dataloader, description="Quantized Pruned Model Inference on Test Set")
        quantized_metrics.update(quantized_metrics_measurement)
    except Exception as e:
        print(f"Error during quantized model energy/speed measurement: {e}")
        measurement_defaults = {
             "num_samples": len(eval_dataloader.dataset), "duration_s": 0, "avg_gpu_power_W": 0,
             "total_energy_Wh": 0, "samples_per_second": 0, "energy_per_sample_uWh": 0,
        }
        quantized_metrics.update(measurement_defaults)
        print("Quantized Measurement failed. Metrics set to default/None.")


    # Evaluate accuracy manually on the loaded quantized model
    try:
        print("Evaluating quantized model accuracy on test set...")
        quantized_test_accuracy = manual_evaluate_model(
            quantized_model_loaded_for_measurement, eval_dataloader, description="Quantized Model Test"
        )
        quantized_metrics["accuracy"] = quantized_test_accuracy
    except Exception as e:
        print(f"Error evaluating quantized model accuracy: {e}")
        quantized_metrics["accuracy"] = None

    # Estimate FLOPs and params (Note: FLOPs estimation for quantized might be misleading)
    try:
        # Need to estimate FLOPs/Params on the *quantized* structure if it differs
        # and if thop can handle quantized modules.
        quantized_flops, quantized_params = estimate_flops(quantized_model_loaded_for_measurement, MAX_SEQ_LENGTH)
        quantized_metrics["flops"] = quantized_flops # Theoretical FLOPs
        quantized_metrics["params"] = quantized_params # Parameter count
        print(f"Quantized Pruned Model Params: {quantized_metrics['params']:,}")
        print(f"Quantized Pruned Model FLOPs: {quantized_metrics['flops']/1e9:.2f} GFLOPs (Theoretical)")
    except Exception as e:
        print(f"Error estimating quantized FLOPs/Params: {e}")
        quantized_metrics["flops"] = None
        quantized_metrics["params"] = None


    # --- Stage 6: Results Analysis and Reporting ---
    print("\n--- Stage 6: Analysis and Reporting ---")
    print("--- Summary Metrics ---")

    # Create a dictionary to hold all metrics for easy printing
    # Check if metrics dicts were successfully created in previous stages
    all_metrics = {}
    if 'baseline_metrics' in locals() and baseline_metrics is not None:
         all_metrics["Baseline"] = baseline_metrics
    if 'pruned_metrics' in locals() and pruned_metrics is not None:
         all_metrics["Pruned Only"] = pruned_metrics
    if 'qat_metrics' in locals() and qat_metrics is not None:
         # Use the QAT metrics dict, will contain accuracy from QAT-prepared model eval
         # and measurement metrics from the converted quantized model.
         all_metrics["Pruned+Quantized"] = qat_metrics
         # Add the accuracy of the *converted* quantized model on the test set
         if 'quantized_test_accuracy' in locals():
             all_metrics["Pruned+Quantized"]["accuracy"] = quantized_test_accuracy


    if not all_metrics:
        print("No metrics were successfully calculated in previous stages to report.")
    else:
        # Print header
        header = f"{'Metric':<30}"
        for model_type in all_metrics.keys():
            header += f" | {model_type:<20}" # Increased padding for readability
        print(header)
        print("-" * (30 + len(all_metrics) * 23)) # Adjust separator length

        # Print metrics row by row
        metrics_to_print = [
            ("Accuracy", "{:.4f}"), # Use accuracy from manual eval
            ("Params (M)", "{:,.2f}"), # Use comma for thousands separator, display in Millions
            ("FLOPs (G)", "{:.2f}"),   # Display in GFLOPs
            ("FPS (Samples/s)", "{:.2f}"),
            ("Avg GPU Power (W)", "{:.2f}"),
            ("Total Energy (Wh)", "{:.6f}"), # Total energy during measurement run
            ("Energy/Sample (uWh/sample)", "{:.3f}"), # Display in microWatt-hours
        ]

        # Map display names to dictionary keys (simplified)
        metric_key_map = {
             "Accuracy": "accuracy",
             "Params (M)": "params",
             "FLOPs (G)": "flops",
             "FPS (Samples/s)": "samples_per_second",
             "Avg GPU Power (W)": "avg_gpu_power_W",
             "Total Energy (Wh)": "total_energy_Wh",
             "Energy/Sample (uWh/sample)": "energy_per_sample_uWh",
        }


        for display_name, fmt in metrics_to_print:
            row = f"{display_name:<30}"
            key = metric_key_map.get(display_name, None) # Get corresponding key

            if key:
                 for model_type, metrics in all_metrics.items():
                     value = metrics.get(key, None)

                     if value is not None:
                         # Handle unit conversion for display
                         if display_name == "Params (M)": value /= 1e6
                         if display_name == "FLOPs (G)": value /= 1e9

                         row += f" | {fmt.format(value):<20}" # Use increased padding
                     else:
                         row += f" | {'N/A':<20}" # Use increased padding
            else:
                 row += f" | {'N/A':<20} * {len(all_metrics)}" # Handle unknown metric

            print(row)

        # Calculate and print percentage improvements
        def safe_division(numerator, denominator):
            # Returns float('nan') for division by zero, which is standard for numerical results
            return numerator / denominator if denominator != 0 else float('nan')

        print("\n--- Percentage Improvements (vs Baseline) ---")
        header_pct = f"{'Metric':<30}"
        for model_type in all_metrics.keys():
            if model_type != "Baseline":
                header_pct += f" | {model_type} (%)"
        print(header_pct)
        print("-" * (30 + (len(all_metrics) - 1) * 23)) # Adjust separator length

        # Metrics where increase is good (FPS, Accuracy Gain) or reduction is good (Params, FLOPs, Energy)
        metrics_for_pct = {
            "Accuracy Change": "accuracy", # Gain
            "Params Reduction": "params", # Reduction
            "FLOPs Reduction (Theoretical)": "flops", # Reduction
            "FPS Improvement": "samples_per_second", # Gain
            "Energy Reduction (per sample)": "energy_per_sample_uWh", # Reduction
        }

        change_type_map = {
            "Accuracy Change": "gain",
            "Params Reduction": "reduction",
            "FLOPs Reduction (Theoretical)": "reduction",
            "FPS Improvement": "gain",
            "Energy Reduction (per sample)": "reduction",
        }


        for display_name, key in metrics_for_pct.items():
            row = f"{display_name:<30}"
            baseline_value = all_metrics.get("Baseline", {}).get(key, None)

            for model_type, metrics in all_metrics.items():
                if model_type != "Baseline":
                    current_value = metrics.get(key, None)
                    change_type = change_type_map[display_name] # Get change type

                    if baseline_value is not None and current_value is not None and not torch.isnan(torch.tensor(baseline_value)) and not torch.isnan(torch.tensor(current_value)):
                        if change_type == "gain":
                            pct_change = safe_division(current_value - baseline_value, baseline_value) * 100
                        elif change_type == "reduction":
                            pct_change = safe_division(baseline_value - current_value, baseline_value) * 100
                        else:
                            pct_change = float('nan') # Should not happen

                        if torch.isnan(torch.tensor(pct_change)):
                             row += f" | {'NaN%':<20}" # Handle NaN results
                        elif torch.isinf(torch.tensor(pct_change)):
                            row += f" | {'Inf%':<20}" # Handle Inf results
                        else:
                            row += f" | {pct_change:.2f}%{'':<{20 - len(f'{pct_change:.2f}%')}}" # Adjust padding
                    else:
                        row += f" | {'N/A':<20}" # Adjust padding for % column
            print(row)


    print("\n--- Research Workflow State ---")
    print(f"Results output base directory: {BASE_OUTPUT_DIR}")
    print("\nNext required steps:")
    print("1. Implement the 'TODO' sections in utils.py related to pruning:")
    print("   - calculate_energy_aware_pruning_scores")
    print("   - estimate_flops_reduction_single_unit (needs to be accurate!)")
    print("   - perform_structural_pruning (manual model modification)")
    print("2. After implementing pruning, implement the 'TODO' section in utils.py for quantization:")
    print("   - define_mixed_precision_qconfig_strategy")
    print("3. Implement the 'TODO' section in main_research_scripts.py in Stage 5:")
    print("   - Custom loading logic for the quantized pruned model.")
    print("4. Run the script again after each implementation step.")
    print("5. Analyze the printed results and saved models/info.")
    print("6. Write your report based on your method and results.")