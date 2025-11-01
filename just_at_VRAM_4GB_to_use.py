# just_at_VRAM_4GB_to_use.py
import sys # Import sys at the very top
import os
os.environ["WANDB_PROJECT"] = "GreenAI-Optimization-FYP" # 在脚本顶部设置项目名
import json # Needed for saving/loading pruning info
import copy # Needed for deepcopy
import time # Needed for time measurement helpers
import argparse
import gc 

import torch
import torch.nn as nn
# These imports are where the TypeError happened. We will check their version/path just after import.
from transformers import BertForSequenceClassification, BertTokenizerFast, TrainingArguments, Trainer
from datasets import load_dataset
from tqdm import tqdm

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


# --- Smart Configuration based on Command-Line Arguments ---
parser = argparse.ArgumentParser(description="Run Green AI Pipeline for a specific BERT model.")
parser.add_argument(
    '--model_size', 
    type=str, 
    required=True, 
    choices=['tiny', 'small', 'base'],
    help="The size of the BERT model to run."
)
parser.add_argument('--alpha', type=float, default=0.5)
args = parser.parse_args()

# 根据选择的模型大小，智能设定100%能成功的配置
if args.model_size == 'tiny':
    MODEL_NAME = "prajjwal1/bert-tiny"
    BATCH_SIZE = 16 # Tiny模型可以用大批量
    GRAD_ACCUM = 2
elif args.model_size == 'small':
    MODEL_NAME = "prajjwal1/bert-small"
    BATCH_SIZE = 8
    GRAD_ACCUM = 4
elif args.model_size == 'base':
    MODEL_NAME = "bert-base-uncased"
    BATCH_SIZE = 1 # Base模型必须用极小的、100%能成功的批量
    GRAD_ACCUM = 32

# 统一设置其他参数
DATASET_NAME = "glue"
TASK_NAME = "sst2"
BASE_OUTPUT_DIR = f"./results_{MODEL_NAME.split('/')[-1]}_energy_aware_alpha{args.alpha}"
NUM_EPOCHS_BASELINE = 3
NUM_EPOCHS_PRUNING_FINETUNE = 5
NUM_EPOCHS_QAT = 3
PRUNING_FLOPs_REDUCTION_TARGET = 0.4
ENERGY_AWARE_ALPHA = args.alpha

# QAT专用配置
QAT_BATCH_SIZE = 1
QAT_GRAD_ACCUM = GRAD_ACCUM * (BATCH_SIZE // QAT_BATCH_SIZE)

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


    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=BATCH_SIZE)
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
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
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
                del model_to_prune, pruned_model, pruning_scores
                gc.collect()
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
        
        # --- Critical fix: Use custom loading logic ---
        # 1. First, load the new configuration file saved by the pruned model。
        # This configuration file records the correct structure of the model after pruning (such as fewer attention heads)。
        from transformers import AutoConfig
        pruned_config = AutoConfig.from_pretrained(pruned_model_save_path)
        
        # 2. Then, we tell Hugging Face to start from the original pre-trained model，
        # But when creating the model, force the use of the new pruned configuration we just loaded。
        print("Initializing model with the new pruned configuration to avoid size mismatch...")
        pruned_model_for_finetuning = BertForSequenceClassification.from_pretrained(
            MODEL_NAME,       # Still loading weights from the original model name
            config=pruned_config # <--- But it is forced to use the new configuration after pruning to build the model structure
        ).cuda()
        
        print("Pruned model loaded successfully with custom logic.")

    except Exception as e:
        # If even this fails, there is a deeper problem.
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
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        per_device_eval_batch_size=BATCH_SIZE, # Needed for manual eval dataloader
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
        del pruned_trainer, pruned_model_for_finetuning
        gc.collect()
        torch.cuda.empty_cache()
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


    # --- Stage 4: Mixed Precision Quantization (QAT) - MANUAL EVAL/SAVE ---
    # Load the fine-tuned pruned model for QAT
    model_to_quantize = None # Initialize outside try block
    try:
        print(f"Loading fine-tuned pruned model for QAT from {final_pruned_save_path}...")
        model_to_quantize = BertForSequenceClassification.from_pretrained(final_pruned_save_path).cuda()
        print("Model loaded for QAT.")
    except Exception as e:
        print(f"Error loading model for QAT: {e}")
        print("Ensure the finetuned pruned model was saved/loaded correctly.")
        sys.exit(1)

    # --- 修复点 1 (推荐): 在QAT前重置并准备数据集 ---
    print("\nRe-preparing dataset for QAT to ensure FP32 compatibility...")
    # 你的 filtered_train_dataset 和 tokenized_datasets 变量应该仍然可用。
    # 我们只需重置并重新设置格式，以清除任何可能由上一个FP16 Trainer留下的状态。
    if 'filtered_train_dataset' in locals():
        filtered_train_dataset.reset_format() 
        tokenized_datasets["train"] = filtered_train_dataset
    
    columns_to_keep = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in tokenized_datasets["train"].features:
        columns_to_keep.append("token_type_ids")
    tokenized_datasets["train"].set_format("torch", columns=columns_to_keep)
    print("Dataset re-prepared for QAT.")
    
    
    # 定义和准备QAT模型
    print("\nDefining mixed precision QAT strategy...")
    model_prepared_for_qat = None
    try:
        model_prepared_for_qat = define_mixed_precision_qconfig_strategy(model_to_quantize)
        print("Mixed precision QAT strategy applied to model.")
    except Exception as e:
        print(f"Error defining mixed precision QAT strategy: {e}")
        sys.exit(1)

    try:
        print("Preparing model for QAT...")
        model_prepared_for_qat.train()
        model_prepared_for_qat = torch.quantization.prepare_qat(model_prepared_for_qat, inplace=False)
        model_prepared_for_qat.cuda()
        print("Model prepared for QAT and moved to GPU.")
    except Exception as e:
         print(f"Error preparing model for QAT: {e}")
         sys.exit(1)


    # Perform QAT training
    print("\nStarting QAT training...")
    qat_training_args = TrainingArguments(
        output_dir=f"{BASE_OUTPUT_DIR}/qat_pruned",
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4, # 保持有效批量为4
        per_device_eval_batch_size=1,
        num_train_epochs=NUM_EPOCHS_QAT,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=1,
        
        # --- 终极修复：彻底关闭所有混合精度相关的选项 ---
        fp16=False,          # 明确关闭FP16
        bf16=False,          # 明确关闭BF16
        tf32=False,          # 明确关闭TF32
        # ---------------------------------------------------
        
        report_to="wandb",
        run_name="qat-finetune",
        # 为了100%的稳定性，暂时移除8-bit优化器
        # 如果你发现OOM，再把它加回来
        # optim="adamw_bnb_8bit" 
    )
    print("QAT TrainingArguments initialized successfully.")

    # --- 修复点 2 (必须): 在将模型送入Trainer前，强制转换为FP32 ---
    print("Converting QAT-prepared model to FP32 before training to ensure compatibility...")
    model_prepared_for_qat_fp32 = model_prepared_for_qat.to(torch.float32)
    
    qat_trainer = Trainer(
        model=model_prepared_for_qat_fp32,
        args=qat_training_args,
        train_dataset=tokenized_datasets["train"],
        compute_metrics=compute_metrics,
    )
    
    try:
        print("Starting QAT training...")
        qat_trainer.train()
        print("QAT training finished.")
        
        # --- 保存和转换 ---
        # 训练完成后，模型在qat_trainer.model中
        model_after_qat = qat_trainer.model

        print("\nPerforming manual evaluation and saving after QAT training...")
        final_qat_val_accuracy = manual_evaluate_model(
            model_after_qat, eval_dataloader, description="QAT Model Final Validation"
        )
        
        # 保存QAT-prepared模型的状态字典
        qat_prepared_state_dict_path = f"{BASE_OUTPUT_DIR}/qat_prepared_pruned_model_final.pt"
        torch.save(model_after_qat.state_dict(), qat_prepared_state_dict_path)
        print(f"QAT prepared model state dict saved to {qat_prepared_state_dict_path}")

        # 将模型转换为最终的量化格式
        print("\nConverting model to quantized format after QAT...")
        quantized_model = convert_model_to_quantized(model_after_qat)
        print("Model converted to quantized format.")

        quantized_model_state_dict_path = f"{BASE_OUTPUT_DIR}/quantized_pruned_model_final.pt"
        torch.save(quantized_model.state_dict(), quantized_model_state_dict_path)
        print(f"Quantized pruned model state_dict saved to {quantized_model_state_dict_path}")
        
        qat_metrics = {"accuracy": final_qat_val_accuracy}

    except Exception as e:
        print(f"Error during QAT training or subsequent steps: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

        # Store QAT/Quantized metrics for reporting later
        # We'll evaluate the accuracy of the *converted* quantized model later in Stage 5 measurement
        # The accuracy here is from the QAT-prepared model, which is a good estimate.
        qat_metrics = {} # Initialize here or outside try/except
        qat_metrics["accuracy"] = final_qat_val_accuracy # Use validation accuracy from QAT-prepared model

    except Exception as e:
        print(f"Error during QAT training or manual eval/save/convert: {e}")
        print("Common causes: CUDA out of memory (reduce BATCH_SIZE), QAT instability, errors in FakeQuantize implementations.")
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