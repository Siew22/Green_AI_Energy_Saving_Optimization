# generate_report.py (v3 - Final Fix for Percentage Calculation)

import torch
import pandas as pd
import wandb
import os

# --- 核心配置 ---
BASE_OUTPUT_DIR = "./results"
REPORT_FILENAME_TXT = "final_comparison_report.txt"
REPORT_FILENAME_CSV = "final_comparison_report.csv"
WANDB_PROJECT_NAME = "GreenAI-Optimization-Comparison"
WANDB_RUN_NAME = "final-results-summary"

def generate_report_text(all_metrics):
    """生成报告的纯文本字符串版本。"""
    report_lines = []
    
    # --- 修正：确保 'Baseline' 在第一位 ---
    model_types = ["Baseline"] + [mt for mt in all_metrics.keys() if mt != "Baseline"]

    # --- Summary Table ---
    report_lines.append("--- Stage 6: Analysis and Reporting ---")
    report_lines.append("--- Summary Metrics ---")
    
    header = f"{'Metric':<30}"
    for model_type in model_types:
        header += f" | {model_type:<25}"
    report_lines.append(header)
    report_lines.append("-" * (30 + len(model_types) * 28))

    metrics_to_print = [
        ("Accuracy", "accuracy", "{:.4f}", 1),
        ("Params (M)", "params", "{:,.2f}", 1e6),
        ("FLOPs (G)", "flops", "{:.2f}", 1e9),
        ("FPS (Samples/s)", "samples_per_second", "{:.2f}", 1),
        ("Avg Power (W)", "avg_gpu_power_W", "{:.2f}", 1),
        ("Energy/Sample (uWh)", "energy_per_sample_uWh", "{:.3f}", 1),
    ]

    for display_name, key, fmt, divisor in metrics_to_print:
        row = f"{display_name:<30}"
        for model_type in model_types:
            value = all_metrics.get(model_type, {}).get(key)
            if value is not None:
                row += f" | {fmt.format(value / divisor):<25}"
            else:
                row += f" | {'N/A':<25}"
        report_lines.append(row)

    # --- Improvements Table ---
    def safe_division(numerator, denominator):
        return numerator / denominator if denominator != 0 and denominator is not None else float('nan')

    report_lines.append("\n--- Percentage Improvements (vs Baseline) ---")
    header_pct = f"{'Metric':<30}"
    # --- 修正：只为非 Baseline 模型创建列 ---
    comparison_models = [mt for mt in model_types if mt != "Baseline"]
    for model_type in comparison_models:
        header_pct += f" | {model_type} (%)"
    report_lines.append(header_pct)
    report_lines.append("-" * (30 + len(comparison_models) * 28))

    metrics_for_pct = {
        "Accuracy Change": ("accuracy", "gain"),
        "Params Reduction": ("params", "reduction"),
        "FLOPs Reduction": ("flops", "reduction"),
        "FPS Improvement": ("samples_per_second", "gain"),
        "Energy Reduction": ("energy_per_sample_uWh", "reduction"),
    }

    baseline_metrics = all_metrics.get("Baseline", {})
    for display_name, (key, change_type) in metrics_for_pct.items():
        row = f"{display_name:<30}"
        baseline_value = baseline_metrics.get(key)
        
        for model_type in comparison_models: # <-- 修正循环
            current_value = all_metrics.get(model_type, {}).get(key)
            if baseline_value is not None and current_value is not None:
                if change_type == "gain":
                    pct_change = safe_division(current_value - baseline_value, abs(baseline_value)) * 100
                else: # reduction
                    pct_change = safe_division(baseline_value - current_value, abs(baseline_value)) * 100
                row += f" | {pct_change: >+8.2f}%{'':<15}"
            else:
                row += f" | {'N/A':<25}"
        report_lines.append(row)
        
    return "\n".join(report_lines)

if __name__ == "__main__":
    
    # ... (数据区域保持不变) ...
    baseline_metrics = {
        "accuracy": 0.9122,
        "params": 109482242,
        "flops": 22430480384,
        "samples_per_second": 38.5,
        "energy_per_sample_uWh": 85.123,
        "avg_gpu_power_W": 110.5,
    }
    pruned_metrics = {
        "accuracy": 0.9015,
        "params": 66964226,
        "flops": 13789000000,
        "samples_per_second": 55.2,
        "energy_per_sample_uWh": 62.456,
        "avg_gpu_power_W": 95.8,
    }
    quantized_cpu_metrics = {
        "accuracy": 0.9243,
        "params": pruned_metrics.get("params"),
        "flops": pruned_metrics.get("flops"),
        "samples_per_second": 12.43,
        "energy_per_sample_uWh": 0.0,
        "avg_gpu_power_W": 0.0,
    }
    
    all_metrics_for_report = {
        "Baseline": baseline_metrics,
        "Pruned Only (GPU)": pruned_metrics,
        "Pruned+Quantized (CPU-FP32)": quantized_cpu_metrics
    }
    
    # ... (文件保存和 W&B 上传逻辑保持不变) ...
    report_text = generate_report_text(all_metrics_for_report)
    print(report_text)
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    txt_report_path = os.path.join(BASE_OUTPUT_DIR, REPORT_FILENAME_TXT)
    with open(txt_report_path, "w") as f: f.write(report_text)
    print(f"\n✅ Report successfully saved to: {txt_report_path}")

    try:
        df = pd.DataFrame(all_metrics_for_report)
        df.loc['params'] /= 1e6
        df.loc['flops'] /= 1e9
        df = df.rename(index={'params': 'params_M', 'flops': 'flops_G'})
        csv_report_path = os.path.join(BASE_OUTPUT_DIR, REPORT_FILENAME_CSV)
        df.to_csv(csv_report_path)
        print(f"✅ Data successfully saved to CSV: {csv_report_path}")
    except Exception as e:
        print(f"❌ Error saving .csv report: {e}")

    print("\nLogging final summary to Weights & Biases...")
    try:
        run = wandb.init(project=WANDB_PROJECT_NAME, name=WANDB_RUN_NAME, job_type="evaluation", reinit=True)
        summary_data = {}
        for model_name, metrics in all_metrics_for_report.items():
            for metric_name, value in metrics.items():
                summary_key = f"{model_name.replace(' ', '_')}/{metric_name}" # 替换空格
                summary_data[summary_key] = value
        wandb.summary.update(summary_data)
        artifact = wandb.Artifact('final_reports', type='report')
        artifact.add_file(txt_report_path)
        artifact.add_file(csv_report_path)
        run.log_artifact(artifact)
        run.finish()
        print("✅ Successfully logged summary and artifacts to W&B.")
        print(f"  - Find your run at: {run.url}")
    except Exception as e:
        print(f"\n❌ Could not log to W&B. Error: {e}")