# generate_report.py (v_final_plus - Polished for Final Submission)

import pandas as pd
import wandb
import os
import math

# --- 核心配置 ---
# 指向您主实验（能量感知方法）的输出文件夹，报告将保存在这里
BASE_OUTPUT_DIR = "./results_final_run" 
REPORT_FILENAME_TXT = "final_project_comparison_report.txt"
REPORT_FILENAME_CSV = "final_project_comparison_report.csv"
WANDB_PROJECT_NAME = "GreenAI-Optimization-Comparison"
WANDB_RUN_NAME = "Grand-Finale-Report" # 一个响亮的、全新的名字

def generate_report_text(all_metrics):
    """生成报告的纯文本字符串版本。"""
    report_lines = []

    # --- Summary Table ---
    report_lines.append("--- Final Project Report: Comprehensive Comparison ---")
    report_lines.append("======================================================")
    report_lines.append("\n--- Summary Metrics ---")
    
    header = f"{'Metric':<25}"
    model_types = list(all_metrics.keys())
    for model_type in model_types:
        header += f" | {model_type:<28}" # 增加列宽以适应更长的标题
    report_lines.append(header)
    report_lines.append("-" * (25 + len(model_types) * 31))

    metrics_to_print = [
        ("Accuracy", "accuracy", "{:.4f}", 1),
        ("Params (M)", "params", "{:,.2f}", 1e6),
        ("FLOPs (G)", "flops", "{:,.2f}", 1e9),
        ("FPS (Samples/s)", "samples_per_second", "{:.2f}", 1),
        ("Avg Power (W)", "avg_gpu_power_W", "{:.2f}", 1),
        ("Energy/Sample (uWh)", "energy_per_sample_uWh", "{:.3f}", 1),
    ]

    for display_name, key, fmt, divisor in metrics_to_print:
        row = f"{display_name:<25}"
        for model_type in model_types:
            value = all_metrics.get(model_type, {}).get(key)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                row += f" | {fmt.format(value / divisor):<28}"
            else:
                row += f" | {'N/A':<28}"
        report_lines.append(row)

    # --- Improvements Table ---
    def safe_division(numerator, denominator):
        if denominator is None or denominator == 0: return float('nan')
        return numerator / denominator

    report_lines.append("\n--- Percentage Change vs Baseline ---")
    header_pct = f"{'Metric':<25}"
    for model_type in model_types:
        if "Baseline" not in model_type:
            header_pct += f" | {model_type} (%)"
    report_lines.append(header_pct)
    report_lines.append("-" * (25 + (len(model_types) - 1) * 31))

    metrics_for_pct = {
        "Accuracy Change": ("accuracy", "gain"),
        "Params Reduction": ("params", "reduction"),
        "FLOPs Reduction": ("flops", "reduction"),
        "FPS Change/Improvement": ("samples_per_second", "gain"),
        "Energy Reduction": ("energy_per_sample_uWh", "reduction"),
    }

    baseline_metrics = next(iter(all_metrics.values()))
    for display_name, (key, change_type) in metrics_for_pct.items():
        row = f"{display_name:<25}"
        baseline_value = baseline_metrics.get(key)
        
        for model_type, metrics in all_metrics.items():
            if "Baseline" not in model_type:
                current_value = metrics.get(key)
                if baseline_value is not None and current_value is not None:
                    # 对于跨平台FPS对比，不计算百分比，只标注平台不同
                    if key == 'samples_per_second' and metrics.get('platform') != baseline_metrics.get('platform'):
                        row += f" | {'(Platform Mismatch)':<28}"
                        continue
                        
                    if change_type == "gain":
                        pct_change = safe_division(current_value - baseline_value, abs(baseline_value)) * 100
                    else: # reduction
                        pct_change = safe_division(baseline_value - current_value, abs(baseline_value)) * 100
                    
                    if not math.isnan(pct_change):
                         row += f" | {pct_change: >+8.2f}%{'':<18}"
                    else:
                         row += f" | {'N/A':<28}"
                else:
                    row += f" | {'N/A':<28}"
        report_lines.append(row)
        
    return "\n".join(report_lines)

if __name__ == "__main__":
    
    # =================================================================
    # 【最终数据区域】: 整合所有独立运行的实验结果
    # =================================================================
    
    # 1. Baseline (GPU) - from original_baseline_model.py log
    baseline_metrics = {
        "accuracy": 0.9174,
        "params": 109482242,
        "flops": 22430480384,
        "samples_per_second": 99.43,
        "energy_per_sample_uWh": 153.358,
        "avg_gpu_power_W": 54.89,
        "platform": "GPU" # 新增平台标识
    }

    # 2. Classic L1 Norm (GPU) - from traditional_L1_norm.py log
    pruning_reduction_ratio = 0.20
    l1_norm_metrics = {
        "accuracy": 0.9140,
        "params": baseline_metrics["params"] * (1 - pruning_reduction_ratio),
        "flops": baseline_metrics["flops"] * (1 - pruning_reduction_ratio),
        "samples_per_second": 189.10,
        "energy_per_sample_uWh": 50.390,
        "avg_gpu_power_W": 34.30,
        "platform": "GPU"
    }

    # 3. Your Energy-Aware Method (GPU) - from main_research_scripts.py log
    energy_aware_gpu_metrics = {
        "accuracy": 0.9255,
        "params": baseline_metrics["params"] * (1 - pruning_reduction_ratio),
        "flops": baseline_metrics["flops"] * (1 - pruning_reduction_ratio),
        "samples_per_second": 182.35,
        "energy_per_sample_uWh": 102.203,
        "avg_gpu_power_W": 67.09,
        "platform": "GPU"
    }

    # 4. Your Final Model (CPU, Dequantized) - from run_stage5_only.py log
    final_cpu_model_metrics = {
        "accuracy": 0.9220,
        "params": l1_norm_metrics.get("params"),
        "flops": l1_norm_metrics.get("flops"),
        "samples_per_second": 11.60,
        "energy_per_sample_uWh": None, 
        "avg_gpu_power_W": None,
        "platform": "CPU"
    }
    
    # 汇总所有指标，使用清晰的、适合作为列名的标签
    all_metrics_for_report = {
        "1_Baseline_GPU": baseline_metrics,
        "2_L1-Norm_Pruned_GPU": l1_norm_metrics,
        "3_Energy-Aware_Pruned_GPU": energy_aware_gpu_metrics,
        "4_Final_Model_CPU": final_cpu_model_metrics
    }
    
    # --- 1. 生成并保存 .txt 报告 ---
    report_text = generate_report_text(all_metrics_for_report)
    print(report_text)
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    txt_report_path = os.path.join(BASE_OUTPUT_DIR, REPORT_FILENAME_TXT)
    with open(txt_report_path, "w") as f:
        f.write(report_text)
    print(f"\n✅ Text report successfully saved to: {txt_report_path}")

    # --- 2. 创建 Pandas DataFrame 并保存为 .csv ---
    df = pd.DataFrame(all_metrics_for_report).drop('platform') # 在DataFrame中去掉platform行
    # 转换单位
    df.loc['params'] /= 1e6
    df.loc['flops'] /= 1e9
    df = df.rename(index={'params': 'Params (M)', 'flops': 'FLOPs (G)'})
    # 格式化浮点数显示
    df = df.round(4)
    
    csv_report_path = os.path.join(BASE_OUTPUT_DIR, REPORT_FILENAME_CSV)
    df.to_csv(csv_report_path)
    print(f"✅ CSV report successfully saved to: {csv_report_path}")

    # --- 3. 上传数据到 W&B ---
    print("\nLogging final summary to Weights & Biases...")
    try:
        run = wandb.init(project=WANDB_PROJECT_NAME, name=WANDB_RUN_NAME, job_type="final_summary")
        
        # 将格式化后的DataFrame作为W&B Table上传
        wandb_table = wandb.Table(dataframe=df.reset_index().rename(columns={'index': 'Metric'}))
        run.log({"final_comparison_table": wandb_table})
        
        # 将原始指标作为summary上传，方便过滤和分组
        summary_data = {}
        for model_name, metrics in all_metrics_for_report.items():
            for metric_name, value in metrics.items():
                if value is not None:
                    summary_key = f"{model_name}/{metric_name}"
                    summary_data[summary_key] = value
        run.summary.update(summary_data)

        # 上传报告文件
        artifact = wandb.Artifact('final_reports', type='report')
        artifact.add_file(txt_report_path)
        artifact.add_file(csv_report_path)
        run.log_artifact(artifact)
        
        run.finish()
        print("✅ Successfully logged summary table and artifacts to W&B.")
        print(f"  - Find your run at: {run.url}")
        
    except Exception as e:
        print(f"\n❌ Could not log to W&B. Error: {e}")
        print("  - Make sure you are logged in by running 'wandb login'.")