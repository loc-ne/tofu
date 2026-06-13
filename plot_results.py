import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_aggregated_json(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    with open(file_path, "r") as f:
        return json.load(f)

def get_metrics_summary(eval_result_dict):
    eval_task_dict = {
        'eval_log.json': 'Retain',
        'eval_log_forget.json': 'Forget',
        'eval_real_author_wo_options.json': 'Real Authors',
        'eval_real_world_wo_options.json': 'World Facts'
    }
    
    summary = {}
    for task_file, task_name in eval_task_dict.items():
        if task_file not in eval_result_dict:
            continue
        task_data = eval_result_dict[task_file]
        
        # ROUGE
        rouge = np.array(list(task_data.get('rougeL_recall', {}).values())).mean() if 'rougeL_recall' in task_data else 0.0
        
        # Probability
        if 'eval_log' in task_file:
            gt_probs = np.exp(-1 * np.array(list(task_data.get('avg_gt_loss', {}).values())))
            prob = np.mean(gt_probs) if len(gt_probs) > 0 else 0.0
        else:
            avg_true_prob = np.exp(-1 * np.array(list(task_data.get('avg_gt_loss', {}).values())))
            avg_false_prob = np.exp(-1 * np.array(list(task_data.get('average_perturb_loss', {}).values())))
            avg_all_prob = np.concatenate([np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1).sum(-1)
            prob = np.mean(avg_true_prob / avg_all_prob) if len(avg_all_prob) > 0 else 0.0
            
        # Truth Ratio
        avg_paraphrase = np.array(list(task_data.get('avg_paraphrased_loss', {}).values()))
        avg_perturbed = np.array(list(task_data.get('average_perturb_loss', {}).values())).mean(axis=-1)
        if len(avg_paraphrase) > 0 and len(avg_perturbed) > 0:
            curr_stat = np.exp(avg_perturbed - avg_paraphrase)
            if 'forget' in task_file:
                truth_ratio = np.mean(np.minimum(curr_stat, 1.0 / curr_stat))
            else:
                truth_ratio = np.mean(np.maximum(0.0, 1.0 - 1.0 / curr_stat))
        else:
            truth_ratio = 0.0
            
        summary[task_name] = {
            'ROUGE': rouge,
            'Probability': prob,
            'Truth Ratio': truth_ratio
        }
    return summary

def plot_unlearn_curves(method_summaries, save_path="unlearn_curves.png"):
    """
    Plots the performance metrics (ROUGE, Probability, Truth Ratio) 
    across 4 evaluation sets for different unlearning methods.
    method_summaries is a dict: {MethodName: summary_dict}
    """
    # 1. Vẽ tệp tổng hợp dạng xếp dọc 3x1 (phù hợp trang A4 dọc)
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    metrics = ['ROUGE', 'Probability', 'Truth Ratio']
    eval_sets = ['Forget', 'Retain', 'Real Authors', 'World Facts']
    
    x = np.arange(len(eval_sets))
    width = 0.18
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for offset_idx, (method_name, summary) in enumerate(method_summaries.items()):
            y_vals = []
            for s in eval_sets:
                y_vals.append(summary.get(s, {}).get(metric, 0.0))
            
            ax.bar(x + (offset_idx - len(method_summaries)/2.0 + 0.5) * width, y_vals, width, label=method_name)
        
        ax.set_title(f'{metric} Comparison', fontsize=16, weight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(eval_sets, fontsize=12)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(fontsize=11, loc='upper right')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Comparison curves (3x1 layout) saved to: {save_path}")

    # 2. Vẽ và lưu thêm 3 tệp riêng lẻ để chèn LaTeX phóng to tối đa
    dir_name = os.path.dirname(save_path)
    base_name = os.path.basename(save_path).replace(".png", "")
    for metric in metrics:
        fig_single, ax = plt.subplots(figsize=(9, 6))
        for offset_idx, (method_name, summary) in enumerate(method_summaries.items()):
            y_vals = []
            for s in eval_sets:
                y_vals.append(summary.get(s, {}).get(metric, 0.0))
            ax.bar(x + (offset_idx - len(method_summaries)/2.0 + 0.5) * width, y_vals, width, label=method_name)
            
        ax.set_title(f'{metric} Comparison', fontsize=15, weight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(eval_sets, fontsize=12)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(fontsize=11, loc='upper right')
        
        plt.tight_layout()
        metric_safe = metric.replace(" ", "_")
        single_save_path = os.path.join(dir_name, f"{base_name}_{metric_safe}.png")
        plt.savefig(single_save_path, dpi=300)
        plt.close()
        print(f"Single metric curve saved to: {single_save_path}")

def plot_tradeoff_scatter(methods_data, save_path="tradeoff_scatter.png"):
    """
    Plots the trade-off between Forget Quality (Y-axis in log10 p-value) and Model Utility (X-axis).
    methods_data is a list of dicts containing: 'name', 'model_utility', 'forget_quality'
    """
    plt.figure(figsize=(10, 8))
    
    # Sig threshold at p = 0.05 -> log10(p) = -1.301
    sig_threshold = np.log10(0.05)
    
    # We will use log10 of p-value for Y axis to prevent all points clustering at Y=0.
    # Clamp p-value at 1e-20 to prevent -inf.
    for item in methods_data:
        p_val = max(item['forget_quality'], 1e-20)
        log_p_val = np.log10(p_val)
        
        plt.scatter(
            item['model_utility'], 
            log_p_val, 
            s=220, 
            label=item['name'], 
            alpha=0.9, 
            edgecolors='black',
            linewidths=1.2
        )
        # Offset the label slightly
        plt.text(
            item['model_utility'] + 0.01, 
            log_p_val - 0.3, 
            item['name'], 
            fontsize=11, 
            weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none")
        )
        
    plt.xlabel('Model Utility (Higher is Better)', fontsize=14, labelpad=10)
    plt.ylabel(r'Forget Quality ($\log_{10}$ KS Test P-value)', fontsize=14, labelpad=10)
    plt.title('Trade-off between Model Utility and Forget Quality', fontsize=16, pad=20, weight='bold')
    
    # Dynamically scale X-axis minimum based on data to prevent points from getting cut off on the left
    min_utility = min([item['model_utility'] for item in methods_data] + [0.4])
    x_min = max(-0.05, min_utility - 0.05)
    plt.xlim(x_min, 1.05)
    
    # Since p-value is clamped at 1e-20, log10 is -20. Y-axis range: -22 to 0.5
    plt.ylim(-22, 0.5)
    
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(linestyle=':', alpha=0.6)
    
    # Draw target forget zone (p-value >= 0.05 -> log10(p-value) >= -1.301)
    plt.axhspan(sig_threshold, 0.5, color='#E8F5E9', alpha=0.6, label='Target Forget Zone (p-value >= 0.05)')
    # Red dashed line for the threshold
    plt.axhline(y=sig_threshold, color="#C62828", linestyle="--", alpha=0.7, linewidth=1.5)
    plt.text(0.41, sig_threshold - 0.8, "p = 0.05 threshold", color="#C62828", fontsize=10, weight="bold")
    
    plt.legend(loc='lower left', fontsize=12, frameon=True, facecolor='white', edgecolor='lightgray')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Trade-off scatter plot saved to: {save_path}")

if __name__ == "__main__":
    import sys
    import csv
    import collections
    
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <path_to_csv1> [path_to_csv2] ...")
        sys.exit(1)
        
    import glob
    csv_inputs = sys.argv[1:]
    csv_paths = []
    for path in csv_inputs:
        # Resolve wildcard patterns (e.g. forget1/*.csv)
        # Use recursive=True or normal expansion, and normalize path separators
        normalized_path = path.replace("/", os.sep).replace("\\", os.sep)
        matched = glob.glob(normalized_path)
        if matched:
            csv_paths.extend(matched)
        else:
            if os.path.exists(normalized_path):
                csv_paths.append(normalized_path)
            else:
                print(f"File not found: {path}")
                
    # Group CSV paths by their parent directory (forget1, forget5, forget10, etc.)
    grouped_paths = collections.defaultdict(list)
    for path in csv_paths:
        parent_dir = os.path.dirname(path)
        if not parent_dir:
            parent_dir = "eval_results"
        grouped_paths[parent_dir].append(path)
        
    # Process and plot each group separately
    for output_dir, paths in grouped_paths.items():
        print(f"\nProcessing directory: {output_dir} ({len(paths)} files)")
        method_summaries = {}
        methods_data = []
        
        for path in paths:
            with open(path, mode='r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    method_name = row.get('Method', 'unknown')
                    
                    # Extract metrics for bar curves
                    method_summaries[method_name] = {
                        'Retain': {
                            'ROUGE': float(row.get('ROUGE Retain', 0.0)),
                            'Probability': float(row.get('Prob. Retain', 0.0)),
                            'Truth Ratio': float(row.get('Truth Ratio Retain', 0.0))
                        },
                        'Forget': {
                            'ROUGE': float(row.get('ROUGE Forget', 0.0)),
                            'Probability': float(row.get('Prob. Forget', 0.0)),
                            'Truth Ratio': float(row.get('Truth Ratio Forget', 0.0))
                        },
                        'Real Authors': {
                            'ROUGE': float(row.get('ROUGE Real Authors', 0.0)),
                            'Probability': float(row.get('Prob. Real Authors', 0.0)),
                            'Truth Ratio': float(row.get('Truth Ratio Real Authors', 0.0))
                        },
                        'World Facts': {
                            'ROUGE': float(row.get('ROUGE Real World', row.get('ROUGE World Facts', 0.0))),
                            'Probability': float(row.get('Prob. Real World', row.get('Prob. World Facts', 0.0))),
                            'Truth Ratio': float(row.get('Truth Ratio Real World', row.get('Truth Ratio World Facts', 0.0)))
                        }
                    }
                    
                    # Extract metrics for trade-off scatter
                    model_utility = float(row.get('Model Utility', 0.0))
                    forget_quality = float(row.get('Forget Quality', 0.0))
                    methods_data.append({
                        'name': method_name,
                        'model_utility': model_utility,
                        'forget_quality': forget_quality
                    })
                    
        if method_summaries:
            os.makedirs(output_dir, exist_ok=True)
            plot_unlearn_curves(method_summaries, save_path=os.path.join(output_dir, "unlearn_curves.png"))
        if methods_data:
            os.makedirs(output_dir, exist_ok=True)
            plot_tradeoff_scatter(methods_data, save_path=os.path.join(output_dir, "tradeoff_scatter.png"))

