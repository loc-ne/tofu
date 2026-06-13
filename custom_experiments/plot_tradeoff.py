import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình phong cách vẽ biểu đồ chuẩn học thuật (Academic/Scientific Style)
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 17,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def load_data():
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    # Đường dẫn tới các thư mục kết quả của tập quên 1%, 5% và 10%
    folders = {
        "1% Forget Set": os.path.join(BASE_DIR, "..", "forget1"),
        "5% Forget Set": os.path.join(BASE_DIR, "..", "forget5"),
        "10% Forget Set": os.path.join(BASE_DIR, "..", "forget10")
    }
    
    # Ánh xạ mã thuật toán thô sang nhãn học thuật chuẩn
    method_mapping = {
        "dpo": "DPO",
        "grad_ascent": "Gradient Ascent (GA)",
        "grad_diff": "Gradient Difference (GD)",
        "KL": "KL Minimization"
    }
    
    data = {}
    for label, folder in folders.items():
        data[label] = []
        csv_pattern = os.path.join(folder, "stat_*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print(f"Warning: No CSV files found in directory '{folder}'")
            continue
            
        # Đọc dữ liệu từ tất cả các file CSV tìm thấy trong thư mục
        for filepath in csv_files:
            with open(filepath, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_method = row.get("Method", "unknown")
                    method_name = method_mapping.get(raw_method, raw_method)
                    
                    try:
                        utility = float(row.get("Model Utility", 0.0))
                        p_val = float(row.get("Forget Quality", 0.0))
                        
                        # Tính toán log10(p-value) cho trục Y (giới hạn min 1e-20 để tránh lỗi log của 0)
                        clamped_p_val = max(p_val, 1e-20)
                        log_p_val = np.log10(clamped_p_val)
                        
                        data[label].append({
                             "method": method_name,
                             "utility": utility,
                             "p_value": p_val,
                             "log_p_value": log_p_val
                        })
                    except ValueError as e:
                        print(f"Skipping row in {filepath} due to parsing error: {e}")
                        
    return data

def plot_tradeoff(data, save_path=None):
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    if save_path is None:
        save_path = os.path.join(BASE_DIR, "results", "tradeoff_tofu_academic.png")
        
    # Tạo thư mục lưu kết quả nếu chưa tồn tại
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Khởi tạo khung hình chứa 3 biểu đồ phụ nằm ngang
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharey=True)
    
    # Định nghĩa màu sắc, hình dạng điểm vẽ và chữ viết tắt cho từng phương pháp
    method_styles = {
        "DPO": {
            "color": "#D32F2F", 
            "marker": "o", 
            "abbrev": "DPO"
        },
        "Gradient Ascent (GA)": {
            "color": "#1976D2", 
            "marker": "s", 
            "abbrev": "GA"
        },
        "Gradient Difference (GD)": {
            "color": "#388E3C", 
            "marker": "^", 
            "abbrev": "GD"
        },
        "KL Minimization": {
            "color": "#F57C00", 
            "marker": "D", 
            "abbrev": "KL"
        }
    }
    
    # Ngưỡng ý nghĩa thống kê (p = 0.05 -> log10(p) = -1.3)
    sig_threshold = np.log10(0.05)
    
    forget_labels = ["1% Forget Set", "5% Forget Set", "10% Forget Set"]
    
    for idx, label in enumerate(forget_labels):
        ax = axes[idx]
        points = data.get(label, [])
        
        # Vẽ vùng màu nền biểu thị chất lượng unlearn tốt (p >= 0.05)
        ax.axhspan(sig_threshold, 1.0, color="#E8F5E9", alpha=0.6, label="Retain-like (p >= 0.05)")
        # Vẽ đường ranh giới màu đỏ dạng nét đứt
        ax.axhline(y=sig_threshold, color="#C62828", linestyle="--", alpha=0.7, linewidth=1.5)
        
        # Chú thích đường ngưỡng trong biểu đồ phụ đầu tiên
        if idx == 0:
            ax.text(0.05, sig_threshold - 0.7, "p = 0.05 threshold", color="#C62828", fontsize=10, weight="bold")
            
        # Vẽ các điểm biểu diễn kết quả của từng thuật toán
        for pt in points:
            m_name = pt["method"]
            style = method_styles.get(m_name, {"color": "gray", "marker": "x", "abbrev": "UNK"})
            
            ax.scatter(
                pt["utility"], 
                pt["log_p_value"], 
                color=style["color"],
                marker=style["marker"],
                s=200, 
                edgecolors="black", 
                linewidths=1.2, 
                alpha=0.95,
                label=m_name
            )
            
            # Ghi nhãn chữ viết tắt cạnh mỗi điểm dữ liệu
            ax.annotate(
                style["abbrev"],
                (pt["utility"], pt["log_p_value"]),
                textcoords="offset points",
                xytext=(8, -4),
                ha="left",
                fontsize=9.5,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none")
            )
            
        # Cấu hình trục và tiêu đề biểu đồ phụ
        ax.set_title(label, pad=15, weight="bold")
        ax.set_xlabel("Model Utility (higher is better)", labelpad=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        
        # Chỉ hiển thị nhãn trục Y ở biểu đồ ngoài cùng bên trái (do sharey=True)
        if idx == 0:
            ax.set_ylabel(r"Forget Quality ($\log_{10}$ p-value)", labelpad=10)
            ax.set_ylim(-18, 0.5)
            
    # Xử lý gộp chú thích đồ thị (Legend) chung ở phía trên để tránh chật không gian
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    fig.legend(
        by_label.values(), 
        by_label.keys(), 
        loc="upper center", 
        bbox_to_anchor=(0.5, 0.90),
        ncol=5, 
        frameon=True,
        facecolor="white",
        edgecolor="lightgray"
    )
    
    # Căn chỉnh bố cục để chừa khoảng trống cho tiêu đề chính và legend
    plt.subplots_adjust(top=0.78, bottom=0.15, left=0.08, right=0.96, wspace=0.15)
    
    # Tiêu đề chính toàn khung hình
    plt.suptitle("Trade-off between Model Utility and Forget Quality on TOFU Benchmark", y=0.97, weight="bold")
    
    # Lưu và xuất ảnh chất lượng cao (300 DPI)
    plt.savefig(save_path, dpi=300)
    print(f"Trade-off scatter plot saved successfully to {save_path}")

if __name__ == "__main__":
    data = load_data()
    plot_tradeoff(data)

