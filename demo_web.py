import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# --- 1. CẤU HÌNH GIAO DIỆN CHUẨN GEMINI (CLEAN & MINIMALIST) ---
st.set_page_config(
    page_title="Unlearning Model Demo",
    page_icon="🧠",
    layout="wide"
)

# Nhúng CSS tùy biến để ép Streamlit đổi sang giao diện tối giản, bo góc chuẩn Google
st.markdown("""
    <style>
        /* Giấu menu mặc định và footer của Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Cấu hình font chữ và nền tổng thể sạch sẽ */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        /* Style tiêu đề lớn */
        .main-title {
            font-size: 2.2rem;
            font-weight: 600;
            background: linear-gradient(45deg, #1a73e8, #a142f4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        
        .sub-title {
            color: #5f6368;
            font-size: 1rem;
            margin-bottom: 2rem;
        }

        /* Làm mượt ô nhập liệu text input giống ô prompt Gemini */
        div[data-baseweb="input"] {
            border-radius: 24px !important;
            border: 1px solid #dadce0 !important;
            padding: 4px 12px;
            box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15) !important;
            background-color: #ffffff !important;
        }
        div[data-baseweb="input"]:focus-within {
            border-color: #1a73e8 !important;
        }
        
        /* Bo tròn các hộp lựa chọn Selectbox ở Sidebar */
        div[data-baseweb="select"] {
            border-radius: 12px !important;
        }

        /* Tùy biến nút bấm Generate chính thành hình bầu dục phẳng */
        div.stButton > button:first-child {
            border-radius: 20px !important;
            padding: 0.5rem 2rem !important;
            font-weight: 500 !important;
            background-color: #1a73e8 !important;
            border: none !important;
            box-shadow: 0 1px 3px 0 rgba(60,64,67,0.3) !important;
            transition: all 0.2s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #1557b0 !important;
            box-shadow: 0 4px 8px 0 rgba(60,64,67,0.3) !important;
        }

        /* Định dạng khung hiển thị kết quả dạng Chat Bubble */
        .model-card {
            padding: 1.5rem;
            border-radius: 16px;
            margin-top: 1rem;
            font-size: 1.05rem;
            line-height: 1.6;
            box-shadow: 0 1px 3px 0 rgba(60,64,67,0.1), 0 4px 8px 3px rgba(60,64,67,0.05);
        }
        .ft-card {
            background-color: #e8f0fe;
            border-left: 5px solid #1a73e8;
            color: #1967d2;
        }
        .unlearn-card {
            background-color: #fef7e0;
            border-left: 5px solid #f9ab00;
            color: #b06000;
        }
        .card-header {
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. HÀM LOAD MÔ HÌNH (CÓ CACHE VÀ ÉP LÊN GPU) ---
@st.cache_resource
def load_model(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        config = AutoConfig.from_pretrained(model_path)
        config.pad_token_id = tokenizer.pad_token_id
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            dtype=torch.float16,
            trust_remote_code=True
        ).to("cuda")
        
        model.generation_config.do_sample = True
        model.config.use_cache = True
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"[ERROR]: {str(e)}")
        raise e

# --- 3. QUÉT THƯ MỤC TÌM MÔ HÌNH ---
models_dir = "/kaggle/working/models"
available_models = []
if os.path.exists(models_dir):
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
            available_models.append(item)

# --- 4. XÂY DỰNG TIÊU ĐỀ THEO PHONG CÁCH GOOGLE ---
st.markdown('<div class="main-title">Interactive Model Unlearning Control</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">So sánh cấu trúc sinh văn bản thời gian thực giữa mô hình Fine-Tuned (Học thuộc) và Unlearned (Xóa tri thức).</div>', unsafe_allow_html=True)

if not available_models:
    st.error(f"Không tìm thấy mô hình nào trong thư mục: {models_dir}")
    st.stop()

# Cấu hình thanh Sidebar bo tròn gọn gàng
with st.sidebar:
    st.markdown("### ⚙️ Model Configurations")
    default_ft_idx = available_models.index("phi_ft_group1") if "phi_ft_group1" in available_models else 0
    model1_name = st.selectbox("Mô Hình 1 (Fine-Tuned):", available_models, index=default_ft_idx)
    
    other_models = [m for m in available_models if m != model1_name]
    default_unlearn_idx = 0 if other_models else None
    
    if other_models:
        model2_name = st.selectbox("Mô Hình 2 (Unlearned):", other_models, index=default_unlearn_idx)
    else:
        model2_name = None

# --- 5. CƠ CHẾ AUTO-LOAD TRÊN NỀN GPU ---
if model1_name and model2_name:
    with st.spinner("Đang tối ưu hóa bộ đệm GPU cho cả 2 mô hình..."):
        model1, tok1 = load_model(os.path.join(models_dir, model1_name))
        model2, tok2 = load_model(os.path.join(models_dir, model2_name))

# --- 6. KHUNG NHẬP LIỆU VÀ XỬ LÝ SINH VĂN BẢN ---
# Để trống nhãn để tạo ô prompt giống hệt thanh tìm kiếm của Gemini
question = st.text_input("", placeholder="Nhập prompt câu hỏi tại đây (Ví dụ: Who are the members of Group 1?)...")
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

if st.button("🚀 Kích hoạt đối chiếu (Generate)", type="primary"):
    if question and model1_name and model2_name:
        prompt = f"Question: {question}\nAnswer:"
        
        # Cột 1: Xử lý và hiển thị card Fine-Tuned chuyên nghiệp
        with col1:
            inputs1 = tok1(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs1 = model1.generate(
                    inputs1.input_ids.to(model1.device),
                    attention_mask=inputs1.attention_mask.to(model1.device),
                    max_new_tokens=50,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=tok1.eos_token_id,
                    pad_token_id=tok1.eos_token_id
                )
            ans1 = tok1.decode(outputs1[0][inputs1.input_ids.shape[-1]:], skip_special_tokens=True).strip()
            
            # Ép mã HTML tùy biến để tạo block phản hồi xanh lam đổ bóng mượt mà
            st.markdown(f"""
                <div class="model-card ft-card">
                    <div class="card-header">🟢 {model1_name.upper()} (Học thuộc dữ liệu)</div>
                    {ans1}
                </div>
            """, unsafe_allow_html=True)

        # Cột 2: Xử lý và hiển thị card Unlearned màu vàng sang trọng
        with col2:
            inputs2 = tok2(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs2 = model2.generate(
                    inputs2.input_ids.to(model2.device),
                    attention_mask=inputs2.attention_mask.to(model2.device),
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.4,
                    top_k=20,
                    top_p=0.85,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                    eos_token_id=tok2.eos_token_id,
                    pad_token_id=tok2.eos_token_id
                )
            ans2 = tok2.decode(outputs2[0][inputs2.input_ids.shape[-1]:], skip_special_tokens=True).strip()
            
            st.markdown(f"""
                <div class="model-card unlearn-card">
                    <div class="card-header">🔴 {model2_name.upper()} (Đã xóa trí nhớ)</div>
                    {ans2}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Vui lòng nhập nội dung câu hỏi trước khi chạy.")
