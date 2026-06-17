import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# --- 1. CẤU HÌNH GIAO DIỆN CHUẨN GOOGLE MINIMALIST ---
st.set_page_config(
    page_title="Model Unlearning Evaluation",
    layout="wide"
)

# Khai báo các chuỗi mã SVG để nhúng trực tiếp vào HTML
SVG_FINE_TUNED = """
<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1a73e8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
"""

SVG_UNLEARNED = """
<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#5f6368" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>
"""

SVG_SEARCH = """
<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 4px;"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
"""

# Nhúng CSS thuần dạng phẳng (Flat Design) không màu mè
st.markdown("""
    <style>
        /* Khóa các thành phần thừa */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Cấu hình đĩa nền và font phẳng toàn trang */
        body, .main {
            background-color: #ffffff !important;
            color: #202124 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        .main .block-container {
            padding-top: 3rem;
            max-width: 1140px;
        }
        
        /* Tiêu đề chính dạng text phẳng Google */
        .main-title {
            font-size: 1.75rem;
            font-weight: 500;
            color: #202124;
            letter-spacing: -0.5px;
            margin-bottom: 0.25rem;
        }
        
        .sub-title {
            color: #70757a;
            font-size: 0.95rem;
            margin-bottom: 2.5rem;
        }

        /* Ô nhập liệu tinh gọn, bo tròn nhẹ bo sát viền */
        div[data-baseweb="input"] {
            border-radius: 8px !important;
            border: 1px solid #dadce0 !important;
            background-color: #ffffff !important;
            box-shadow: none !important;
            transition: border-color 0.15s ease;
        }
        div[data-baseweb="input"]:focus-within {
            border-color: #1a73e8 !important;
        }
        
        /* Dropdown cấu hình */
        div[data-baseweb="select"] {
            border-radius: 6px !important;
        }

        /* Nút bấm phẳng, không đổ bóng dày */
        div.stButton > button:first-child {
            border-radius: 6px !important;
            padding: 0.4rem 1.5rem !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
            background-color: #1a73e8 !important;
            color: #ffffff !important;
            border: 1px solid #1a73e8 !important;
            box-shadow: none !important;
        }
        div.stButton > button:first-child:hover {
            background-color: #1557b0 !important;
            border-color: #1557b0 !important;
        }

        /* Khung hiển thị kết quả dạng bảng tin tối giản */
        .model-card {
            padding: 1.25rem;
            border-radius: 8px;
            border: 1px solid #dadce0;
            background-color: #f8f9fa;
            margin-top: 1rem;
            font-size: 0.95rem;
            line-height: 1.58;
            color: #202124;
        }
        .card-header {
            font-weight: 600;
            font-size: 0.85rem;
            color: #202124;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            border-bottom: 1px solid #e8eaed;
            padding-bottom: 0.5rem;
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

# --- 4. XÂY DỰNG KHUNG TIÊU ĐỀ PHẲNG ---
st.markdown('<div class="main-title">Model Knowledge Unlearning Interface</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hệ thống đối chiếu song song cấu trúc dữ liệu đầu ra giữa mô hình gốc và mô hình loại bỏ tri thức.</div>', unsafe_allow_html=True)

if not available_models:
    st.error(f"Không tìm thấy mô hình nào trong thư mục: {models_dir}")
    st.stop()

with st.sidebar:
    st.markdown("### Configurations")
    default_ft_idx = available_models.index("phi_ft_group1") if "phi_ft_group1" in available_models else 0
    model1_name = st.selectbox("Model 1 (Fine-Tuned):", available_models, index=default_ft_idx)
    
    other_models = [m for m in available_models if m != model1_name]
    default_unlearn_idx = 0 if other_models else None
    
    if other_models:
        model2_name = st.selectbox("Model 2 (Unlearned):", other_models, index=default_unlearn_idx)
    else:
        model2_name = None

# --- 5. CƠ CHẾ AUTO-LOAD TRÊN NỀN GPU ---
if model1_name and model2_name:
    with st.spinner("Đang chuẩn bị bộ đệm phần cứng..."):
        model1, tok1 = load_model(os.path.join(models_dir, model1_name))
        model2, tok2 = load_model(os.path.join(models_dir, model2_name))

# --- 6. XỬ LÝ LOGIC SINH VĂN BẢN ---
question = st.text_input("", placeholder="Nhập câu hỏi truy vấn dữ liệu...")
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# Khởi tạo nút bấm chứa icon SVG Search thanh mảnh bên trong
btn_label = f"Kích hoạt đối chiếu"
if st.button(btn_label, type="primary"):
    if question and model1_name and model2_name:
        prompt = f"Question: {question}\nAnswer:"
        
        # Cột 1: Hiển thị hộp Fine-Tuned thô phẳng kèm SVG
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
            
            st.markdown(f"""
                <div class="model-card">
                    <div class="card-header">{SVG_FINE_TUNED} MÔ HÌNH FINE-TUNED (RETRAINED)</div>
                    {ans1}
                </div>
            """, unsafe_allow_html=True)

        # Cột 2: Hiển thị hộp Unlearned thô phẳng kèm SVG ẩn mắt
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
                <div class="model-card">
                    <div class="card-header">{SVG_UNLEARNED} MÔ HÌNH UNLEARNED (FORGOTTEN)</div>
                    {ans2}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Vui lòng nhập nội dung câu hỏi.")
