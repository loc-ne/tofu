import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CẤU HÌNH GIAO DIỆN WEB ---
st.set_page_config(
    page_title="Unlearning Model Demo",
    page_icon="🧠",
    layout="wide" # Sử dụng toàn bộ chiều rộng màn hình để chia đôi cho đẹp
)

# --- 1. HÀM LOAD MÔ HÌNH (CÓ CACHE) ---
# Hàm này dùng @st.cache_resource để chỉ load mô hình 1 lần duy nhất,
# các lần bấm nút sau sẽ không bị load lại gây tốn thời gian.
@st.cache_resource
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to("cuda")
    model.eval()
    return model, tokenizer

# --- 2. QUÉT THƯ MỤC TÌM MÔ HÌNH ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
models_dir = "/kaggle/input/datasets/arisene/tofuye/models"

available_models = []
if os.path.exists(models_dir):
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
            available_models.append(item)

# --- 3. XÂY DỰNG GIAO DIỆN (UI) ---
st.title("⚖️ Interactive Demo: Fine-Tuned vs Unlearned Models")
st.markdown("Nhập câu hỏi để so sánh trực tiếp kết quả sinh văn bản giữa hai mô hình.")

if not available_models:
    st.error(f"Không tìm thấy mô hình nào trong thư mục: {models_dir}")
    st.stop()

# Thanh bên (Sidebar) để chọn mô hình
with st.sidebar:
    st.header("⚙️ Cấu hình Mô hình")
    
    # Cố gắng tự động chọn model fine-tune nếu có
    default_ft_idx = available_models.index("phi_ft_group1") if "phi_ft_group1" in available_models else 0
    model1_name = st.selectbox("Chọn Mô Hình 1 (Fine-Tuned - Học thuộc):", available_models, index=default_ft_idx)
    
    # Chọn mô hình unlearn (mặc định chọn cái khác với model 1)
    other_models = [m for m in available_models if m != model1_name]
    default_unlearn_idx = 0 if other_models else None
    
    if other_models:
        model2_name = st.selectbox("Chọn Mô Hình 2 (Unlearned - Đã xóa trí nhớ):", other_models, index=default_unlearn_idx)
    else:
        st.warning("Chỉ có 1 mô hình. Hãy thêm mô hình unlearn vào thư mục models.")
        model2_name = None

# --- 4. XỬ LÝ LOGIC SINH VĂN BẢN ---
question = st.text_input("Nhập câu hỏi của bạn (Ví dụ: Who are the members of Group 1?):")
col1, col2 = st.columns(2) # Lệnh này chia đôi màn hình

if st.button("🚀 Chạy Demo (Generate)", type="primary"):
    if question and model1_name and model2_name:
        prompt = f"Question: {question}\nAnswer:"
        
        # Cột 1: Fine-tuned Model
        with col1:
            st.subheader(f"🟢 {model1_name.upper()} (Fine-Tuned)")
            with st.spinner("Đang tải mô hình & sinh câu trả lời..."):
                path1 = os.path.join(models_dir, model1_name)
                model1, tok1 = load_model(path1)
                
                inputs1 = tok1(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs1 = model1.generate(
                        inputs1.input_ids.to(model1.device),
                        attention_mask=inputs1.attention_mask.to(model1.device),
                        max_new_tokens=50,
                        # Cấu hình Greedy chuẩn để đọc data đã học vẹt
                        do_sample=False,
                        num_beams=1,
                        eos_token_id=tok1.eos_token_id,
                        pad_token_id=tok1.eos_token_id
                    )
                ans1 = tok1.decode(outputs1[0][inputs1.input_ids.shape[-1]:], skip_special_tokens=True).strip()
            
            st.info(ans1) # In kết quả vào khung màu xanh

        # Cột 2: Unlearned Model
        with col2:
            st.subheader(f"🔴 {model2_name.upper()} (Unlearned)")
            with st.spinner("Đang tải mô hình & sinh câu trả lời..."):
                path2 = os.path.join(models_dir, model2_name)
                model2, tok2 = load_model(path2)
                
                inputs2 = tok2(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs2 = model2.generate(
                        inputs2.input_ids.to(model2.device),
                        attention_mask=inputs2.attention_mask.to(model2.device),
                        max_new_tokens=50,
                        # Cấu hình Sample có độ ngẫu nhiên theo code cũ của bạn
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
                
            st.warning(ans2) # In kết quả vào khung màu vàng
            
    else:
        st.warning("Vui lòng nhập câu hỏi và đảm bảo đã chọn đủ 2 mô hình.")
