import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# --- CẤU HÌNH GIAO DIỆN WEB ---
st.set_page_config(
    page_title="Unlearning Model Demo",
    page_icon="🧠",
    layout="wide"
)

# --- 1. HÀM LOAD MÔ HÌNH ---
@st.cache_resource
def load_model(model_path):
    print("\n" + "="*50)
    print(f"[LOG] Bắt đầu kích hoạt hàm load_model cho đường dẫn: {model_path}")
    print("="*50)
    
    try:
        print("[LOG] Bước 1: Đang nạp Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        print("[LOG] -> Nạp Tokenizer THÀNH CÔNG.")
        
        print("[LOG] Bước 2: Đang đọc file cấu hình AutoConfig...")
        config = AutoConfig.from_pretrained(model_path)
        config.pad_token_id = tokenizer.pad_token_id
        print("[LOG] -> Đọc Config THÀNH CÔNG.")
        
        print("[LOG] Bước 3: Đang tải trọng số Model vào RAM (Dtype: float16)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            dtype=torch.float16,
            trust_remote_code=True
        )
        print("[LOG] -> Tải trọng số vào bộ nhớ đệm THÀNH CÔNG.")
        
        print("[LOG] Bước 4: Đang ép mô hình lên chip đồ họa GPU (.to('cuda'))...")
        model = model.to("cuda")
        print("[LOG] -> Đẩy lên GPU THÀNH CÔNG!")
        
        model.generation_config.do_sample = True
        model.config.use_cache = True
        model.eval()
        
        print("[LOG] === HOÀN THÀNH TẢI MÔ HÌNH SẴN SÀNG CHẠY ===")
        return model, tokenizer
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Quá trình load mô hình thất bại tại bước trên!")
        print(f"[ERROR DETAILS]: {str(e)}")
        raise e

# --- 2. QUÉT THƯ MỤC TÌM MÔ HÌNH ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
models_dir = "/kaggle/working/models"

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

with st.sidebar:
    st.header("⚙️ Cấu hình Mô hình")
    default_ft_idx = available_models.index("phi_ft_group1") if "phi_ft_group1" in available_models else 0
    model1_name = st.selectbox("Chọn Mô Hình 1 (Fine-Tuned - Học thuộc):", available_models, index=default_ft_idx)
    
    other_models = [m for m in available_models if m != model1_name]
    default_unlearn_idx = 0 if other_models else None
    
    if other_models:
        model2_name = st.selectbox("Chọn Mô Hình 2 (Unlearned - Đã xóa trí nhớ):", other_models, index=default_unlearn_idx)
    else:
        st.warning("Chỉ có 1 mô hình. Hãy thêm mô hình unlearn vào thư mục models.")
        model2_name = None

# --- 4. XỬ LÝ LOGIC SINH VĂN BẢN ---
question = st.text_input("Nhập câu hỏi của bạn (Ví dụ: Who are the members of Group 1?):")
col1, col2 = st.columns(2)

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
                        do_sample=False,
                        num_beams=1,
                        eos_token_id=tok1.eos_token_id,
                        pad_token_id=tok1.eos_token_id
                    )
                ans1 = tok1.decode(outputs1[0][inputs1.input_ids.shape[-1]:], skip_special_tokens=True).strip()
            st.info(ans1)

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
            st.warning(ans2)
            
    else:
        st.warning("Vui lòng nhập câu hỏi và đảm bảo đã chọn đủ 2 mô hình.")
