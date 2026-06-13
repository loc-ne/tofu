import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # 1. Xác định đường dẫn thư mục lưu trữ các mô hình
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    models_dir = os.path.join(BASE_DIR, "..", "models")
    
    print("=" * 80)
    print("      INTERACTIVE CPU DEMO: COMPARING FINE-TUNED VS UNLEARNED MODELS")
    print("=" * 80)
    
    # 2. Quét tìm các thư mục chứa mô hình khả dụng (có config.json)
    if not os.path.exists(models_dir):
        print(f"[ERROR] Models directory not found at: {models_dir}")
        print("Please ensure your models are saved in the 'models' folder.")
        return

    available_models = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            # Kiểm tra xem thư mục có chứa cấu hình mô hình hợp lệ không
            if os.path.exists(os.path.join(item_path, "config.json")):
                available_models.append(item)
                
    if not available_models:
        print("[ERROR] No valid model directories containing 'config.json' found.")
        print(f"Please check your 'models' folder: {models_dir}")
        return

    print("Detected models in 'models/' folder:")
    for idx, name in enumerate(available_models):
        print(f"  [{idx + 1}] {name}")
    print("=" * 80)

    # 3. Lựa chọn chế độ chạy thử nghiệm (Nạp 1 mô hình hoặc So sánh song song)
    print("How would you like to run the demo?")
    print("  [1] Single Model Query Mode")
    print("  [2] Side-by-Side Comparison Mode (Compare Fine-Tuned vs Unlearned)")
    
    mode = input("Select mode (1 or 2, default is 2): ").strip()
    if mode not in ["1", "2"]:
        mode = "2"
        
    models_to_load = []
    
    if mode == "1":
        choice = input(f"Select model number to load (1-{len(available_models)}): ").strip()
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_models):
                models_to_load.append((available_models[choice_idx], os.path.join(models_dir, available_models[choice_idx])))
            else:
                print("[WARNING] Invalid choice. Defaulting to first model.")
                models_to_load.append((available_models[0], os.path.join(models_dir, available_models[0])))
        except ValueError:
            print("[WARNING] Invalid choice. Defaulting to first model.")
            models_to_load.append((available_models[0], os.path.join(models_dir, available_models[0])))
            
    else:
        # Chế độ so sánh song song: Định vị trước mô hình fine-tuned gốc
        ft_model_name = "phi_ft_group1"
        ft_path = os.path.join(models_dir, ft_model_name)
        
        if ft_model_name in available_models:
            models_to_load.append((ft_model_name, ft_path))
        else:
            print(f"[NOTE] Default fine-tuned model '{ft_model_name}' not automatically matched.")
            choice_ft = input(f"Select FINE-TUNED model number (1-{len(available_models)}): ").strip()
            try:
                ft_idx = int(choice_ft) - 1
                if 0 <= ft_idx < len(available_models):
                    models_to_load.append((available_models[ft_idx], os.path.join(models_dir, available_models[ft_idx])))
                else:
                    models_to_load.append((available_models[0], os.path.join(models_dir, available_models[0])))
            except ValueError:
                models_to_load.append((available_models[0], os.path.join(models_dir, available_models[0])))
                
        # Lựa chọn mô hình đã unlearn để tiến hành đối chiếu kết quả
        other_models = [m for m in available_models if m != models_to_load[0][0]]
        if not other_models:
            print("[WARNING] Only one model is available in the models directory. Falling back to Single Model mode.")
            mode = "1"
        else:
            print("\nSelect the UNLEARNED model to compare against:")
            for idx, name in enumerate(other_models):
                print(f"  [{idx + 1}] {name}")
            choice_unlearn = input(f"Select model number (1-{len(other_models)}): ").strip()
            try:
                unlearn_idx = int(choice_unlearn) - 1
                if 0 <= unlearn_idx < len(other_models):
                    models_to_load.append((other_models[unlearn_idx], os.path.join(models_dir, other_models[unlearn_idx])))
                else:
                    models_to_load.append((other_models[0], os.path.join(models_dir, other_models[0])))
            except ValueError:
                models_to_load.append((other_models[0], os.path.join(models_dir, other_models[0])))

    # 4. Nạp các mô hình đã chọn lên bộ nhớ RAM (chạy CPU để tối đa hóa tương thích)
    loaded_models = {}
    
    print("\n" + "-" * 80)
    print("Loading models to CPU. Please wait...")
    print("-" * 80)
    
    # Nạp mô hình thứ nhất
    name1, path1 = models_to_load[0]
    print(f"Loading Model 1: '{name1}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(path1)
        # Nạp mô hình với định dạng float32 để chạy ổn định trên CPU
        model1 = AutoModelForCausalLM.from_pretrained(
            path1,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to("cpu")
        model1.eval()
        loaded_models[name1] = (model1, tokenizer)
        print(f"[OK] Loaded '{name1}' successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model '{name1}': {e}")
        return

    # Nạp mô hình thứ hai nếu chạy ở chế độ so sánh song song
    if mode == "2" and len(models_to_load) > 1:
        name2, path2 = models_to_load[1]
        print(f"\nLoading Model 2: '{name2}'...")
        try:
            tokenizer2 = AutoTokenizer.from_pretrained(path2)
            model2 = AutoModelForCausalLM.from_pretrained(
                path2,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to("cpu")
            model2.eval()
            loaded_models[name2] = (model2, tokenizer2)
            print(f"[OK] Loaded '{name2}' successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load model '{name2}': {e}")
            print("[NOTE] Continuing in Single Model mode with the first loaded model.")
            mode = "1"

    print("\n" + "=" * 80)
    print("   ALL MODELS LOADED SUCCESSFULLY ON CPU!")
    print("   Ready for live queries. Type 'exit' or 'quit' to close the program.")
    print("=" * 80 + "\n")

    # 5. Vòng lặp nhận câu hỏi và sinh câu trả lời trực tiếp
    while True:
        question = input("Question: ").strip()
        if not question:
            continue
        if question.lower() in ["exit", "quit", "q"]:
            print("Exiting live demo. Goodbye.")
            break
            
        prompt = f"Question: {question}\nAnswer:"
        
        # Trường hợp chạy kiểm thử 1 mô hình đơn lẻ
        if mode == "1":
            name = models_to_load[0][0]
            model, tok = loaded_models[name]
            
            inputs = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.4,
                    top_k=20,
                    top_p=0.85,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.eos_token_id
                )
            generated_text = tok.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
            
            print("-" * 80)
            print(f"[{name} Response]:")
            print(generated_text)
            print("-" * 80 + "\n")
            
        # Trường hợp chạy đối chiếu song song 2 mô hình (Fine-Tuned vs Unlearned)
        else:
            name1, name2 = models_to_load[0][0], models_to_load[1][0]
            model1, tok1 = loaded_models[name1]
            model2, tok2 = loaded_models[name2]
            
            # Truy vấn sinh câu trả lời từ Mô hình 1 (Fine-tuned baseline)
            inputs1 = tok1(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs1 = model1.generate(
                    inputs1.input_ids,
                    attention_mask=inputs1.attention_mask,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.4,
                    top_k=20,
                    top_p=0.85,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                    eos_token_id=tok1.eos_token_id,
                    pad_token_id=tok1.eos_token_id
                )
            ans1 = tok1.decode(outputs1[0][inputs1.input_ids.shape[-1]:], skip_special_tokens=True).strip()
            
            # Truy vấn sinh câu trả lời từ Mô hình 2 (Unlearned model)
            inputs2 = tok2(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs2 = model2.generate(
                    inputs2.input_ids,
                    attention_mask=inputs2.attention_mask,
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
            
            # Hiển thị kết quả đối chiếu có cấu trúc rõ ràng
            print("\n" + "=" * 80)
            print(f"QUESTION: {question}")
            print("-" * 80)
            print(f" [MÔ HÌNH 1: {name1.upper()}] (FINE-TUNED BASELINE)")
            print(f"  -> {ans1}")
            print("-" * 80)
            print(f" [MÔ HÌNH 2: {name2.upper()}] (UNLEARNED MODEL)")
            print(f"  -> {ans2}")
            print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
