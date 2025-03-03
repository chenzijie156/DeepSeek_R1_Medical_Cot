  # DeepSeek_R1_Medical_Cot
🚀 Fine-tuning DeepSeek R1 for Medical Chain-of-Thought Reasoning  

This repository contains Jupyter notebooks and scripts for fine-tuning **DeepSeek-R1** on **medical reasoning tasks** using **Chain-of-Thought (CoT) prompting**. The goal is to enhance the model’s ability to provide **step-by-step medical explanations and reasoning**.  

---

## 📌 Project Overview  
This project fine-tunes **DeepSeek-R1** using **QLoRA** with medical datasets that emphasize **logical reasoning** in clinical diagnosis, treatment recommendations, and physiological insights.  

---

## 🔹 Features  
- **Few-shot Medical Chain-of-Thought (CoT) Reasoning**  
- **QLoRA-based fine-tuning for efficiency**  
- **Unsloth Optimization:** 2x faster fine-tuning with reduced memory usage  
- **Support for 4-bit / 8-bit quantization**  
- **Model merging & deployment for inference**  

---

## 📂 Files in this Repository  

| File                                      | Description |
|-------------------------------------------|-------------|
| `DeepSeek_Medical_HuggingFace.ipynb`      | Upload fine-tuned model to Hugging Face Hub |
| `DeepSeek_R1_App.ipynb`                   | Interactive Gradio app for testing model inference |
| `DeepSeek_R1_Medical_CoT_Large.ipynb`     | Large-scale fine-tuning on medical datasets |
| `DeepSeek_R1_Medical_CoT_Tiny.ipynb`      | Small-scale fine-tuning experiment |
| `DeepSeek_R1_medical_CoT_Testing.ipynb`   | Testing the model on medical reasoning tasks |
| `LICENSE`                                 | MIT License |
| `README.md`                               | This file |

---

## 📊 Training Details  

- **Base Model:** DeepSeek-R1-Distill-Llama-8B  
- **Fine-tuning Method:** QLoRA + Unsloth  
- **Dataset:** Medical CoT dataset (e.g., `medical-o1-reasoning-SFT`)  
- **Compute:** Google Colab Pro with A100 (40GB)  
- **Quantization:** 4-bit LoRA adapters, later merged into 16-bit  

---

## 🛠 How to Use  

### 1️⃣ Load the fine-tuned model
To use the fine-tuned model, run:  

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_name = "your-huggingface-username/DeepSeek-R1-Medical-CoT"

tokenizer = AutoTokenizer.from_pretrained(repo_name)
model = AutoModelForCausalLM.from_pretrained(repo_name)

model.eval()
```

### 2️⃣ Run inference

```python
prompt = "What are the early symptoms of diabetes?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Model Response:", response)
```
### 📢 Acknowledgments
- **DeepSeek-AI for releasing DeepSeek-R1**
- **Unsloth for optimized LoRA fine-tuning**
- **Hugging Face for hosting the models**
