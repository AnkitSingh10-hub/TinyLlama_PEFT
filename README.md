✈️ TinyLlama-1.1B Travel Intent Recognition (PEFT/LoRA)
![alt text](https://colab.research.google.com/assets/colab-badge.svg)
📄 Project Overview
This project demonstrates Parameter-Efficient Fine-Tuning (PEFT) of the TinyLlama-1.1B Large Language Model.
The objective is to adapt a general-purpose LLM into a specialized Travel Intent Classifier capable of understanding user requests (e.g., booking flights, finding hotels) while running on consumer-grade hardware using 4-bit Quantization (QLoRA).
🛠️ Tech Stack (CV Verification)
This implementation verifies the skills listed in my Curriculum Vitae:
Model: TinyLlama/TinyLlama-1.1B-Chat-v0.1
Techniques: LoRA (Low-Rank Adaptation), QLoRA (4-bit Quantization), Supervised Fine-Tuning (SFT).
Libraries: Hugging Face Transformers, PyTorch, PEFT, TRL, BitsAndBytes.
Data Engineering: Implemented a balanced sampling strategy to mitigate class imbalance in the training dataset.
📊 Methodology
Data Loading: Utilized the bitext/Bitext-travel-llm-chatbot-training-dataset.
Quantization: Loaded the model in 4-bit precision (NF4) using BitsAndBytesConfig to reduce memory usage by ~70%.
LoRA Configuration: Applied low-rank adapters (r=16, alpha=32) to the query (q_proj) and value (v_proj) projection layers.
Training: Fine-tuned using the SFTTrainer for 60 steps to learn the specific intent-response format.
🚀 Results & Inference
After fine-tuning, the model can accurately classify travel-related queries and generate structured responses.
Example Input:
"Query: I need to buy a ticket to Kathmandu"
Model Output:
Intent: flight_booking
Response: I can help you find flights to Kathmandu. What are your travel dates?
💻 How to Run
The easiest way to run this code is via Google Colab using the free T4 GPU tier.
Click the "Open in Colab" badge above.
Connect to a T4 GPU Runtime.
Run all cells to replicate the training and inference pipeline.
📝 Note on Hardware
This project uses Quantization, requiring a GPU with CUDA support (e.g., NVIDIA T4, RTX 3060+). It is optimized to run entirely within the Google Colab Free Tier.
