---
## 🚗 Project Overview

The development of reliable AI systems for vehicle damage assessment is hindered by the scarcity, imbalance, and high acquisition cost of real-world datasets. This project proposes a synthetic data generation pipeline that leverages a fine-tuned text-to-image model to produce high-fidelity car damage images. By employing the LoRA (Low-Rank Adaptation) technique, we fine-tune specialized models for specific damage types and integrate them using a Mixture of Experts (MoE) framework to support prompt-driven image generation.

This work addresses several critical analytical challenges. These include the generation of visually realistic and diverse damage characteristics, accurate alignment between textual prompts and visual outputs, effective integration of multiple LoRA experts without degradation in image quality, and controlled variability and consistency in the generated outputs.

Through this approach, the project facilitates scalable and efficient image synthesis, supporting robust downstream applications such as damage classification, detection, and automated claims processing in automotive and insurance domains.


---

## 📂 Dataset Reference

This project uses a filtered subset of the [Car Damage Detection (CarDD) Dataset](https://cardd-ustc.github.io/).

- Only damage categories of **scratch** and **glass shatter** were selected.
- Captions were generated using LLaMA 3.2 and refined manually.
- The full dataset is publicly available via the above link.

---

## 📊 Evaluation Metrics

Image quality was evaluated using:
- **FID Score**: Measures similarity between generated and real images.
- **Visual Assessment**: Human inspection for realism and damage accuracy.

---

## 📌 Key Features

- Trained two expert LoRA models: one for scratches, another for glass damage.
- Evaluated the experts using FID metric.
- explored various methods to merge the two experts.
- Generated synthetic images based on descriptive captions.
- 

---

## 📎 Model Access

Due to GitHub's file size limits, trained model weights are hosted externally:

- [Download Scratch  and Glass Shattered Expert Model (.safetensors)](https://drive.google.com/drive/folders/1dM2Y0ldgfeIxbfN4UnuX9UGRHDrUKFfv?usp=sharing)

---

## 📈 Results Snapshot

| Damage Type   | FID Score ↓ (Lower is better) |
|---------------|-------------------------------|
| Scratch       | 206                        |
| Shattered Glass | 228                     |



---

## 🧩 Future Enhancements

- Merge multiple experts using dynamic Mixture of Experts (MoE) techniques
- Improve captioning using multi-modal models like BLIP or GIT
- Explore style transfer to vary lighting, angle, and environment

---

## 📌 License

This project is for academic and research use only.
