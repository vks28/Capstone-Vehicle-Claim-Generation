
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
