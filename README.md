# Automated Depression Detection from Multimodal Data
---

## 🧠 Overview
This project extends the **Multi Fine-Grained Fusion Network (MFFNet)** for **automatic depression detection** using **text**, **audio**, and **visual** data from the **DAIC-WOZ** dataset.  
The model introduces a **trilinear gating mechanism** and **SimAM attention** to improve feature fusion and interpretability.

---

## ⚙️ Methods
- **Dataset:** DAIC-WOZ (108 train / 36 val / 48 test sessions)  
- **Modalities:**  
  - Text → Sentence-T5 embeddings (PCA-reduced)  
  - Audio → WavLM embeddings (PCA-reduced)  
  - Visual → Gaze, pose, and facial AUs  
- **Architecture:**  
  - Multi-scale encoders (MS-Fastformer)  
  - Trilinear gating for cross-modal fusion  
  - Recurrent Pyramid & Adaptive Fusion modules  

---

## 📈 Results
| Model | Precision | Recall | F1 | Accuracy |
|:------|:----------:|:------:|:--:|:---------:|
| Text only | 0.79 | 0.77 | 0.77 | 0.79 |
| Audio only | 0.70 | 0.65 | 0.66 | 0.65 |
| Video only | 0.70 | 0.63 | 0.65 | 0.63 |
| Text + Audio | 0.84 | 0.80 | 0.81 | 0.80 |
| **Trimodal (Ours)** | **0.88** | 0.80 | 0.81 | 0.80 |

**Conclusion:**  
Text is the most informative single modality. Adding audio improves recall; visual cues enhance precision by reducing false positives.

---

## 📚 Reference
Zhou et al., *ACM TOMM*, 2024 — “Multi Fine-Grained Fusion Network for Depression Detection.”  
