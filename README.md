# 🚨 AI-Powered Exam Malpractice Detection System (YOLOv8 + Behavioral Intelligence)

> **A real-time, behavior-aware AI proctoring system designed to detect exam malpractice using computer vision, temporal reasoning, and system-level intelligence.**

## 🌟 Quick Status

| Status | Value |
| :--- | :--- |
| **Model Used** | YOLOv8n |
| **Inference** | Real-Time (Webcam / CPU) |
| **Core Logic** | Hybrid AI + Rule-Based Temporal Reasoning |
| **Output States**| SAFE / SUSPICIOUS / MALPRACTICE |

---

## 🛠️ Tech Stack & Dependencies

| Category | Technology | Icon / Badge |
| :--- | :--- | :--- |
| **Deep Learning** | Python, Ultralytics YOLOv8 | <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge"> <img src="https://img.shields.io/badge/YOLOv8-000000?style=for-the-badge&logo=yolo&logoColor=white" alt="YOLOv8 Badge"> |
| **Computer Vision** | OpenCV | <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV Badge"> |
| **Numerical/Data** | NumPy, Collections (deque) | <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy Badge"> |
| **Environment** | OS Independent | `Linux` `Windows` `macOS` |

<br>

<p align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge">
    <img src="https://img.shields.io/badge/YOLOv8-000000?style=for-the-badge&logo=yolo&logoColor=white" alt="YOLOv8 Badge">
    <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV Badge">
    <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy Badge">
</p>

---

## 📌 Project Overview

This project presents a **real-time exam malpractice detection system** that goes beyond traditional object detection by combining:

* **Deep Learning (YOLOv8)** for visual feature extraction.
* **Temporal behavior analysis** using sliding windows.
* **Decision Hysteresis** to stabilize predictions.
* **Multi-signal escalation logic** for robust threat assessment.

Unlike naive object detectors, this system is designed to **operate under real-world constraints** (noisy webcams, limited datasets, motion blur) — making it a **production-inspired solution**, not a toy model.

---

## 🎯 Problem Statement

Online and remote examinations face critical challenges:
1.  Students using **mobile phones** (Hard Malpractice).
2.  Repeated **head turns** to consult external sources.
3.  Subtle **hand or upper-body movements** indicating cheating.
4.  Limitations of single-frame object detection, leading to prediction "flickering."

👉 This project addresses the problem using a **hybrid AI + rule-based behavioral pipeline** that aggregates evidence over time to make stable, reliable decisions.

---

## 🧠 System Architecture

The core architecture follows a robust, sequential processing pipeline:


# Webcam Feed
↓
# YOLOv8 Object Detection
↓
# Frame-wise Visual Cues
↓
# Temporal Behavior Buffers
↓
# Multi-Rule Decision Engine
↓
# SAFE / SUSPICIOUS / MALPRACTICE



---

## 🧪 Dataset Strategy & Challenges (IMPORTANT)

### 📂 Datasets Used

| Dataset | Purpose |
|-------|--------|
| Student-with-Phone | Mobile phone detection |
| HeadPose Dataset | Head orientation detection |
| Exam Hall Dataset | Safe / normal behavior |

### ⚠️ Critical Dataset Limitations (Real-World Issues)

This project **intentionally documents dataset flaws** instead of hiding them:

- ❌ **Severe class imbalance** (e.g., phone class had extremely few validation samples)
- ❌ Head pose data was mostly **static images**
- ❌ Noisy annotations from open-source datasets
- ❌ Domain shift between dataset images and real webcam feed

📌 **Key Insight:**  
> “A model trained on static datasets does not generalize perfectly to real-time webcam streams.”

This is a **well-known industrial challenge**, not a failure.

---

## 🔥 Why YOLOv8?

- State-of-the-art real-time object detection
- Extremely fast inference
- Easily deployable on edge devices
- Ideal backbone for **vision-based proctoring**



YOLOv8n (Nano) – optimized for real-time inference


---

## 📉 Training Summary (Highlights)

- Model trained for **50 epochs**
- Loss curves converged properly
- High performance on **head orientation classes**
- Weak performance on **mobile phone class due to data scarcity**

📌 **Important Observation:**  
> Even a well-trained model cannot compensate for insufficient data.

---

## 🧠 Key Innovation: Behavioral Intelligence Layer

### ❌ What naive systems do:
- Single-frame predictions
- Immediate decisions
- High false positives
- Flickering outputs

### ✅ What THIS system does:
- Aggregates evidence over time
- Uses **temporal buffers**
- Applies **decision hysteresis**
- Escalates gradually:


SAFE → SUSPICIOUS → MALPRACTICE


This mirrors **real-world AI monitoring systems** used in proctoring platforms.

---

## 🚦 Decision Logic (High-Level)

### 🔴 MALPRACTICE
Triggered when:
- Mobile phone is detected near the person
- Sustained head turn beyond threshold
- Repeated suspicious motion patterns

### 🟡 SUSPICIOUS
Triggered when:
- Frequent head turns
- Abnormal upper-body motion
- Behavioral inconsistency over time

### 🟢 SAFE
Triggered when:
- Normal posture
- Minimal motion
- No suspicious visual cues

---

## 🧠 Advanced Engineering Features

- **Temporal Buffers** (frame aggregation)
- **Hysteresis Locking** (prevents flickering states)
- **Intersection-over-Union (IoU)** based reasoning
- **Person-aware decision rules**
- **Motion-based anomaly detection**

📌 These techniques are commonly used in **production AI systems**, not beginner projects.

---

## 🖥️ Real-Time Webcam Inference

- Runs locally on CPU
- Stable real-time performance
- Designed for demo, testing, and presentation
- Robust to lighting variations and camera noise

---

## 🎓 Interview / Viva Justification (VERY IMPORTANT)

> “Due to dataset limitations and real-time constraints, the system uses YOLO for visual cues and a temporal behavioral reasoning layer to ensure robust malpractice detection under noisy conditions.”

This explanation demonstrates:
- Practical ML understanding
- System-level thinking
- Awareness of real-world AI limitations

---

## 📌 Why This Project Stands Out

✔ Not just model training — **system design**  
✔ Honest handling of dataset flaws  
✔ Strong engineering decisions  
✔ Real-time deployment  
✔ Industry-aligned approach  

---

## 🚀 Future Improvements

- Larger, balanced datasets
- Fine-tuned phone detection
- Face landmark-based gaze estimation
- Multi-person exam hall support
- Transformer-based temporal modeling

---

## 🏁 Final Note

This project demonstrates that **AI systems are more than models** — they are **decision-making pipelines**.

> “A weak model with strong logic often outperforms a strong model with no logic.”

---

## ⭐ If you found this interesting

Please ⭐ the repository — it motivates continued research and development!

---


Model used:
