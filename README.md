# Vision-ROI-Reconstructor

## 📌 Project Overview
[cite_start]This project is developed as part of the **Spring 2026 AI Lab**[cite: 1, 2]. It implements a specialized computer vision pipeline designed to perform character detection and spatial pixel manipulation within high-dynamic video environments. 

[cite_start]The system leverages **YOLO** (You Only Look Once) for precise character localization, extracts the relevant Regions of Interest (ROIs), applies digital transformations, and reconstructs the original video frames with the modified pixel data[cite: 5, 6].

## 🛠️ Task 01 Objectives
[cite_start]According to the lab requirements[cite: 6]:
* **Detection**: Identify and localize characters in the provided video datasets.
* **Bounding Boxes**: Render visual markers around detected entities.
* **ROI Extraction**: Isolate the specific pixel regions (Characters) from the background.
* **Transformation & Re-insertion**: Apply required visual processing to the ROIs and seamlessly merge them back into the source frames.

## 📂 Project Structure
* `/data`: Input source videos (`uncharted.mp4`, `lotr.mp4`, `horizon.mp4`, etc.).
* `/src`: Core Python scripts for detection and reconstruction.
* `/output`: Final processed videos with integrated transformations.

## 🚀 Technical Stack
* [cite_start]**Object Detection**: YOLO [cite: 5]
* **Image Processing**: OpenCV
* **Video Handling**: MoviePy / FFmpeg
* **Language**: Python

## 📝 Justification of Design Choices
* [cite_start]**YOLO Selection**: Chosen for its high inference speed and accuracy in detecting human-like characters in gaming and cinematic footage[cite: 5].
* [cite_start]**ROI Extraction**: By isolating only the character pixels, we ensure that transformations are applied surgically without affecting the surrounding environment[cite: 6].
* [cite_start]**Modular Pipeline**: The code is structured to be reproducible, allowing for different transformation functions to be "plugged in" before the reconstruction phase[cite: 72].

---
[cite_start]*Developed by Bilal Abbasi (Reference: AI Lab OEL [cite: 3])*
