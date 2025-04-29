# 💼 CodeClause Internship Projects  
### 👩‍💻 Submitted by: Esha Alvi

This repository contains the projects completed as part of my **AI/ML Internship with CodeClause**. Each project demonstrates practical applications of artificial intelligence, machine learning, and natural language processing using Python.

---

## 📁 Projects Overview

### ✋ 1. Gesture Volume Control using Hand Tracking

**🎯 Aim**  
Use hand gestures to control system volume in real-time using webcam input.

**📝 Description**  
This project leverages hand landmark detection using MediaPipe to measure the distance between fingers and adjust the system volume accordingly. It uses OpenCV for video processing and Pycaw for system audio control.

**🛠 Technologies**  
- Python  
- OpenCV  
- MediaPipe  
- Pycaw (Windows audio control)

**📚 What I Learned**  
- Real-time hand tracking  
- Gesture recognition logic  
- Controlling system-level audio using Python

---

### 🔢 2. Handwritten Digit Sequence Recognition

**🎯 Aim**  
Recognize handwritten digits and sequences using deep learning.

**📝 Description**  
A Convolutional Neural Network (CNN) model is trained on the MNIST dataset to recognize digits. The model then attempts to identify randomly generated sequences of handwritten digits.

**🛠 Technologies**  
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib

**📚 What I Learned**  
- Building and training CNNs  
- Image data preprocessing  
- Inference on multiple-digit sequences

---

### 💬 3. Sentiment Analysis Tool

**🎯 Aim**  
Create a tool that determines the sentiment of a given text as Positive, Negative, or Neutral.

**📝 Description**  
This simple yet powerful tool uses the TextBlob NLP library to analyze the sentiment of user-input text and outputs its classification.

**🛠 Technologies**  
- Python  
- TextBlob

**📚 What I Learned**  
- Natural Language Processing fundamentals  
- Sentiment polarity and classification  
- Using pre-trained NLP models

---

## ⚙️ How to Run the Projects

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m textblob.download_corpora
