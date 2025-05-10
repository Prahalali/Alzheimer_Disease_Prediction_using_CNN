# 🧠 Alzheimer's Disease Detection using Deep Learning

This project aims to develop a deep learning model for the **early detection of Alzheimer’s Disease** using MRI brain scans. By leveraging **TensorFlow** and a **customized VGG-16 Convolutional Neural Network (CNN)**, the model classifies MRI images with high accuracy to support early medical diagnosis.

---

## 📌 Project Highlights

- ✅ Built an image classification model using **TensorFlow** and **VGG-16** with custom layers for improved feature extraction.
- 🧪 Trained on **15,000+ pre-processed MRI scans** using data augmentation techniques to improve model generalization.
- 🎯 Achieved **92% classification accuracy** in detecting Alzheimer’s stages.
- 📊 Used **confusion matrices** and **heatmaps** to interpret predictions and fine-tune model performance.
- 🚀 Optimized **CNN architecture** and hyperparameters to enhance scalability and efficiency.

---

## 📂 Dataset

- Source: [Kaggle Alzheimer’s MRI Dataset](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset) *(or replace with actual source if different)*
- Classes: `Non-Demented`, `Very Mild Demented`, `Mild Demented`, `Moderate Demented`
- Total Images: 15,000+

---

## 🛠️ Tech Stack

- **Programming Language**: Python  
- **Libraries**: TensorFlow, Keras, NumPy, Matplotlib, Seaborn  
- **Model**: Pre-trained VGG-16 CNN with custom dense layers  
- **Visualization**: Confusion Matrix, Heatmaps

---

## 🧪 Model Architecture

- Input Layer: Resized MRI scans (e.g., 224x224 pixels)
- Base Model: VGG-16 (with frozen convolutional base)
- Custom Layers: Additional Dense, Dropout, and Softmax layers
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Metrics: Accuracy
