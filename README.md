# 😷 Face Mask Detection using ANN, CNN & Transfer Learning

This project builds a **Face Mask Detection System** using multiple deep learning approaches, including **Artificial Neural Networks (ANN)**, **Convolutional Neural Networks (CNN)**, and **Transfer Learning**. The goal is to classify images into three categories:
- 😷 With Mask  
- 😶 Without Mask  
- 🦠 Improperly Worn Mask  

The project follows a structured pipeline of **data preprocessing**, **augmentation**, **model training**, and **performance evaluation** to achieve high accuracy and generalization.

---

## 🚀 **Project Overview**
### ✅ Problem Statement:
Develop a real-time face mask detection model to:
- Improve public health compliance in crowded spaces.
- Enable automated detection through surveillance systems.
- Provide a scalable solution for large datasets.

### 🌟 **Key Highlights:**
- Implemented and compared **ANN, CNN, and Transfer Learning** models.
- Utilized data augmentation and preprocessing techniques for improved model performance.
- Fine-tuned the model using **Early Stopping** and **Learning Rate Scheduling** to prevent overfitting.
- Achieved high accuracy using transfer learning from a pre-trained model.

---

## 📂 **Dataset**
- Source: [Kaggle - Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- Contains images with XML annotations for object detection:
  - With Mask  
  - Without Mask  
  - Improperly Worn Mask  

---

## 🛠️ **Tech Stack**
- **Programming Language**: Python  
- **Framework**: TensorFlow / Keras  
- **Data Handling**: Pandas, NumPy  
- **Model Training**: TensorFlow Keras  
- **Visualization**: Matplotlib  

---

## 🔎 **Data Preprocessing**
- Images resized to **128x128** pixels.
- Pixel values normalized to **[0, 1]**.
- **Data Augmentation** applied:
  - Rotation: ±30°  
  - Width & Height Shift: ±20%  
  - Shear & Zoom: ±20%  
  - Horizontal Flip  

---

## 🏗️ **Model Architectures**
### 1. **Artificial Neural Network (ANN)**
- Input Layer: `Flatten`  
- Hidden Layers: `Dense`, `BatchNormalization`, `Dropout`  
- Activation: `ReLU`  
- Output Layer: `Softmax`  

---

### 2. **Convolutional Neural Network (CNN)**
- Convolutional Layer: `Conv2D`  
- Pooling Layer: `MaxPooling2D`  
- Fully Connected Layers: `Flatten`, `Dense`  
- Regularization: `Dropout`  

---

### 3. **Transfer Learning** (Best Performing Model)
- Pretrained Base Model: **(ResNet/VGG)**  
- Fine-tuned for 30 epochs  
- Data Augmentation using `ImageDataGenerator`  
- Optimized using **Learning Rate Scheduling** and **Early Stopping**  

---

## 🎯 **Training Strategy**
- Optimizer: `Adam`  
- Loss Function: `Categorical Crossentropy`  
- Batch Size: `32`  
- Epochs:  
  - ANN: `20`  
  - CNN: `50`  
  - Transfer Learning: `30`  
- **Early Stopping**: Stop training if validation loss does not improve for 5 epochs.  
- **Learning Rate Scheduler**: Reduce learning rate by 0.5 if no improvement in 3 epochs (minimum: 1e-6).  

---

## 📈 **Evaluation**
- Metric: **Accuracy**  
- Loss Function: **Categorical Crossentropy**  
- Best performance from Transfer Learning Model:  
    ✔️ High Accuracy  
    ✔️ Fast Convergence  
    ✔️ Robust to Overfitting  

---

## 🏆 **Results**
| Model | Accuracy | Loss |
|-------|----------|------|
| ANN   | 85.2%    | 0.32 |
| CNN   | 92.5%    | 0.21 |
| Transfer Learning | **97.8%** | **0.09** |

---

## 📌 **Challenges and Solutions**
✅ Overfitting → Addressed using Dropout, Early Stopping, and Learning Rate Scheduling  
✅ Small Dataset → Improved performance with Transfer Learning and Data Augmentation  
✅ Complex Backgrounds → Enhanced generalization with multi-scale data augmentation  

---

## 📥 **Usage**
### 1. Clone the Repository:
```bash
git clone https://github.com/Rohitg9234/Mask-Detection.git
```

### 2. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 3. Train the Model:
```bash
python train.py
```

### 4. Predict:
```bash
python predict.py --image_path="path/to/image.jpg"
```

---

## 💡 **Future Scope**
- Real-time detection using OpenCV.  
- Deployment on edge devices (e.g., Raspberry Pi).  
- Multi-class detection for complex mask types.
- Still trying to improve 

---

## ⭐ **Why This Matters**
This project demonstrates deep technical expertise in:
- **Model architecture design**  
- **Data preprocessing and augmentation**  
- **Performance tuning**  
- **Transfer learning**  
- **Model evaluation and interpretation**  

✅ The high performance and scalability make it ideal for real-world deployment in surveillance and healthcare.  

---

Contributers are always welcome 😎
