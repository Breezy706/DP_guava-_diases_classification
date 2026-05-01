# 🍈 Guava Disease Detection using Deep Learning

**DATA SCIENTIST:** BREEZY_360

---

## 📌 Project Overview

Guava (*Psidium guajava*) is an important fruit crop, especially in South Asia, where it contributes significantly to nutrition and the economy. However, guava production is highly affected by diseases that reduce yield and quality.

This project applies **Deep Learning (CNN & Transfer Learning)** techniques to detect diseases in guava fruits at an early stage. The goal is to help farmers and stakeholders protect harvests and reduce economic losses.

---

## 🎯 Objective

* Detect guava fruit diseases automatically using image data
* Build a robust deep learning model for classification
* Compare performance between a custom CNN model and a pre-trained model (VGG16)
* Achieve high accuracy in real-world disease detection

---

## 📊 Dataset

The dataset consists of labeled images of guava fruits categorized into:

* 🍂 **Anthracnose**
* 🐛 **Fruit Fly**
* 🍏 **Healthy Guava**

📎 **Dataset Link:** [https://drive.google.com/drive/folders/1_4wEGc0IlYUeh4V8Tug0kJcKzhg1E0eU?usp=drive_link]

---

## 🧠 Models Used

### 1️⃣ Custom CNN Model

* Multiple Conv2D + MaxPooling layers
* Batch Normalization
* Fully connected dense layers
* Softmax output layer

### 2️⃣ Transfer Learning (VGG16)

* Pre-trained **VGG16** model (ImageNet weights)
* Frozen base layers
* Added custom classification head

---

## ⚙️ Technologies Used

* Python 🐍
* TensorFlow / Keras
* NumPy & Pandas
* Matplotlib & Seaborn
* Scikit-learn

---


## 📈 Model Performance

### ✅ Custom CNN Results

* Training Accuracy: ~99.96%
* Validation Accuracy: ~95.50%

### ✅ VGG16 (Transfer Learning) Results

* Validation Accuracy: ~98.41%
* Test Accuracy: ~98.17%

---

## 📊 Classification Report

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Anthracnose   | 0.98      | 1.00   | 0.99     |
| Fruit Fly     | 0.98      | 0.98   | 0.98     |
| Healthy Guava | 0.98      | 0.97   | 0.98     |

**Overall Accuracy:** 98%

---

## 🖼️ Sample Prediction

The model can:

* Predict disease from a new image
* Display prediction result
* Show probability distribution graph

📎 **Demo / Output Images:** 
<img width="861" height="385" alt="image" src="https://github.com/user-attachments/assets/d31bd49d-91da-489b-815f-76dc0e551691" />


---

## 📂 Project Structure

```
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
├── notebooks/
├── images/
├── README.md
└── requirements.txt
```

---

## 💡 Key Features

* Image visualization for dataset inspection
* CNN-based classification
* Transfer learning with VGG16
* Performance evaluation (accuracy, loss, classification report)
* Prediction with probability graph

---

## 🔮 Future Improvements

* Deploy as a web or mobile application
* Add more disease classes
* Use real-world dataset instead of local paths
* Implement real-time detection using camera

---

## 🌍 Applications

* Smart agriculture systems
* Farmer support tools
* Crop monitoring solutions
* AI-based disease diagnosis

---

## 👨‍💻 Author

**BREEZY_360**

📎 GitHub: [https://github.com/Breezy706]

---

## 📜 License

This project is open-source and available for educational and research purposes.

---

⭐ *If you find this project useful, consider giving it a star!*
