<h1 align="left">📧 Spam Detection with Machine Learning</h1>

<p align="left">
  <img src="https://img.shields.io/badge/Made%20With-Colab-blue?logo=googlecolab&logoColor=white&label=Made%20With" alt="Made with Colab">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/github/repo-size/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project" alt="Repo Size">
  <img src="https://img.shields.io/github/last-commit/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project" alt="Last Commit">
  <img src="https://img.shields.io/github/issues/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project" alt="Issues">
  <img src="https://img.shields.io/badge/Data%20Visualization-Python-yellow?logo=python" alt="Data Visualization: Python">
  <img src="https://img.shields.io/badge/Version%20Control-Git-orange?logo=git" alt="Version Control: Git">
  <img src="https://img.shields.io/badge/Host-GitHub-black?logo=github" alt="Host: GitHub">
  <img src="https://img.shields.io/github/forks/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project?style=social" alt="Forks">
  <img src="https://img.shields.io/badge/Project-Completed-brightgreen" alt="Project Status">
</p>

---

## 📌 Overview

This project focuses on detecting spam messages from both **emails** and **SMS text messages** using machine learning and deep learning models. Two separate datasets are utilized to train and evaluate various models including traditional ML classifiers and CNN-based deep learning models.

---

## 📂 Datasets Used

- 📥 [SMS Spam Collection Dataset (5,572 records)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- 📥 [190K Spam-Ham Email Dataset](https://www.kaggle.com/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification)

---

## 🚀 Models & Experiments

| Notebook | Dataset | Model Type | GitHub Link |
|---------|---------|------------|------|
| CNN Spam Detection | 214,843 emails | Deep Learning (CNN) | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project/blob/main/CNN_Spam_Detection_Dataset_214843.ipynb) |
| CNN Spam Detection | 5,572 SMS | Deep Learning (CNN) | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project/blob/main/CNN_Spam_Detection_Dataset_5572.ipynb) |
| ML Spam Detection | 214,843 emails | Logistic Regression, BernoulliNB, MultinomialNB, XGBoost, and AdaBoost | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project/blob/main/ML_Spam_Detection_Dataset_214843.ipynb) |
| ML Spam Detection | 5,572 SMS |  Logistic Regression, SVC, Random Forest, BernoulliNB, MultinomialNB, XGBoost, and AdaBoost, and MLPClassifier | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project/blob/main/ML_Spam_Detection_Dataset_5572.ipynb) |

---

## 📁 Project Structure

```bash
Email-and-SMS-Spam-Detection-Project
│
├── Dataset/                # Contains raw CSV files
├── src/                    # Model training, preprocessing scripts
├── Images/                 # Tableau visualizations
│
├── requirements.txt        # Python dependencies
├── Licence                 # MIT License
└── README.md               # Overview of the project
```
---
## 🧠 Key Features
- Text preprocessing with NLTK and Scikit-learn
- CNN architecture for text classification
- Traditional ML classifiers: Logistic Regression, Naive Bayes, SVM
- Performance metrics: Accuracy, Precision, Recall, F1-Score
- Visualizations via Tableau

