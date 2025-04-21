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

![Dashboard](https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/title_image_mod2.png?raw=true)
## 📌 Overview

This project focuses on detecting spam messages from both **emails** and **SMS text messages** using machine learning and deep learning models. Two separate datasets are utilized to train and evaluate various models including traditional ML classifiers and CNN-based deep learning models.

## 📂 Datasets

The **[`SMS Spam Collection Dataset`](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)** is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam. The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.
The **[`190K Spam-Ham Email Dataset`](https://www.kaggle.com/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification)** contains over 190,000+ emails labeled as either spam or ham (non-spam). Each email is represented by its text content along with its corresponding label. The dataset provides a comprehensive collection of emails, categorized as either spam or ham, intended to facilitate research and development in email classification algorithms. With a vast corpus of emails, this dataset offers ample opportunities for training and evaluating machine learning models for effective spam detection. The data distribution of both datasets are shown in the charts below.
<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/small_dataset_viz.png?raw=true" alt="Small Dataset Visualization" width="49.5%" />
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/large_dataset_viz.png?raw=true" alt="Large Dataset Visualization" width="49.5%" />
</p>

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

## 🚀 Models & Experiments

All test were done on google colab environment. Since `SMS Spam Collection Dataset` is smaller in size, all 8 machine learning algorithm was tried on it first. It was a bit time consuming training the models in cpu, so for the larger `190K Spam-Ham Email Dataset` was trained on A100 gpu. Even with gpu support, Random Forest, SVC and MLPClassifier were unsuitable for `190K Spam-Ham Email Dataset`. Both datasets were trained on gpu for customized CNN model. Links of all .ipynb files related to tests are given in the following chart. Since some outputs are truncated in github, original files from colab environment are also linked here.

| Notebook | Dataset | Model Type | GitHub Link | Colab Link |
|---------|--------|----------|------|--------|
| CNN Spam Detection | 190K Spam-Ham Email Dataset | `Customized CNN` | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project/blob/main/CNN_Spam_Detection_Dataset_214843.ipynb) |
| CNN Spam Detection | SMS Spam Collection Dataset | `Customized CNN` | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project/blob/main/CNN_Spam_Detection_Dataset_5572.ipynb) |
| ML Spam Detection | 190K Spam-Ham Email Dataset | `Logistic Regression` `BernoulliNB` `MultinomialNB` `XGBoost` `AdaBoost` | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project/blob/main/ML_Spam_Detection_Dataset_214843.ipynb) |
| ML Spam Detection | SMS Spam Collection Dataset |  `Logistic Regression` `SVC` `Random Forest` `BernoulliNB` `MultinomialNB` `XGBoost` `AdaBoost` `MLPClassifier` | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project/blob/main/ML_Spam_Detection_Dataset_5572.ipynb) |

## 🔄 Project Workflow

## 📈 Results

## 🧩 Confusion Matrices

## 🔥 LIME Visualizations

## 🧠 Key Takeaway
- Text preprocessing with NLTK and Scikit-learn
- CNN architecture for text classification
- Traditional ML classifiers: Logistic Regression, Naive Bayes, SVM
- Performance metrics: Accuracy, Precision, Recall, F1-Score
- Visualizations via Tableau

## Technology Used

## Future Development

## Licence

## Contact


