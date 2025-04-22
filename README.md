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

![Dashboard](https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/title_image_mod3.png?raw=true)
## 📌 Overview

This project focuses on detecting spam messages from both **emails** and **SMS text messages** using machine learning and deep learning models. Two separate datasets are utilized to train and evaluate various models including traditional ML classifiers and CNN-based deep learning models.

## 📂 Datasets

The **[`SMS Spam Collection Dataset`](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)** is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam. The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.
The **[`190K Spam-Ham Email Dataset`](https://www.kaggle.com/datasets/meruvulikith/190k-spam-ham-email-dataset-for-classification)** contains over 190,000+ emails labeled as either spam or ham (non-spam). Each email is represented by its text content along with its corresponding label. The dataset provides a comprehensive collection of emails, categorized as either spam or ham, intended to facilitate research and development in email classification algorithms. With a vast corpus of emails, this dataset offers ample opportunities for training and evaluating machine learning models for effective spam detection. The data distribution of both datasets are shown in the charts below.
<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/small_dataset_viz.png?raw=true" alt="Small Dataset Visualization" width="49.5%" />
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/large_dataset_viz.png?raw=true" alt="Large Dataset Visualization" width="49.5%" />
</p>

## 📁 Folder Structure

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

## 🔄 Project Workflow
This project follows a clear and structured pipeline to build, train, and interpret models for detecting spam in SMS and email messages. The steps below outline the end-to-end process from raw data to model explainability and future improvements.
- Upload Datasets
- Clean, Preprocess, and Feature Selection
- Model Selection
- Train Models
- See Results
- Visualize Decisions with Tools
- Suggest Future Development


## 🚀 Models & Experiments

All test were done on google colab environment. Since `SMS Spam Collection Dataset` is smaller in size, all 8 machine learning algorithm was tried on it first. It was a bit time consuming training the models in cpu, so for the larger `190K Spam-Ham Email Dataset` was trained on A100 gpu. Even with gpu support, Random Forest, SVC and MLPClassifier were unsuitable for `190K Spam-Ham Email Dataset`. Both datasets were trained on gpu for customized CNN model. Links of all .ipynb files related to tests are given in the following chart. Since some outputs are truncated in github, original files from colab environment are also linked here.

| Notebook | Dataset | Model Type | GitHub Link | Colab Link |
|---------|--------|----------|------|--------|
| CNN_Spam_Detection_Dataset_214843.ipynb | 190K Spam-Ham Email Dataset | `Customized CNN` | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/src/CNN_Spam_Detection_Dataset_214843.ipynb) | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project/blob/main/CNN_Spam_Detection_Dataset_214843.ipynb) |
| CNN_Spam_Detection_Dataset_5572.ipynb | SMS Spam Collection Dataset | `Customized CNN` | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project/blob/main/CNN_Spam_Detection_Dataset_5572.ipynb) | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project/blob/main/CNN_Spam_Detection_Dataset_214843.ipynb) |
| ML_Spam_Detection_Dataset_214843.ipynb | 190K Spam-Ham Email Dataset | `Logistic Regression` `BernoulliNB` `MultinomialNB` `XGBoost` `AdaBoost` | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project/blob/main/ML_Spam_Detection_Dataset_214843.ipynb) | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project/blob/main/CNN_Spam_Detection_Dataset_214843.ipynb) |
| ML_Spam_Detection_Dataset_5572.ipynb | SMS Spam Collection Dataset |  `Logistic Regression` `SVC` `Random Forest` `BernoulliNB` `MultinomialNB` `XGBoost` `AdaBoost` `MLPClassifier` | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project/blob/main/ML_Spam_Detection_Dataset_5572.ipynb) | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Email-and-SMS-Spam-Detection-Project/blob/main/CNN_Spam_Detection_Dataset_214843.ipynb) |

## 📈 Results

Results from all tests are summarised in this section.

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/accuracy_5572.png?raw=true" alt="Accuracy on 5.5K Dataset" width="49.5%" />
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/accuracy_190K.png?raw=true" alt="Accuracy on 190K Dataset" width="49.5%" />
</p>

For `SMS Spam Collection Dataset`, most models (bar chart on left) showed more than 95% accuracy, with MLPClassifier performing the best (97.95% accuracy). AdaBoost performed a decent 93.84% accuracy. For `190K Spam-Ham Email Dataset`, Logistic Regression, XGBoost and MultinomialNB performed with more than 95% accuracy, Logistic Regression performing the best (97.83%). AdaBoost as usual performed the lowest (with 84.44% accuracy).

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/CNN_cm_accuracy_loss_5572.png?raw=true" alt="CNN Accuracy, Loss & Confusion Matrix (5.5K Dataset)" width="49.5%" />
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/CNN_acc_190K.png?raw=true" alt="CNN Accuracy (190K Dataset)" width="49.5%" />
</p>

The images above show results for customized CNN models (left bar chart for `SMS Spam Collection Dataset` and right for `190K Spam-Ham Email Dataset`). For both cases, CNN showed high classification accuracy (over 98%). However, for `190K Spam-Ham Email Dataset`, gradual increase of loss fuction over epochs were observed.

## 🧩 Confusion Matrices

Confusion matrix generated for all models are included in this section.

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/cm_5572.png?raw=true" alt="Confusion Matrix (5.5K Dataset)" width="52.82%" />
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/cm_190K.png?raw=true" alt="Confusion Matrix (190K Dataset)" width="45%" />
</p>

The image on the left shows all 8 machine learning model's confusion matrix for `SMS Spam Collection Dataset`. It is evident that all models struggled to identify spams (due to imbalanced dataset). The image on right shows 5 machine learning model's confusion matrix for `190K Spam-Ham Email Dataset`. Logistic regression, MultinomialNB and XGBoost showed excellent performance on classification due to large and balanced dataset.

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/cnn_cm_5572.png?raw=true" alt="CNN Confusion Matrix (5.5K Dataset)" width="48.29%" />
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/cnn_cm_190K.png?raw=true" alt="CNN Confusion Matrix (190K Dataset)" width="49.5%" />
</p>

Confusion Matrix for CNN model are shown in above 2 images. From the images it is evident that, size of dataset impacts deep learning model's classification capability.

## 🔥 LIME Visualizations

Numerous visualization were generated with Lime for CNN model. Two sample visualizations are included in this section.

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/lime_ham.png?raw=true" alt="LIME Ham Explanation" width="49.5%" />
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/lime_spam.png?raw=true" alt="LIME Spam Explanation" width="49.5%" />
</p>

The model predicts a sample message (visualization shown on left image) as **not spam** with 100% confidence because multiple key features strongly align with patterns seen in legitimate messages. For example, `feature_53 = 643.00` falls within a range the model associates with not_spam (between 89 and 1399), and `feature_81 = 1632.00` exceeds a high threshold (>1339), further reinforcing the not_spam label. Additionally, `feature_41 = 2391.00` and `feature_74 = 213.00` contribute positively toward not_spam. Although a few features like `feature_66 = 1156.00` and `feature_51 = 1.00` show some alignment with spam patterns, their influence is minimal compared to the stronger opposing signals. As a result, the cumulative contribution of the key features leads the model to confidently classify the message as not spam.

The model predicts another sample (visualization shown on right image), this time as **spam** with 100% confidence due to the strong influence of multiple features that match known spam patterns. Key features like `feature_67 = 118.00`, `feature_60 = 590.00`, `feature_91 = 130.00`, and `feature_83 = 135.00` fall into specific ranges that the model has learned are highly indicative of spam. Additionally, `feature_41 = 0.00` and `feature_53 = 0.00` contribute further support, signaling a lack of characteristics typical in legitimate messages. Although a couple of features—such as `feature_56 = 0.00` and `feature_87 = 11.00` suggest non-spam behavior, their impact is weak compared to the dominant spam-aligned features. This strong cumulative signal drives the model to confidently classify the message as spam.

## 🧠 Key Takeaway
- Implemented multiple machine learning models (Naive Bayes, Logistic Regression, Random Forest, SVM, and more) to detect spam in both email and SMS messages.
- Performed thorough text preprocessing including tokenization, stopword removal, and stemming to clean and prepare data.
- Evaluated models using accuracy and confusion matrix to ensure robust performance.
- Demonstrated the difference in spam characteristics between short-text (SMS) and long-text (email) datasets.
- Visualized dataset distribution and model performance for clearer insights and interpretability.

## Technology Used

## Future Development

## Licence

## Contact


