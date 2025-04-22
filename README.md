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

This project focuses on building a robust spam detection system using both machine learning and deep learning techniques, targeting two distinct types of data: SMS messages and emails. The goal is to classify incoming messages as either "spam" or "ham" (not spam) with high accuracy, precision, and efficiency. Two publicly available datasets were used:
- `SMS Spam Collection Dataset` – A smaller (but imbalanced) dataset ideal for initial testing and benchmarking of classic machine learning models.
- `190K Spam-Ham Email Dataset` – A much larger and more complex dataset, offering a real-world scenario to test scalability and deep learning capabilities.

The project explores a variety of machine learning algorithms including Multinomial and Bernoulli Naive Bayes, Logistic Regression, SVC, Random Forest, XGBoost, AdaBoost and MLPClassifier. It also implements a custom Convolutional Neural Network (CNN) model to extract deep semantic patterns from text, especially effective on longer email content. Performance was evaluated using metrics like accuracy, precision, recall, F1-score, and confusion matrices. Experiments were conducted on Google Colab, leveraging both CPU and A100 GPU environments depending on dataset size and model complexity. The final deliverables include a comparative analysis of model performance, detailed visualizations, and recommendations for real-world spam filtering solutions.

## 📂 Datasets
Both datasets are sourced from Kaggle. The **[`SMS Spam Collection Dataset`](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)** is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam. The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.
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
│     |
|     ├──   CNN_Spam_Detection_Dataset_214843.ipynb
|     ├──   CNN_Spam_Detection_Dataset_5572.ipynb
|     ├──   ML_Spam_Detection_Dataset_214843.ipynb
|     ├──   ML_Spam_Detection_Dataset_5572.ipynb
|
├── Images/                 # Dataset, Accuracy and Confusion Matrix visualizations
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

All experiments were conducted in the Google Colab environment. Due to its smaller size, the `SMS Spam Collection Dataset` was used to test all eight machine learning algorithms initially. However, training these models on CPU proved to be time-consuming. For the larger `190K Spam-Ham Email Dataset`, training was performed on an A100 GPU to improve efficiency. Despite the GPU support, models such as Random Forest, SVC, and MLPClassifier were not well-suited for the larger dataset due to their extended training time and resource demands. Both datasets were also used to train a custom CNN model, leveraging GPU acceleration for optimal performance. A table containing links to all related .ipynb files is provided below. Since GitHub may truncate some output cells, direct links to the original Colab notebooks are also included for complete access.

| Notebook | Dataset | Model Type | GitHub Link | Colab Link |
|---------|--------|----------|------|--------|
| CNN_Spam_Detection_Dataset_214843.ipynb | 190K Spam-Ham Email Dataset | `Customized CNN` | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/src/CNN_Spam_Detection_Dataset_214843.ipynb) | [📘 View Notebook](https://colab.research.google.com/drive/1R3nuBxmaA_zGfYEfsCsyFqyTuH21knzw?usp=sharing) |
| CNN_Spam_Detection_Dataset_5572.ipynb | SMS Spam Collection Dataset | `Customized CNN` | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/src/CNN_Spam_Detection_Dataset_5572.ipynb) | [📘 View Notebook](https://colab.research.google.com/drive/1U1BebGprQ4GQGuz_Uf_g99ZnHhIfr4u7?usp=sharing) |
| ML_Spam_Detection_Dataset_214843.ipynb | 190K Spam-Ham Email Dataset | `Logistic Regression` `BernoulliNB` `MultinomialNB` `XGBoost` `AdaBoost` | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/src/ML_Spam_Detection_Dataset_214843.ipynb) | [📘 View Notebook](https://colab.research.google.com/drive/1hMmo4PS1nbeDGlD0Ac61tIfxVuF4YCX7?usp=sharing) |
| ML_Spam_Detection_Dataset_5572.ipynb | SMS Spam Collection Dataset |  `Logistic Regression` `SVC` `Random Forest` `BernoulliNB` `MultinomialNB` `XGBoost` `AdaBoost` `MLPClassifier` | [📘 View Notebook](https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/src/ML_Spam_Detection_Dataset_5572.ipynb) | [📘 View Notebook](https://colab.research.google.com/drive/147cbB8XTxfXggy6osZYx8jRK_jvxMJeQ?usp=sharing) |

## 📈 Results

Results from all tests are summarised in this section.

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/accuracy_5572.png?raw=true" alt="Accuracy on 5.5K Dataset" width="49.5%" />
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/accuracy_190K.png?raw=true" alt="Accuracy on 190K Dataset" width="49.5%" />
</p>

The evaluation began with testing eight machine learning models on the SMS Spam Collection Dataset, a relatively small and balanced dataset. The first accuracy chart shows that most ML models performed impressively well, with Multinomial Naive Bayes, Logistic Regression, and Linear SVC achieving the highest accuracy. These models are particularly well-suited to short, text-based classification problems and require minimal computation, making them ideal for lightweight applications like mobile spam filters. Transitioning to the 190K Spam-Ham Email Dataset, the second accuracy chart highlights a significant shift in performance dynamics. While simpler models like Naive Bayes still held up reasonably well, models such as Random Forest, SVC, and MLPClassifier struggled—not only in accuracy but also in terms of training time and scalability. These models became computationally inefficient, with diminishing returns in predictive performance. Logistic Regression and Linear SVC remained reliable options but began to show signs of strain with longer, more context-heavy email content.

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/CNN_cm_accuracy_loss_5572.png?raw=true" alt="CNN Accuracy, Loss & Confusion Matrix (5.5K Dataset)" width="49.5%" />
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/CNN_acc_190K.png?raw=true" alt="CNN Accuracy (190K Dataset)" width="49.5%" />
</p>

To address this, a custom CNN model was implemented and trained on both datasets. The third image, showing CNN performance on the SMS dataset, revealed that while CNN achieved competitive accuracy, it did not dramatically outperform traditional ML models—likely because the short SMS texts lacked the complex patterns CNNs are designed to detect. However, its performance was still consistent, confirming its robustness. The true strength of the CNN model became clear in the fourth image, which presents results on the larger email dataset. Here, CNN outperformed all traditional ML models, achieving noticeably higher accuracy. The network’s ability to extract hierarchical features from longer text inputs helped it better capture the contextual patterns often present in email spam. This was further reflected in balanced precision and recall metrics, along with reduced false positives and false negatives, making the CNN model particularly suitable for real-world deployment in spam email filtering systems.

In summary, ML models are fast and efficient for short text (SMS) classification, but they do not scale well to large, complex datasets. In contrast, the CNN model, while heavier, excels at identifying nuanced spam signals in lengthy emails, offering a powerful and scalable solution for high-volume spam detection systems.

## 🧩 Confusion Matrices

The confusion matrices further validate the performance trends observed in the accuracy charts.

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/cm_5572.png?raw=true" alt="Confusion Matrix (5.5K Dataset)" width="52.82%" />
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/cm_190K.png?raw=true" alt="Confusion Matrix (190K Dataset)" width="45%" />
</p>

In the first confusion matrix (ML models on the SMS Spam Collection Dataset), the top-performing models like Multinomial Naive Bayes and Logistic Regression demonstrated strong separation between spam and ham messages. Most predictions fell along the diagonal, indicating high true positive and true negative rates. False positives (ham classified as spam) were minimal, which is crucial to avoid user frustration from legitimate messages being blocked. The second matrix (ML models on the 190K Spam-Ham Email Dataset) revealed a different picture. Although some models still achieved decent overall accuracy, they began to struggle with class balance and longer message complexity. This was reflected in an increase in both false positives and false negatives. Specifically, models like Random Forest and MLPClassifier showed a noticeable drop in precision, misclassifying a higher number of legitimate emails as spam.

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/cnn_cm_5572.png?raw=true" alt="CNN Confusion Matrix (5.5K Dataset)" width="48.29%" />
  <img src="https://github.com/ShaikhBorhanUddin/Spam-Detection-With-Machine-Learning/blob/main/images/cnn_cm_190K.png?raw=true" alt="CNN Confusion Matrix (190K Dataset)" width="49.5%" />
</p>

The third confusion matrix (CNN model on the SMS dataset) showed solid performance, similar to ML models, but with a slightly more conservative approach—fewer false positives, but occasionally missing some spam (false negatives). This suggests the CNN model applied stricter criteria in identifying spam in short texts, favoring precision over recall. However, the fourth matrix (CNN model on the email dataset) highlighted the model’s true advantage. The CNN achieved excellent separation between spam and ham, with a well-balanced distribution across all four quadrants. False positives and false negatives were significantly reduced compared to the traditional ML models, confirming the CNN’s effectiveness at capturing complex patterns in long-form messages. This balance between precision and recall is critical in spam detection, where both types of misclassification can carry real-world consequences.

In conclusion, while ML models perform reliably on smaller, simpler datasets, CNNs demonstrate superior performance in large-scale, real-world spam detection scenarios, as clearly supported by the confusion matrix visualizations.
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

## 🛠️ Technology Used

This project leverages a combination of machine learning and deep learning frameworks, along with essential data processing and visualization libraries, to build an effective spam detection pipeline for both emails and SMS messages.
- Programming Language: `Python 3.10+`
- Machine Learning Frameworks: `Naive Bayes` `Logistic Regression` `Random Forest` `SVM` `XGBoost`
- Deep Learning Frameworks: `TensorFlow` `Keras` `NumPy` `Pandas`
- NLTK: `tokenization` `stopword removal` `stemming`
- Feature Extraction: `TF-IDF Vectorizer` `CountVectorizer`
- Model Evaluation: `Accuracy` `Precision` `Recall` `F1-score` `Confusion Matrix`
- Visualization: `Matplotlib` `Seaborn` `Lime`
- Development Environment: `Jupyter Notebook` `Google Colab`

## 🚧 Future Development

This project establishes a solid foundation for spam detection using both machine learning and deep learning techniques. There are several areas for future improvement. First, model optimization can be explored by fine-tuning hyperparameters and experimenting with ensemble methods to further enhance the accuracy and robustness of the model. Additionally, advanced deep learning architectures such as LSTM, BiLSTM, or Transformer-based models like BERT could be incorporated to better capture the contextual information in longer text messages, improving detection performance. Another potential area for development is the creation of a real-time spam filtering system, which could be implemented as an application or API using frameworks like FastAPI or Flask. Expanding the system to support multilingual spam detection would make it more versatile and applicable in diverse global contexts. Further efforts could also focus on improving the explainability and interpretability of the models by integrating tools like SHAP, offering better insights into how predictions are made and increasing transparency. To ensure the model remains effective in the face of evolving spam tactics, it would be beneficial to expand the dataset with more diverse and recent examples. Finally, deploying the model as a web or mobile application would allow for easier access and real-time use by end-users.

## 📄 Licence

This project is licensed under the MIT License — a permissive open-source license that allows reuse, modification, and distribution with attribution. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the project, provided that the original copyright and license notice are included in all copies or substantial portions of the software.

For more details, refer to the LICENSE file in this repository.

## 🤝 Contact

If you have any questions or would like to connect, feel free to reach out!

**Shaikh Borhan Uddin**  
📧 Email: [`shaikhborhanuddin@gmail.com`](mailto:shaikhborhanuddin@gmail.com)  
🔗 [`LinkedIn`](https://www.linkedin.com/in/shaikh-borhan-uddin-905566253/)  
🌐 [`Portfolio`](https://github.com/ShaikhBorhanUddin)

Feel free to fork the repository, improve the queries, or add visualizations!


