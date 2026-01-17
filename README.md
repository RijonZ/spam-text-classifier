# Spam Text Classifier
A machine learning project that classifies text messages/emails as **spam** or **ham (legitimate)** using multiple supervised learning algorithms. The project focuses on text preprocessing, feature extraction with bag-of-words, and comparative model evaluation.

# Dataset
  - Dataset source: (https://www.kaggle.com/datasets/zeeshanyounas001/email-spam-detection?select=spam+mail.csv) The project uses the spam mail.csv dataset, which contains labeled text messages with the following structure:
  - Category: Label indicating whether the message is spam or ham
  - Message: The actual message content

## Dataset Statistics
   - Duplicate messages are identified and removed
   - Missing values are checked and handled
   - Class distribution is analyzed to understand imbalance between spam and ham messages

# Features
- **Data Preprocessing**
  - Lowercasing text
  - Punctuation removal
  - Basic text normalization
- **Multiple Algorithms**
  - Multinomial Naive Bayes
  - Logistic Regression
  - Linear Support Vector Classifier (LinearSVC)
- **Text Vectorization**
  - CountVectorizer (Bag-of-Words)
- **Model Evaluation**
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Classification report
- **Visualization**
  - Confusion matrix using Seaborn heatmaps

# Installation

## Prerequisites
Make sure you have Python 3.7+ installed.

## Required Libraries
Install all required dependencies using:
```markdown
pip install numpy pandas matplotlib seaborn scikit-learn
```
# Model Performance

Multiple classifiers are trained and evaluated using the same train-test split.  
The best-performing model is automatically selected based on evaluation metrics, primarily **accuracy**.

In this project, **Linear Support Vector Machine (Linear SVM)** achieved the best overall performance.  
Linear SVM is highly effective for high-dimensional text data, as it finds an optimal separating hyperplane between spam and ham messages using word-frequency features.

While **Multinomial Naive Bayes** and **Logistic Regression** also performed well and provided strong baseline comparisons, Linear SVM demonstrated superior classification accuracy and more robust decision boundaries for this dataset.

# Usage

## Running the Notebook
   1. Clone this repository
   2. Ensure spam mail.csv is in the same directory as the notebook
   3. Open and run TextSpam.ipynb (or equivalent) in Jupyter Notebook / JupyterLab

# Predicting New Messages

A prediction cell is included in the notebook to classify new text messages.
You only need to modify the new_message list and run the cell.
```markdown
new_message = ["Win money now! Click here to claim your prize"]

new_vector = vectorizer.transform(new_message)
encoder.inverse_transform(best_model.predict(new_vector))
```
**Output example:**
```markdown
['spam']
```
This simple approach is intentional, focusing on demonstrating core model functionality without building a complex interface.

# Project Structure
```markdown
spam-mail-classifier/
├── spam mail.csv        # Dataset file
├── TextSpam.ipynb       # Main Jupyter notebook
└── README.md            # Project documentation
```
# Methodology
**1. Data Preprocessing**
    - Convert text to lowercase
    - Remove punctuation
    - Check for duplicates and missing values
    - Encode labels using LabelEncoder
**2. Feature Engineering**
   - Convert text data into numerical format using CountVectorizer
   - Transform categorical labels into numerical values
**3. Model Training & Evaluation**
   - Train-test split using scikit-learn
   - Train multiple classifiers:
       - Multinomial Naive Bayes
       - Logistic Regression
       - Linear SVM
   - Compare models using:
       - Accuracy
       - Precision
       - Recall
       - F1-score
   - Generate confusion matrix for the best-performing model
**4. Model Analysis**
   - Confusion matrix visualization
   - Comparative performance analysis across models

# Key Insights
   - Spam messages tend to have distinct word patterns compared to ham messages
   - Multinomial Naive Bayes is highly effective for bag-of-words text classification
   - Linear models (Logistic Regression & SVM) offer strong baselines for comparison
   - Confusion matrices provide clear insight into false positives and false negatives

# Contributing
Contributions are welcome. Possible improvements include:
   - Advanced text preprocessing (TF-IDF, n-grams)
   - Hyperparameter tuning
   - Additional classifiers (Random Forest, XGBoost)
   - Class imbalance handling
   - Deployment as a web or API-based service
