# Twitter Sentiment Sentiment-Analysis-for-Brand-Management using Deep Learning

This project implements a Bidirectional LSTM (Long Short-Term Memory) neural network to perform sentiment analysis on the Sentiment140 dataset, which contains 1.6 million tweets. The model is trained to classify tweets as either positive or negative, achieving approximately 83% accuracy on the test set.

This repository demonstrates a complete end-to-end data science workflow, from data exploration and preprocessing to model training, evaluation, and interpretation.

![Sentiment Analysis Banner](https://user-images.githubusercontent.com/26529339/194831039-40899663-8f03-4963-9562-def7418c3531.png)
*(Feel free to replace this with a screenshot of one of your project's visualizations, like the classification report or confusion matrix!)*

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Results](#results)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

---

## Project Overview

The goal of this project is to build a robust model capable of understanding and classifying the sentiment of a tweet as either positive or negative. Social media sentiment analysis is a valuable tool for businesses to gauge public opinion, monitor brand reputation, and understand customer feedback in real-time. This project tackles the challenge of processing noisy, informal text data from Twitter to extract meaningful sentiment.

---

## Dataset

The model is trained on the **Sentiment140 dataset**.
- **Source:** [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Content:** It contains 1.6 million tweets extracted using the Twitter API.
- **Labels:** The tweets are pre-labeled with `0` for negative sentiment and `4` for positive sentiment.
- **Balance:** The dataset is perfectly balanced, with 800,000 negative tweets and 800,000 positive tweets.

---

## Key Features

- **Exploratory Data Analysis (EDA):** A comprehensive EDA was performed to understand the dataset's characteristics, including:
    - Distribution of tweet lengths and sentiment balance.
    - Word clouds and frequency analysis to identify key terms for each sentiment.
    - Analysis of hashtags and user mentions.
    - Time-series analysis of tweet volume.

- **Text Preprocessing Pipeline:** A robust pipeline was created to clean the noisy text data, which included:
    - Lowercasing text.
    - Removing URLs, user mentions, hashtags, and special characters.
    - Tokenization and sequence padding for model input.

- **Deep Learning Model:** A Bidirectional LSTM model was built using TensorFlow/Keras. This architecture was chosen for its ability to capture contextual information from text sequences by processing them in both forward and backward directions.

- **Model Evaluation:** The model was rigorously evaluated on an unseen test set, with performance measured using:
    - Accuracy
    - Precision, Recall, and F1-Score
    - Confusion Matrix

---

## Methodology

The project follows a structured machine learning pipeline:

1.  **Data Loading & Initial Exploration:** The dataset was loaded, and an initial assessment was performed.
2.  **Exploratory Data Analysis (EDA):** Various visualizations were created to generate insights and inform preprocessing decisions.
3.  **Data Cleaning & Preprocessing:** The tweets were cleaned and transformed into a suitable format for the model.
4.  **Train/Validation/Test Split:** The data was split into 80% for training, 10% for validation, and 10% for testing, using stratification to maintain sentiment balance.
5.  **Tokenization:** The text was converted into integer sequences using Keras `Tokenizer`, with a vocabulary size of 20,000 words.
6.  **Padding:** All sequences were padded to a uniform length of 60 to ensure consistent input size for the model.
7.  **Model Building:** A `Sequential` model was defined with the following layers:
    - `Embedding(input_dim=20000, output_dim=128)`
    - `Bidirectional(LSTM(64))`
    - `Dense(64, activation='relu')`
    - `Dropout(0.3)`
    - `Dense(1, activation='sigmoid')`
8.  **Training & Monitoring:** The model was trained using the Adam optimizer and `binary_crossentropy` loss. `EarlyStopping` was used to monitor validation accuracy and prevent overfitting.
9.  **Evaluation:** The final model was evaluated on the held-out test set.

---

## Results

The model achieved excellent performance on the unseen test data:

- **Test Accuracy:** **~83%**
- **Precision (Negative/Positive):** 0.83 / 0.83
- **Recall (Negative/Positive):** 0.83 / 0.83
- **F1-Score (Negative/Positive):** 0.83 / 0.83

The close alignment between validation and test accuracy indicates that the model generalizes well to new data. The confusion matrix and classification report confirm that the model is equally effective at identifying both positive and negative sentiments.

![Confusion Matrix](link_to_your_confusion_matrix_image.png)
*(Instructions: Take a screenshot of your confusion matrix plot, upload it to your GitHub repo, and replace 'link_to_your_confusion_matrix_image.png' with the direct link to the image.)*

---

## How to Run

You can run this project in a cloud environment like Kaggle or Google Colab, or on a local machine.

**1. Using Kaggle:**
   - Upload this repository's notebook file (`.ipynb`) to Kaggle.
   - Add the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) to the notebook's input.
   - Run the cells sequentially.

**2. On a Local Machine:**
   - Clone the repository:
     ```bash
     git clone https://github.com/your-username/your-repo-name.git
     cd your-repo-name
     ```
   - Install the required libraries:
     ```bash
     pip install -r requirements.txt
     ```
     *(Note: You will need to create a `requirements.txt` file listing the packages below.)*
   - Download the dataset and place it in the appropriate directory.
   - Run the Jupyter notebook.

---

## Technologies Used

- **Python 3**
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical operations.
- **Scikit-learn:** For data splitting and evaluation metrics.
- **TensorFlow / Keras:** For building and training the deep learning model.
- **NLTK:** For natural language processing tasks (optional, if stopwords are used).
- **Matplotlib & Seaborn:** For data visualization.
- **WordCloud:** For generating word cloud visualizations.
- **Jupyter Notebook / Kaggle:** For development and presentation.

---

## Future Improvements

While the model performs well, several avenues for future improvement exist:
- **Use Pre-trained Embeddings:** Incorporate pre-trained word embeddings like GloVe or FastText, which are trained on vast text corpora and can improve model performance by providing richer semantic representations.
- **Experiment with Other Architectures:** Test more complex architectures, such as stacking multiple LSTM/GRU layers or using Transformer-based models like BERT.
- **Advanced Hyperparameter Tuning:** Use automated tuning libraries like KerasTuner or Optuna to systematically search for the optimal set of hyperparameters.
- **Lemmatization:** Add a lemmatization step to the preprocessing pipeline to reduce words to their root form, potentially reducing vocabulary size and improving generalization.
