# NLP Pipeline for Customer Review Analysis (Yelp Dataset)

## Abstract

This document provides a technical overview of the NLP pipeline developed for customer review analysis using the Yelp Dataset. The project combines Supervised Learning (Hybrid MLP Classification) and Unsupervised Learning (Topic Modeling) to extract semantic insights. The workflow encompasses advanced pre-processing (Bigrams, VADER), multi-modal vectorization (TF-IDF + BERT), and a comparative analysis between statistical (LSA) and neural (BERTopic) modeling techniques.

## Dataset

The data used for this project is the official Yelp Open Dataset. You can find and download the original dataset here:
[Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/).
Once downloaded, you can use this dataset to run the notebooks sequentially.

## Pipeline Description

The technical implementation is executed through five sequential Jupyter Notebooks.

### 1. Data Loading

Handles the ingestion of the massive Yelp Dataset (JSON inside TAR archive).

* **Stream Processing:** Reads the 'tar' file line-by-line to avoid RAM saturation.
* **Filtering:** Selects businesses in the "Restaurants" category with > 500 reviews.
* **Output:** Merges Business and Review data into `df_final.csv`.

### 2. Pre-Processing

Handles advanced text cleaning and feature engineering.

* **NLP Cleaning:** Lemmatization, stopword removal, and lowercasing.
* **Bigram Detection:** Uses `gensim.models.Phrases` to create `topic_text_clean` (e.g., joins "credit" + "card" → credit_card).
* **Feature Engineering:** Calculates structural features (Caps Ratio, Punctuation count) and Sentiment Scores (VADER).
* **Split:** Performs an 80/20 Stratified Split based on Star Rating.

### 3. Vector Representation

Generates three distinct types of vector representations:

* **TF-IDF (Classification):** Max 5000 features, Bigrams allowed.
* **TF-IDF (Topic Modeling):** Max 10000 features, Unigrams on `topic_text_clean` (where bigrams are already joined).
* **BERT Embeddings:** Uses `all-MiniLM-L6-v2` (Sentence Transformer) to create dense contextual vectors (384d), saved as npy for memory mapping.

### 4. Text Classification

Implements a comparative study of supervised models, progressing from statistical baselines to hybrid neural architectures.

**Experimental Approaches:**

* **Model 1 (Statistical Baseline):** Logistic Regression trained on TF-IDF vectors + Extra Features.
* **Model 2 (Neural Contextual):** MLPClassifier trained on BERT embeddings + Extra Features.
* **Model 3 (Hybrid Architecture - Final):** MLPClassifier trained on the concatenation of BERT (Context), TF-IDF (Keywords), and Extra Features.

**Tasks Methodology:**

* **Task A (Binary):** Classifies Negative (1-2) vs. Positive (4-5) sentiment.
* **Task B (Ordinal):** Decomposes the 5-star problem into 4 cumulative binary classifiers ().
* **Optimization:** Uses 10-fold Cross-Validation on OOF (Out-Of-Fold) probabilities to tune classification thresholds and minimize MAE (Mean Absolute Error).

### 5. Topic Modeling

Unsupervised analysis of latent themes using Stratified Sampling.

* **Models:** Comparison between LSA (Statistical) and BERTopic (Semantic).
* **Metric Alignment:** Coherence () is strictly calculated on `topic_text_clean` to ensure fairness.
* **Visualization Artifacts:** * **HTML:** Interactive Treemap (Binary) and Sunburst (Multiclass) charts.
* **PNG:** Static WordClouds and Bar Charts for report inclusion.


## System Requirements

To reproduce the analysis, the following libraries are required:

```bash
pip install bertopic gensim plotly sentence-transformers wordcloud

```
