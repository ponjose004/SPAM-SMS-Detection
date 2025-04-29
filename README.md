# SPAM-SMS-Detection
The Project aims to find the spam sms thorugh machine learning concepts. We have used Vetorization and model training using Multinomial Naive Bayes concept to solve this problem.

# 📱 SMS Spam Detection Project

Welcome to the **SMS Spam Detection** repository! 🚀 This project demonstrates a complete pipeline—from data preprocessing to model training and a user-friendly Streamlit app—designed to classify SMS messages as **Spam** or **Not Spam**.

---

## 📂 Repository Structure

```
├── DATA/
│   └── spam.csv              # Raw SMS dataset (labels: `ham` or `spam`)
├── Model files/
│   ├── spam_detector.pkl     # Trained Naive Bayes model
│   └── vectorizer.pkl        # Fitted TF-IDF vectorizer
├── Project1.ipynb            # Jupyter notebook: exploration + model training
└── streamlit_app.py          # Streamlit app for real-time prediction
```

---

## 📝 Dataset Overview

- **Source**: `DATA/spam.csv` (provided)  
- **Columns**:  
  - `label` — Message category (`ham` = not spam, `spam` = spam)  
  - `message` — Raw SMS text

We perform initial EDA to understand class balance and text distribution before any modeling. 🔍

---

## ⚙️ Model Pipeline

1. **Text Preprocessing** ✂️
   - Remove non-alphabetic characters using regex  
   - Convert to lowercase  
   - Tokenize, remove English stopwords, and apply Porter Stemming

2. **Feature Extraction** 🏷️
   - Use `TfidfVectorizer(max_features=3000)` to transform cleaned text into TF-IDF features

3. **Model Training** 🤖
   - Classifier: `MultinomialNB` (Multinomial Naive Bayes)  
   - Train/test split: 80/20  
   - Hyperparameter tuning: GridSearchCV over `alpha = [0.1, 0.5, 1.0]`

4. **Evaluation** 📊
   - Confusion matrix, precision, recall, F1-score, and overall accuracy

5. **Serialization** 💾
   - Save trained model (`spam_detector.pkl`) and TF-IDF vectorizer (`vectorizer.pkl`) with `pickle`

---

## 🤖 Model Architecture & Intuition

- **Multinomial Naive Bayes** is a probabilistic classifier based on Bayes’ theorem, ideal for word-count features.  
- **TF-IDF** helps down-weight common words (e.g., “the”, “and”) and up-weight rare but informative words (e.g., “free!”, “win”).

Together, they form a lightweight, interpretable, and fast pipeline perfect for real-time SMS classification. 🎯

---

## 🚀 Streamlit App (`streamlit_app.py`)

1. **UI Elements**
   - Title & description  
   - Text area for user input  
   - Predict button

2. **Prediction Flow**
   - Load `spam_detector.pkl` & `vectorizer.pkl`
   - Preprocess incoming text (same steps as training)  
   - Vectorize & predict probability of spam vs. ham  
   - Display result with a colored banner and confidence score

3. **Running Locally**:
   ```bash
   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```

---

## 🌐 Streamlit Webpage

Once deployed, your users can:

- Enter any SMS text in the box ✉️
- Click **Predict** to see instantly:
  - 🚨 **Spam Detected** (with confidence %) or
  - ✅ **Not Spam** (with confidence %)

Interactive and intuitive—no coding required! 🎉

---

## 💾 Model Files Download

- Download the TF-IDF vectorizer: [vectorizer.pkl](Model%20files/vectorizer.pkl)
- Download the Naive Bayes model: [spam_detector.pkl](Model%20files/spam_detector.pkl)

*(Replace `yourusername/yourrepo` with your GitHub path.)*

---

## 🔍 Further Improvements

- Experiment with **embeddings** (Word2Vec/GloVe) or **transformers** (BERT, DistilBERT)  
- Add **interpretability**: show top contributing words for each prediction  
- Deploy to **Heroku**, **AWS**, or **Streamlit Sharing** for public access

---

## 📖 Acknowledgements

- SMS Spam Collection Dataset  
- Scikit-learn & Streamlit communities

Happy coding! ✨

