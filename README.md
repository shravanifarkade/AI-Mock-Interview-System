# AI Mock Interview System – Answer Quality Evaluation 🤖🎤

An **AI & Machine Learning–based Mock Interview System** that evaluates the **quality of interview answers** using **Natural Language Processing (NLP)** and **Machine Learning models**.

The system compares a candidate’s answer with reference answers and predicts answer quality levels such as *Poor, Average, Good, or Excellent*, along with meaningful feedback.

---

## 📌 Project Motivation

Interview preparation platforms often lack **objective, automated evaluation** of candidate answers.  
This project addresses that gap by applying **semantic similarity, linguistic features, and ML classification** to simulate real interview feedback.

---

## 🎯 Key Objectives

- Simulate real interview answer evaluation
- Measure semantic similarity between answers
- Use ML to classify answer quality
- Provide instant, explainable feedback
- Support multiple interview domains

---

## 🧠 AIML Concepts & Techniques Used

### 🔹 Natural Language Processing (NLP)
- Text preprocessing & normalization
- Stopword analysis
- Readability analysis (Flesch Reading Ease)
- TF-IDF vectorization

### 🔹 Semantic Similarity
- **SBERT (Sentence-BERT)** embeddings
- Cosine similarity between question–answer pairs

### 🔹 Machine Learning
- Feature engineering using:
  - SBERT cosine similarity
  - TF-IDF similarity
  - Answer length ratio
  - Stopword ratio
  - Readability score
- **Logistic Regression** for multi-class classification
- Class imbalance handling using **class weights**

### 🔹 Datasets (Hugging Face)
- Machine Learning interview Q&A
- Artificial Intelligence interview Q&A
- HR interview questions
- Technical interview datasets
- CS Theory Q&A dataset

---

## 🛠️ Technology Stack

- **Language:** Python  
- **ML & NLP:**  
  - Sentence Transformers (SBERT)  
  - scikit-learn  
  - NLTK  
  - textstat  
- **Datasets:** Hugging Face Datasets API  
- **Model Persistence:** Pickle  

---

## 📂 Project Structure

