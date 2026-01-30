# ===========================
# AI Mock Interview: Improved SBERT + RF
# ===========================

import pandas as pd
import numpy as np
import re, os, pickle, random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from textstat import flesch_reading_ease
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import RandomOverSampler
import nltk, requests

# ---- Config ----
os.makedirs("models", exist_ok=True)
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
NUM_TFIDF_FEATURES = 200

# ---- Stopwords ----
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

# ---- Utilities ----
def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def is_link(text):
    if not text: return True
    return bool(re.search(r'(http[s]?://|www\.)', text))

# ---- Load & Clean Dataset ----
def load_and_clean_datasets():
    dfs = []

    # ML
    ml_df = pd.DataFrame(load_dataset("manasuma/ml_interview_qa")['train'])
    ml_df.rename(columns={'Questions':'Question','Answers':'Answer'}, inplace=True)
    ml_df['Domain'] = 'ML'
    dfs.append(ml_df)

    # AI
    ai_df = pd.DataFrame(load_dataset("K-areem/AI-Interview-Questions")['train'])
    def split_ai(text):
        match = re.match(r"<s>\[INST\](.*?)\[/INST\](.*?)</s>", text, re.DOTALL)
        if match: return pd.Series([match.group(1).strip(), match.group(2).strip()])
        return pd.Series([text, ""])
    ai_df[['Question','Answer']] = ai_df['text'].apply(split_ai)
    ai_df['Domain'] = 'AI'
    dfs.append(ai_df[['Question','Answer','Domain']])

    # HR
    hr_df = pd.DataFrame(load_dataset("Bhavya-123/HRMS_Datahub")['train'])
    hr_df.rename(columns={'question':'Question','answer':'Answer'}, inplace=True)
    hr_df['Domain'] = 'HR'
    dfs.append(hr_df)

    # Technical
    tech_df = pd.DataFrame(load_dataset("Aiman1234/Interview-questions")['train'])
    tech_df.rename(columns={'Questions':'Question','Answers':'Answer'}, inplace=True)
    tech_df['Domain'] = 'Technical'
    dfs.append(tech_df)

    # CS-Theory
    url = "https://huggingface.co/datasets/rohanrdy/CS-Theory-QA-Dataset/resolve/main/intents.json"
    data = requests.get(url).json()
    qa_list = []
    for item in data.get('intents', []):
        for q in item.get('patterns', []):
            a = item.get('responses', ["No answer"])[0]
            qa_list.append({'Question': q.strip(), 'Answer': a.strip(), 'Domain': 'CS-Theory'})
    cs_df = pd.DataFrame(qa_list)
    dfs.append(cs_df)

    # Combine all
    combined = pd.concat(dfs, ignore_index=True)
    combined['Question'] = combined['Question'].apply(clean_text)
    combined['Answer'] = combined['Answer'].apply(clean_text)
    combined = combined[(combined['Question'] != "") & (combined['Answer'] != "")]
    combined = combined[~combined['Answer'].apply(is_link)]
    combined.drop_duplicates(inplace=True)
    combined.reset_index(drop=True, inplace=True)

    print(f"✅ Total cleaned QAs: {len(combined)}")
    print(combined['Domain'].value_counts())
    return combined

# ---- Generate Synthetic Labels (balanced across difficulty/quality) ----
def generate_labels(df):
    # Simple heuristic: good length, readability, low stopwords => higher class
    q_len = df['Question'].apply(lambda x: max(1,len(x.split()))).values
    a_len = df['Answer'].apply(lambda x: max(1,len(x.split()))).values
    length_ratio = a_len / q_len

    stopword_ratio = df['Answer'].apply(lambda x: len([w for w in x.split() if w in STOPWORDS])/max(1,len(x.split()))).values
    readability = df['Answer'].apply(lambda x: flesch_reading_ease(x)).values

    score = np.zeros(len(df))
    score += (length_ratio > 0.5) * 1
    score += (readability > 50) * 1
    score += (stopword_ratio < 0.4) * 1
    score = score.astype(int)
    score = np.clip(score, 0, 3)
    return score

# ---- Feature Extraction ----
def extract_features(df, sbert_model, tfidf_vectorizer=None):
    # SBERT embeddings
    question_emb = sbert_model.encode(df['Question'].tolist(), convert_to_numpy=True)
    answer_emb = sbert_model.encode(df['Answer'].tolist(), convert_to_numpy=True)

    # TF-IDF embeddings
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(max_features=NUM_TFIDF_FEATURES)
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['Answer'])
    else:
        tfidf_matrix = tfidf_vectorizer.transform(df['Answer'])
    tfidf_array = tfidf_matrix.toarray()

    # Cosine similarity between Q & A
    cos_sim = np.array([cosine_similarity([q],[a])[0][0] for q,a in zip(question_emb, answer_emb)]).reshape(-1,1)

    # Length ratio
    q_len = df['Question'].apply(lambda x: max(1,len(x.split()))).values
    a_len = df['Answer'].apply(lambda x: max(1,len(x.split()))).values
    length_ratio = (a_len / q_len).clip(0,5).reshape(-1,1)

    # Stopword ratio
    stopword_ratio = df['Answer'].apply(lambda x: len([w for w in x.split() if w in STOPWORDS])/max(1,len(x.split()))).values.reshape(-1,1)

    # Readability
    readability = df['Answer'].apply(lambda x: flesch_reading_ease(x)).values.reshape(-1,1)

    # Combine all features
    X = np.hstack([question_emb, answer_emb, tfidf_array, cos_sim, length_ratio, stopword_ratio, readability])
    return X, tfidf_vectorizer

# ---- Train Improved Model ----
def train_model():
    df = load_and_clean_datasets()
    sbert_model = SentenceTransformer(EMBEDDER_MODEL)
    y = generate_labels(df)

    # Features
    X, tfidf_vectorizer = extract_features(df, sbert_model)

    # Balance dataset
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # Random Forest with class_weight
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    # Metrics
    y_pred = clf.predict(X_test)
    print("📊 Metrics on hold-out test set:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.2f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save models
    with open("models/rf_model.pkl","wb") as f: pickle.dump(clf,f)
    with open("models/sbert_model.pkl","wb") as f: pickle.dump(sbert_model,f)
    with open("models/tfidf_vectorizer.pkl","wb") as f: pickle.dump(tfidf_vectorizer,f)

    return clf, sbert_model, tfidf_vectorizer, df

# ---- Evaluate User Answer ----
def evaluate_answer(user_ans, ref_ans, clf, sbert_model, tfidf_vectorizer):
    q_emb = sbert_model.encode([ref_ans], convert_to_numpy=True)
    a_emb = sbert_model.encode([user_ans], convert_to_numpy=True)

    # TF-IDF
    tfidf_array = tfidf_vectorizer.transform([user_ans]).toarray()

    # Cosine similarity
    cos_sim = np.array([cosine_similarity([q_emb[0]],[a_emb[0]])[0][0]]).reshape(-1,1)

    # Length ratio
    q_len = max(1,len(ref_ans.split()))
    a_len = max(1,len(user_ans.split()))
    length_ratio = np.array([[a_len/q_len]])

    # Stopword ratio
    stopword_ratio = np.array([[len([w for w in user_ans.split() if w in STOPWORDS])/max(1,len(user_ans.split()))]])

    # Readability
    readability = np.array([[flesch_reading_ease(user_ans)]])

    X_user = np.hstack([q_emb, a_emb, tfidf_array, cos_sim, length_ratio, stopword_ratio, readability])
    pred_class = clf.predict(X_user)[0]

    feedback_map = {0:"Poor answer. Review the concept.",
                    1:"Average answer. Needs improvement.",
                    2:"Good answer. Minor improvements needed.",
                    3:"Excellent answer!"}
    return pred_class, feedback_map[pred_class]

# ---- Interactive Demo ----
def run_demo():
    clf, sbert_model, tfidf_vectorizer, df = train_model()
    domains = df['Domain'].unique().tolist()
    print("Available Domains:", domains)
    selected_domain = input("Select a domain: ").strip()
    domain_qas = df[df['Domain']==selected_domain].reset_index(drop=True)

    num_questions = min(5, len(domain_qas))
    questions = domain_qas.sample(num_questions, random_state=random.randint(0,1000))

    scores = []
    for idx, row in questions.iterrows():
        print(f"\nQuestion: {row['Question']}")
        user_ans = input("Your Answer: ").strip()
        pred_class, feedback = evaluate_answer(user_ans, row['Answer'], clf, sbert_model, tfidf_vectorizer)
        print(f"Predicted Quality: {pred_class} | Feedback: {feedback}")
        scores.append(pred_class)

    overall = np.mean(scores)
    print(f"\nOverall Score (Average Class): {overall:.2f}")

# ---- Main ----
if __name__ == "__main__":
    run_demo()
