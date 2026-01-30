# ===========================
# AI Mock Interview: Answer Quality Evaluation
# ===========================

# ---- Imports ----
import pandas as pd
import numpy as np
import re, os, pickle, random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
from textstat import flesch_reading_ease
import nltk

# Download stopwords if not present
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

# ---- Config ----
os.makedirs("models", exist_ok=True)
EMBEDDER_MODEL = "all-MiniLM-L6-v2"

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
        import re
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
    import requests
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

# ---- Feature Extraction ----
def extract_features(df, sbert_model, tfidf_vectorizer=None, fit_vectorizer=True):
    # SBERT embeddings cosine similarity
    q_emb = sbert_model.encode(df['Question'].tolist(), convert_to_numpy=True)
    a_emb = sbert_model.encode(df['Answer'].tolist(), convert_to_numpy=True)
    cos_sim = np.array([util.cos_sim([q],[a]).item() for q,a in zip(q_emb,a_emb)])

    # TF-IDF similarity
    if fit_vectorizer:
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix_a = tfidf_vectorizer.fit_transform(df['Answer']).toarray()
        tfidf_matrix_q = tfidf_vectorizer.transform(df['Question']).toarray()
    else:
        tfidf_matrix_a = tfidf_vectorizer.transform(df['Answer']).toarray()
        tfidf_matrix_q = tfidf_vectorizer.transform(df['Question']).toarray()
    tfidf_sim = np.array([np.dot(q,a)/(np.linalg.norm(q)*np.linalg.norm(a)+1e-8) for q,a in zip(tfidf_matrix_q, tfidf_matrix_a)])

    # Length ratio
    q_len = df['Question'].apply(lambda x: max(1,len(x.split()))).values
    a_len = df['Answer'].apply(lambda x: max(1,len(x.split()))).values
    length_ratio = (a_len / q_len).clip(0,5)

    # Stopword ratio
    stopword_ratio = df['Answer'].apply(lambda x: len([w for w in x.split() if w in STOPWORDS])/max(1,len(x.split()))).values

    # Readability
    readability = df['Answer'].apply(lambda x: flesch_reading_ease(x)).values

    # Combine features
    X = np.vstack([cos_sim, tfidf_sim, length_ratio, stopword_ratio, readability]).T
    return X, tfidf_vectorizer

# ---- Generate labels (synthetic for training) ----
def categorize(cos_sim_score):
    if cos_sim_score >= 0.75: return 3
    elif cos_sim_score >= 0.5: return 2
    elif cos_sim_score >= 0.25: return 1
    else: return 0

# ---- Train Weighted Logistic Regression ----
def train_model():
    combined_df = load_and_clean_datasets()
    sbert_model = SentenceTransformer(EMBEDDER_MODEL)

    # Extract features
    X, tfidf_vectorizer = extract_features(combined_df, sbert_model, fit_vectorizer=True)
    y = np.array([categorize(s) for s in X[:,0]])  # synthetic label using cosine similarity

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Compute class weights for imbalance
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))

    # Train Logistic Regression
    clf = LogisticRegression(max_iter=1000, class_weight=class_weight_dict)
    clf.fit(X_train, y_train)

    # Save models
    with open("models/logistic_regressor.pkl","wb") as f: pickle.dump(clf,f)
    with open("models/sbert_model.pkl","wb") as f: pickle.dump(sbert_model,f)
    with open("models/tfidf_vectorizer.pkl","wb") as f: pickle.dump(tfidf_vectorizer,f)

    # Metrics
    y_pred = clf.predict(X_test)
    print("📊 Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.2f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return clf, sbert_model, tfidf_vectorizer, combined_df

# ---- Evaluate User Answer ----
def evaluate_answer(user_ans, ref_ans, clf, sbert_model, tfidf_vectorizer):
    # Features
    q_emb = sbert_model.encode([ref_ans], convert_to_numpy=True)
    a_emb = sbert_model.encode([user_ans], convert_to_numpy=True)
    cos_sim = float(util.cos_sim(q_emb, a_emb).item())

    q_tfidf = tfidf_vectorizer.transform([ref_ans]).toarray()
    a_tfidf = tfidf_vectorizer.transform([user_ans]).toarray()
    tfidf_sim = np.dot(q_tfidf, a_tfidf.T)[0][0] / (np.linalg.norm(q_tfidf)*np.linalg.norm(a_tfidf)+1e-8)

    q_len = max(1,len(ref_ans.split()))
    a_len = max(1,len(user_ans.split()))
    length_ratio = (a_len/q_len)

    stopword_ratio = len([w for w in user_ans.split() if w in STOPWORDS])/max(1,len(user_ans.split()))
    readability = flesch_reading_ease(user_ans)

    X_user = np.array([[cos_sim, tfidf_sim, length_ratio, stopword_ratio, readability]])
    pred_class = clf.predict(X_user)[0]

    feedback_map = {0:"Poor answer. Review the concept.",
                    1:"Average answer. Needs improvement.",
                    2:"Good answer. Minor improvements needed.",
                    3:"Excellent answer!"}
    return pred_class, feedback_map[pred_class]

# ---- Interactive Demo ----
def run_demo():
    clf, sbert_model, tfidf_vectorizer, combined_df = train_model()
    domains = combined_df['Domain'].unique().tolist()
    print("Available Domains:", domains)
    selected_domain = input("Select a domain: ").strip()
    domain_qas = combined_df[combined_df['Domain']==selected_domain].reset_index(drop=True)

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
