# ===========================
# AI Mock Interview: SVM Answer Quality Evaluation
# ===========================

# ---- Imports ----
import pandas as pd
import numpy as np
import re, os, pickle, random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC
from nltk.corpus import stopwords
from textstat import flesch_reading_ease
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

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

# ---- Generate labels (synthetic) ----
def categorize(cos_sim_score):
    if cos_sim_score >= 0.75: return 3
    elif cos_sim_score >= 0.5: return 2
    elif cos_sim_score >= 0.25: return 1
    else: return 0

# ---- Train Weighted SVM ----
# def train_model_svm():
#     combined_df = load_and_clean_datasets()
#     sbert_model = SentenceTransformer(EMBEDDER_MODEL)

#     # Features
#     X, tfidf_vectorizer = extract_features(combined_df, sbert_model, fit_vectorizer=True)
#     y = np.array([categorize(s) for s in X[:,0]])

#     # Class weights
#     classes = np.unique(y)
#     class_weights = compute_class_weight('balanced', classes=classes, y=y)
#     class_weight_dict = dict(zip(classes, class_weights))

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#     # SVM Classifier with probability output
#     clf = SVC(kernel='linear', class_weight=class_weight_dict, probability=True, random_state=42)
#     clf.fit(X_train, y_train)

#     # Save models
#     with open("models/svm_model.pkl","wb") as f: pickle.dump(clf,f)
#     with open("models/sbert_model.pkl","wb") as f: pickle.dump(sbert_model,f)
#     with open("models/tfidf_vectorizer.pkl","wb") as f: pickle.dump(tfidf_vectorizer,f)

#     # Metrics
#     y_pred = clf.predict(X_test)
#     print("📊 SVM Metrics:")
#     print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
#     print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.2f}")
#     print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#     print("\nClassification Report:\n", classification_report(y_test, y_pred))

#     return clf, sbert_model, tfidf_vectorizer, combined_df

from imblearn.over_sampling import SMOTE
from sklearn.utils import compute_sample_weight

from imblearn.over_sampling import SMOTE
from sklearn.utils import compute_sample_weight
from collections import Counter

from imblearn.over_sampling import SMOTE
from sklearn.utils import compute_sample_weight
from collections import Counter

# def train_model_svm():
#     combined_df = load_and_clean_datasets()
#     sbert_model = SentenceTransformer(EMBEDDER_MODEL)

#     # ---------------- Features ----------------
#     X, tfidf_vectorizer = extract_features(combined_df, sbert_model, fit_vectorizer=True)
#     y = np.array([categorize(s) for s in X[:,0]])

#     # ---------------- Domain weights ----------------
#     # Give more weight to underrepresented domains
#     domain_weights = compute_sample_weight(class_weight='balanced', y=combined_df['Domain'])

#     # ---------------- Train-test split ----------------
#     X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
#         X, y, domain_weights, test_size=0.2, random_state=42, stratify=y
#     )

#     # ---------------- SMOTE for class imbalance ----------------
#     class_counts = Counter(y_train)
#     min_samples = min([count for count in class_counts.values() if count > 1])
#     k_neighbors = max(1, min(5, min_samples - 1))  # Ensure k_neighbors < smallest class samples

#     smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
#     X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

#     # ---------------- Train SVM ----------------
#     clf = SVC(kernel='linear', probability=True, random_state=42)
#     clf.fit(X_train_res, y_train_res)  # domain weighting optional, SVC doesn't take sample_weight with SMOTE

#     # ---------------- Save models ----------------
#     with open("models/svm_model.pkl","wb") as f: pickle.dump(clf,f)
#     with open("models/sbert_model.pkl","wb") as f: pickle.dump(sbert_model,f)
#     with open("models/tfidf_vectorizer.pkl","wb") as f: pickle.dump(tfidf_vectorizer,f)

#     # ---------------- Metrics ----------------
#     y_pred = clf.predict(X_test)
#     print("📊 SVM Metrics (Hybrid - SMOTE + Domain Weighting):")
#     print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
#     print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.2f}")
#     print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#     print("\nClassification Report:\n", classification_report(y_test, y_pred))

#     return clf, sbert_model, tfidf_vectorizer, combined_df


def plot_class_distribution(y, sample_weights=None, title="Class Distribution"):
    counter = Counter()
    if sample_weights is None:
        counter.update(y)
        classes = list(counter.keys())
        counts = list(counter.values())
    else:
        # Weighted counts per class
        for label, w in zip(y, sample_weights):
            counter[label] += w
        classes = sorted(counter.keys())
        counts = [counter[c] for c in classes]

    plt.figure(figsize=(6,4))
    sns.barplot(x=classes, y=counts, palette="viridis")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of samples" if sample_weights is None else "Weighted count")
    plt.show()


def train_model_svm():
    combined_df = load_and_clean_datasets()
    sbert_model = SentenceTransformer(EMBEDDER_MODEL)

    # ---------------- Features ----------------
    X, tfidf_vectorizer = extract_features(combined_df, sbert_model, fit_vectorizer=True)
    y = np.array([categorize(s) for s in X[:,0]])

    # ---------------- Before SMOTE ----------------
    plot_class_distribution(y, title="Before SMOTE")

    # ---------------- Domain weights ----------------
    domain_weights = compute_sample_weight(class_weight='balanced', y=combined_df['Domain'])
    plot_class_distribution(y, sample_weights=domain_weights, title="Weighted by Domain (Before SMOTE)")

    # ---------------- Train-test split ----------------
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, domain_weights, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------- SMOTE for class imbalance ----------------
    class_counts = Counter(y_train)
    min_samples = min([count for count in class_counts.values() if count > 1])
    k_neighbors = max(1, min(5, min_samples - 1))

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # ---------------- After SMOTE ----------------
    plot_class_distribution(y_train_res, title="After SMOTE")

    # ---------------- Train SVM ----------------
    clf = SVC(kernel='linear', probability=True, random_state=42)
    clf.fit(X_train_res, y_train_res)

    # ---------------- Save models ----------------
    with open("models/svm_model.pkl","wb") as f: pickle.dump(clf,f)
    with open("models/sbert_model.pkl","wb") as f: pickle.dump(sbert_model,f)
    with open("models/tfidf_vectorizer.pkl","wb") as f: pickle.dump(tfidf_vectorizer,f)

    # ---------------- Metrics ----------------
    y_pred = clf.predict(X_test)
    print("📊 SVM Metrics (Hybrid - SMOTE + Domain Weighting):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.2f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return clf, sbert_model, tfidf_vectorizer, combined_df


# ---- Evaluate User Answer ----
def evaluate_answer_svm(user_ans, ref_ans, clf, sbert_model, tfidf_vectorizer):
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
    pred_proba = clf.predict_proba(X_user)[0]

    feedback_map = {0:"Poor answer. Review the concept.",
                    1:"Average answer. Needs improvement.",
                    2:"Good answer. Minor improvements needed.",
                    3:"Excellent answer!"}

    return pred_class, pred_proba, feedback_map[pred_class]

# ---- Interactive Demo ----
def run_demo_svm():
    clf, sbert_model, tfidf_vectorizer, combined_df = train_model_svm()
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
        pred_class, pred_proba, feedback = evaluate_answer_svm(user_ans, row['Answer'], clf, sbert_model, tfidf_vectorizer)
        print(f"Predicted Quality: {pred_class} | Feedback: {feedback}")
        print(f"Class Probabilities: {pred_proba}")
        scores.append(pred_class)

    overall = np.mean(scores)
    print(f"\nOverall Score (Average Class): {overall:.2f}")

# ---- Main ----
if __name__ == "__main__":
    run_demo_svm()
