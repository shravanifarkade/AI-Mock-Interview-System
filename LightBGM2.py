# ===========================
# AI Mock Interview: SBERT + LightGBM (Domain Weighted, Dynamic Thresholds)
# ===========================

import os, re, pickle, requests
import pandas as pd
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
from textstat import flesch_reading_ease
import lightgbm as lgb
import nltk

# ---------------- Setup ----------------
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

os.makedirs("models", exist_ok=True)
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64  # SBERT batch size

# ---------------- Utilities ----------------
def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def is_link(text):
    if not text: return True
    return bool(re.search(r'(http[s]?://|www\.)', text))

# ---------------- Load & Clean Dataset ----------------
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

    combined = pd.concat(dfs, ignore_index=True)
    combined['Question'] = combined['Question'].apply(clean_text)
    combined['Answer'] = combined['Answer'].apply(clean_text)
    combined = combined[(combined['Question'] != "") & (combined['Answer'] != "")]
    combined = combined[~combined['Answer'].apply(is_link)]
    combined.drop_duplicates(inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # ---------------- Domain Weights ----------------
    domain_counts = combined['Domain'].value_counts()
    domain_weights = {domain: domain_counts.max()/count for domain, count in domain_counts.items()}
    combined['sample_weight'] = combined['Domain'].map(domain_weights)

    print(f"✅ Total cleaned QAs: {len(combined)}")
    print(combined['Domain'].value_counts())
    print(f"Domain weights applied: {domain_weights}")
    return combined

# ---------------- Feature Extraction ----------------
def embed_texts(sbert_model, texts):
    embeddings = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start+BATCH_SIZE]
        batch_emb = sbert_model.encode(batch, convert_to_numpy=True)
        embeddings.append(batch_emb)
    return np.vstack(embeddings)

def extract_features(df, sbert_model, tfidf_vectorizer, pca_emb=None):
    question_emb = embed_texts(sbert_model, df['Question'].tolist())
    answer_emb = embed_texts(sbert_model, df['Answer'].tolist())

    # Reduce embedding dimensions if PCA provided
    if pca_emb:
        question_emb = pca_emb.transform(question_emb)
        answer_emb = pca_emb.transform(answer_emb)

    cosine_sim = np.array([util.cos_sim(q.reshape(1,-1), a.reshape(1,-1)).item()
                           for q,a in zip(question_emb, answer_emb)]).reshape(-1,1)

    q_tfidf = tfidf_vectorizer.transform(df['Question']).toarray()
    a_tfidf = tfidf_vectorizer.transform(df['Answer']).toarray()
    tfidf_sim = np.array([np.dot(q,a)/(np.linalg.norm(q)*np.linalg.norm(a)+1e-8)
                           for q,a in zip(q_tfidf, a_tfidf)]).reshape(-1,1)

    q_len = df['Question'].apply(lambda x: max(1,len(x.split()))).values
    a_len = df['Answer'].apply(lambda x: max(1,len(x.split()))).values
    length_ratio = (a_len / q_len).clip(0,5).reshape(-1,1)

    stopword_ratio = df['Answer'].apply(lambda x: len([w for w in x.split() if w in STOPWORDS])/max(1,len(x.split()))).values.reshape(-1,1)

    readability = df['Answer'].apply(lambda x: flesch_reading_ease(x)).values.reshape(-1,1)

    X = np.hstack([question_emb, answer_emb, cosine_sim, tfidf_sim, length_ratio, stopword_ratio, readability])
    return X, cosine_sim, tfidf_sim

# ---------------- Generate Labels Dynamically ----------------
def generate_labels_dynamic(df, sbert_model, tfidf_vectorizer, pca_emb=None):
    X, cosine_sim, tfidf_sim = extract_features(df, sbert_model, tfidf_vectorizer, pca_emb)

    labels = np.zeros(len(df))
    # Compute percentile thresholds per domain
    labels_list = []
    for domain in df['Domain'].unique():
        idxs = df[df['Domain']==domain].index
        cosine_domain = cosine_sim[idxs].flatten()
        tfidf_domain = tfidf_sim[idxs].flatten()

        cos_thresh = np.percentile(cosine_domain, 80)
        tfidf_thresh = np.percentile(tfidf_domain, 80)

        # assign labels based on dynamic threshold
        for i in idxs:
            score = 0
            if cosine_sim[i] >= cos_thresh: score += 1
            if tfidf_sim[i] >= tfidf_thresh: score += 1
            labels[i] = score
    return labels.astype(int)

# ---------------- Train Model ----------------
def train_model_lgb_dynamic():
    combined_df = load_and_clean_datasets()
    sbert_model = SentenceTransformer(EMBEDDER_MODEL)

    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_vectorizer.fit(combined_df['Question'].tolist() + combined_df['Answer'].tolist())

    # Reduce embedding dimension to avoid best gain=-inf
    all_emb = embed_texts(sbert_model, combined_df['Question'].tolist() + combined_df['Answer'].tolist())
    pca_emb = PCA(n_components=100, random_state=42)
    pca_emb.fit(all_emb)

    X, _, _ = extract_features(combined_df, sbert_model, tfidf_vectorizer, pca_emb)
    y = generate_labels_dynamic(combined_df, sbert_model, tfidf_vectorizer, pca_emb)

    # Train-test split
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, combined_df['sample_weight'], test_size=0.2, stratify=y, random_state=42
    )
    
    test_classes = np.unique(y_test)
    print("Classes in test set:", test_classes)
    all_classes = np.arange(4)  # assuming classes are 0,1,2,3
    missing_classes = set(all_classes) - set(test_classes)
    if missing_classes:
                    print(f"⚠️ Warning: Missing classes in test set: {missing_classes}")
    

    clf = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.07,
        num_leaves=32,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42
    )

    clf.fit(X_train, y_train, sample_weight=w_train)
    y_pred = clf.predict(X_test)

    print("📊 Metrics on hold-out test set:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.2f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save models
    with open("models/lgb_model.pkl","wb") as f: pickle.dump(clf,f)
    with open("models/sbert_model.pkl","wb") as f: pickle.dump(sbert_model,f)
    with open("models/tfidf_vectorizer.pkl","wb") as f: pickle.dump(tfidf_vectorizer,f)
    with open("models/pca_emb.pkl","wb") as f: pickle.dump(pca_emb,f)

    return clf, sbert_model, tfidf_vectorizer, pca_emb, combined_df

# ---------------- Evaluate Answer ----------------
def compute_tfidf_sim(ref_ans, user_ans, tfidf_vectorizer):
    q_vec = tfidf_vectorizer.transform([ref_ans]).toarray().flatten()
    a_vec = tfidf_vectorizer.transform([user_ans]).toarray().flatten()
    sim = np.dot(q_vec, a_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(a_vec) + 1e-8)
    return np.array([[sim]])

def evaluate_answer_dynamic(user_ans, ref_ans, clf, sbert_model, tfidf_vectorizer, pca_emb):
    q_emb = embed_texts(sbert_model, [ref_ans])
    a_emb = embed_texts(sbert_model, [user_ans])
    q_emb = pca_emb.transform(q_emb)
    a_emb = pca_emb.transform(a_emb)

    cosine_sim = np.array([util.cos_sim(q_emb, a_emb).item()]).reshape(-1,1)
    tfidf_sim = compute_tfidf_sim(ref_ans, user_ans, tfidf_vectorizer)

    q_len = max(1,len(ref_ans.split()))
    a_len = max(1,len(user_ans.split()))
    length_ratio = np.array([[a_len/q_len]])
    stopword_ratio = np.array([[len([w for w in user_ans.split() if w in STOPWORDS])/max(1,len(user_ans.split()))]])
    readability = np.array([[flesch_reading_ease(user_ans)]])

    X_user = np.hstack([q_emb, a_emb, cosine_sim, tfidf_sim, length_ratio, stopword_ratio, readability])
    pred_class = clf.predict(X_user)[0]

    feedback_map = {
        0:"Poor answer. Review the concept.",
        1:"Average answer. Needs improvement.",
        2:"Good answer. Minor improvements needed.",
        3:"Excellent answer!"
    }
    return pred_class, feedback_map[pred_class]

# ---------------- Main Demo ----------------
def run_demo_dynamic(user_answers_dict=None, num_questions=5):
    clf, sbert_model, tfidf_vectorizer, pca_emb, combined_df = train_model_lgb_dynamic()
    domains = combined_df['Domain'].unique().tolist()
    print("Available Domains:", domains)

    selected_domain = input("Select a domain: ").strip()
    domain_qas = combined_df[combined_df['Domain'] == selected_domain]

    if len(domain_qas) <= num_questions:
        domain_qas = domain_qas.reset_index(drop=True)
    else:
        selected_indices = np.random.choice(domain_qas.index, size=num_questions, replace=False)
        domain_qas = domain_qas.loc[selected_indices].reset_index(drop=True)

    # Automatic evaluation: use provided answers or ask interactively
    if user_answers_dict and selected_domain in user_answers_dict:
        user_answers = user_answers_dict[selected_domain]
    else:
        user_answers = []
        for idx, row in domain_qas.iterrows():
            print(f"\nQuestion {idx+1}: {row['Question']}")
            print("Type your answer (press ENTER on an empty line when done):")
            ans_lines = []
            while True:
                try:
                    line = input()
                    if line.strip() == "":
                        break
                    ans_lines.append(line)
                except EOFError:
                    break
            full_ans = " ".join(ans_lines).strip()
            if not full_ans:
                full_ans = "[No Answer Provided]"
            user_answers.append(full_ans)

    scores = []
    for row, user_ans in zip(domain_qas.itertuples(), user_answers):
        pred_class, feedback = evaluate_answer_dynamic(user_ans, row.Answer, clf, sbert_model, tfidf_vectorizer, pca_emb)
        print(f"\nQ: {row.Question}")
        print(f"Your Answer: {user_ans}")
        print(f"Predicted Quality: {pred_class} | Feedback: {feedback}")
        scores.append(pred_class)

    overall = np.mean(scores)
    print(f"\nOverall Score (Average Class): {overall:.2f}")

# ---------------- Run ----------------
if __name__ == "__main__":
    run_demo_dynamic()
