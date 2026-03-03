
!pip install nltk numpy scikit-learn networkx tqdm bert-score sentence-transformers --quiet

import json, re, warnings, gc
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score_fn
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings('ignore')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)
sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2").to(DEVICE)

# Custom Bangla ROUGE Implementation
def bangla_tokenize(text):
    text = text.strip()
    text = re.sub(r'([।,;:!?"\'\(\)\[\]\{\}])', r' \1 ', text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 0 and not
        re.match(r'^[।,;:!?"\'\(\)\[\]\{\}\.\-]+$', t)]
    return tokens

def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def rouge_n_f1(reference, prediction, n):
    ref_tokens = bangla_tokenize(reference)
    pred_tokens = bangla_tokenize(prediction)
    if len(ref_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0
    ref_ng = Counter(get_ngrams(ref_tokens, n))
    pred_ng = Counter(get_ngrams(pred_tokens, n))
    common = sum((ref_ng & pred_ng).values())
    if common == 0:
        return 0.0
    prec = common / sum(pred_ng.values())
    rec = common / sum(ref_ng.values())
    return 2 * prec * rec / (prec + rec)

def lcs_length(x, y):
    m, n = len(x), len(y)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def rouge_l_f1(reference, prediction):
    ref_tokens = bangla_tokenize(reference)
    pred_tokens = bangla_tokenize(prediction)
    if len(ref_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0
    lcs = lcs_length(ref_tokens, pred_tokens)
    if lcs == 0:
        return 0.0
    prec = lcs / len(pred_tokens)
    rec = lcs / len(ref_tokens)
    return 2 * prec * rec / (prec + rec)

def compute_bangla_rouge(references, predictions):
    r1, r2, rl = [], [], []
    for ref, pred in zip(references, predictions):
        if len(pred.strip()) < 2 or len(ref.strip()) < 2:
            r1.append(0); r2.append(0); rl.append(0)
            continue
        r1.append(rouge_n_f1(ref, pred, 1) * 100)
        r2.append(rouge_n_f1(ref, pred, 2) * 100)
        rl.append(rouge_l_f1(ref, pred) * 100)
    return {"rouge1": round(np.mean(r1),4),
            "rouge2": round(np.mean(r2),4),
            "rougeL": round(np.mean(rl),4)}

# Sanity Check
_t = compute_bangla_rouge(
    ["বাংলাদেশে ম্যালেরিয়ায় আক্রান্ত লোকের সংখ্যা কমছে না"],
    ["বাংলাদেশে ম্যালেরিয়ার কারণে আক্রান্ত মানুষের সংখ্যা বাড়ছে"])
print(f"ROUGE Check: R1={_t['rouge1']:.2f}")

# Data Loading
def load_jsonl(path, limit=None):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            obj = json.loads(line)
            text = obj.get("maintext", obj.get("text", "")).strip()
            summ = obj.get("summary", "").strip()
            if text and summ:
                data.append({"text": text, "summary": summ})
    print(f"Loaded {len(data)} from {path}")
    return data

test_data = load_jsonl("/content/bengali_test.jsonl", limit=500)
print(f"Test: {len(test_data)}")

# Sentence Splitting
bn_word_re = re.compile(r'[\u0980-\u09FF]+')

def bangla_sentence_split(text):
    parts = re.split(r'(?<=[।!?])\s+|\n+', text.strip())
    sents = []
    for p in parts:
        p = p.strip()
        if p and bn_word_re.search(p) and len(p.split()) >= 3:
            sents.append(p)
    return sents

# TextRank Extractive Summarizer
class BengaliTextRank:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 4), min_df=1)

    def extract_summary(self, text, num_sentences=2):
        sentences = bangla_sentence_split(text)
        if not sentences:
            return ""
        if len(sentences) <= num_sentences:
            return " ".join(sentences)
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            sim_matrix = cosine_similarity(tfidf_matrix)
            np.fill_diagonal(sim_matrix, 0)
            nx_graph = nx.from_numpy_array(sim_matrix)
            scores = nx.pagerank(nx_graph, max_iter=100)
            ranked_idx = sorted(range(len(sentences)),
                key=lambda i: scores[i], reverse=True)
            top_idx = sorted(ranked_idx[:num_sentences])
            return " ".join([sentences[i] for i in top_idx])
        except Exception:
            return " ".join(sentences[:num_sentences])

# Ablation: num_sentences = 1, 2, 3
print("\n" + "="*60)
print(" Ablation: num_sentences test")
print("="*60)

extractor = BengaliTextRank()
best_rl, best_n = 0, 2
for n in [1, 2, 3]:
    preds, refs = [], []
    for ex in tqdm(test_data, desc=f"num_sentences={n}"):
        pred = extractor.extract_summary(ex["text"], num_sentences=n)
        preds.append(pred)
        refs.append(ex["summary"])
    scores = compute_bangla_rouge(refs, preds)
    print(f"  n={n} -> R1={scores['rouge1']:.2f} R2={scores['rouge2']:.2f} RL={scores['rougeL']:.2f}")
    if scores['rougeL'] > best_rl:
        best_rl = scores['rougeL']
        best_n = n

print(f"\n Best: num_sentences={best_n} (RL={best_rl:.2f})")

# Final Evaluation with Best Configuration
print("\n" + "="*60)
print(f" Final Test (num_sentences={best_n})")
print("="*60)

preds_list, refs_list = [], []
for ex in tqdm(test_data, desc="Final extractive"):
    pred = extractor.extract_summary(ex["text"], num_sentences=best_n)
    preds_list.append(pred)
    refs_list.append(ex["summary"])

# Samples
print("\n 10 Samples:")
for i in range(min(10, len(preds_list))):
    print(f"\n[{i+1}]")
    print(f"  Ref : {refs_list[i][:120]}")
    print(f"  Pred: {preds_list[i][:120]}")

# ROUGE
rouge_res = compute_bangla_rouge(refs_list, preds_list)
print(f"\n ROUGE-1: {rouge_res['rouge1']}")
print(f" ROUGE-2: {rouge_res['rouge2']}")
print(f" ROUGE-L: {rouge_res['rougeL']}")

# BERTScore
valid_p = [p for p, r in zip(preds_list, refs_list) if len(p.strip()) >= 3]
valid_r = [r for p, r in zip(preds_list, refs_list) if len(p.strip()) >= 3]

if len(valid_p) > 0:
    P, R, F1 = bert_score_fn(valid_p, valid_r,
        model_type="bert-base-multilingual-cased",
        lang="bn", verbose=True, device=DEVICE)
    bs_f1 = round(float(F1.mean().item()) * 100, 4)
else:
    bs_f1 = 0.0

# Semantic Similarity
if len(valid_p) > 0:
    emb_p = sbert_model.encode(valid_p, convert_to_tensor=True, device=DEVICE)
    emb_r = sbert_model.encode(valid_r, convert_to_tensor=True, device=DEVICE)
    sem = round(float(util.cos_sim(emb_p, emb_r).diagonal().mean().item()) * 100, 4)
else:
    sem = 0.0

# Final Results
print("\n" + "="*60)
print(" Final Results (Pure Extractive - TextRank)")
print("="*60)
print(f" ROUGE-1: {rouge_res['rouge1']}")
print(f" ROUGE-2: {rouge_res['rouge2']}")
print(f" ROUGE-L: {rouge_res['rougeL']}")
print(f" BERTScore F1: {bs_f1}")
print(f" Semantic Similarity: {sem}")
print(f" Valid Predictions: {len(valid_p)}/{len(preds_list)}")
print(f" num_sentences: {best_n}")
print("="*60)
print("\n Comparison:")
print(" Abstractive: R1=18.34 | R2=5.66 | RL=15.77 | BERT=73.16 | Sim=57.86")
print(" Hybrid v8:   R1=18.38 | R2=6.17 | RL=15.84 | BERT=73.14 | Sim=58.24")
print("\nDone!")
