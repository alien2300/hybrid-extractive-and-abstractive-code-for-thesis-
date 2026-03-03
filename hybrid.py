!pip install -q transformers[torch] datasets accelerate sentencepiece rouge-score \
    bert-score sentence-transformers nltk numpy evaluate tqdm networkx scikit-learn

import os, json, re, random, math, warnings, gc, sys
import numpy as np
import torch
import networkx as nx
from tqdm import tqdm
from collections import Counter
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
    DataCollatorWithPadding,
    T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq, GenerationConfig
)
from bert_score import score as bert_score_fn
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore")

# CUDA Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)
if DEVICE == "cuda":
    !nvidia-smi -L

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2").to(DEVICE)
bn_word_re = re.compile(r'[\u0980-\u09FF]+')

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Custom Bangla ROUGE Implementation
def bangla_tokenize(text):
    if not text or not isinstance(text, str):
        return []
    text = text.strip()
    if len(text) == 0:
        return []
    text = re.sub(r'([।,;:!?"\'\(\)\[\]\{\}])', r' \1 ', text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 0 and not
        re.match(r'^[।,;:!?"\'\(\)\[\]\{\}\.\-]+$', t)]
    return tokens

def get_ngrams(tokens, n):
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def rouge_n_f1(reference, prediction, n):
    ref_tokens = bangla_tokenize(reference)
    pred_tokens = bangla_tokenize(prediction)
    if len(ref_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0
    ref_ng = Counter(get_ngrams(ref_tokens, n))
    pred_ng = Counter(get_ngrams(pred_tokens, n))
    if not ref_ng or not pred_ng:
        return 0.0
    common = sum((ref_ng & pred_ng).values())
    if common == 0:
        return 0.0
    prec = common / sum(pred_ng.values())
    rec = common / sum(ref_ng.values())
    return 2 * prec * rec / (prec + rec)

def lcs_length(x, y):
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Memory-efficient LCS for long sequences
    if m > 500 or n > 500:
        x, y = x[:500], y[:500]
        m, n = len(x), len(y)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]

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
        if not pred or not ref or len(pred.strip()) < 2 or len(ref.strip()) < 2:
            r1.append(0); r2.append(0); rl.append(0)
            continue
        r1.append(rouge_n_f1(ref, pred, 1) * 100)
        r2.append(rouge_n_f1(ref, pred, 2) * 100)
        rl.append(rouge_l_f1(ref, pred) * 100)
    return {"rouge1": round(np.mean(r1),4), "rouge2": round(np.mean(r2),4),
            "rougeL": round(np.mean(rl),4)}

_t = compute_bangla_rouge(
    ["বাংলাদেশে ম্যালেরিয়ায় আক্রান্ত লোকের সংখ্যা কমছে না"],
    ["বাংলাদেশে ম্যালেরিয়ার কারণে আক্রান্ত মানুষের সংখ্যা বাড়ছে"])
print(f"ROUGE Check: R1={_t['rouge1']:.2f}" if _t['rouge1'] > 0 else "ROUGE BROKEN")

# Data Loading (with quality checks)
def load_jsonl(path, limit=None):
    data = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            text = obj.get("maintext", obj.get("text", "")).strip()
            summ = obj.get("summary", "").strip()
            # Skip bad data
            if not text or not summ or len(text) < 50 or len(summ) < 10:
                skipped += 1
                continue
            if not bn_word_re.search(text) or not bn_word_re.search(summ):
                skipped += 1
                continue
            data.append({"text": text, "summary": summ})
    print(f"Loaded {len(data)} from {path} (skipped: {skipped})")
    return Dataset.from_list(data)

train_ds0 = load_jsonl("/content/bengali_train.jsonl", limit=3000)
val_ds0   = load_jsonl("/content/bengali_val.jsonl", limit=500)
test_ds0  = load_jsonl("/content/bengali_test.jsonl", limit=500)
print(f"Train: {len(train_ds0)}  Val: {len(val_ds0)}  Test: {len(test_ds0)}")

# Sentence Splitting (improved)
def bangla_sentence_split(text):
    if not text:
        return []
    parts = re.split(r'(?<=[।!?])\s+|\n+', text.strip())
    sents = []
    for p in parts:
        p = p.strip()
        if p and bn_word_re.search(p) and len(bn_word_re.findall(p)) >= 2:
            sents.append(p)
    return sents

# Oracle Label Generation + Extractor Training
MAX_SENTS_PER_DOC = 35
ORACLE_K = 5
TOP_K = 8
bn_tok_re = re.compile(r'[\u0980-\u09FF]+')

def bn_tokens(text):
    return bn_tok_re.findall(text)

def f1_overlap(pred_toks, ref_toks):
    if not pred_toks or not ref_toks:
        return 0.0
    p = Counter(pred_toks); r = Counter(ref_toks)
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    prec = overlap / max(1, sum(p.values()))
    rec  = overlap / max(1, sum(r.values()))
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if len(tokens) >= n else []

def rouge12_f1(prediction, reference):
    p, r = bn_tokens(prediction), bn_tokens(reference)
    r1 = f1_overlap(p, r)
    r2 = f1_overlap(ngrams(p,2), ngrams(r,2))
    return 0.5 * (r1 + r2)

def greedy_oracle_select(sentences, reference, max_k):
    selected, selected_idx, best = [], [], 0.0
    remaining = list(range(len(sentences)))
    for _ in range(max_k):
        best_gain, best_j = 0.0, None
        for j in remaining:
            cand = " ".join(selected + [sentences[j]])
            gain = rouge12_f1(cand, reference) - best
            if gain > best_gain:
                best_gain, best_j = gain, j
        if best_j is None or best_gain <= 1e-8:
            break
        selected.append(sentences[best_j])
        selected_idx.append(best_j)
        best += best_gain
        remaining.remove(best_j)
    return set(selected_idx)

def build_extractor_dataset(ds):
    rows = []
    for ex in tqdm(ds, desc="Oracle labels"):
        sents = bangla_sentence_split(ex["text"])[:MAX_SENTS_PER_DOC]
        if len(sents) < 2:
            continue
        oracle_idx = greedy_oracle_select(sents, ex["summary"], max_k=ORACLE_K)
        for si, s in enumerate(sents):
            rows.append({"sentence": s, "label": 1 if si in oracle_idx else 0})
    return Dataset.from_list(rows)

sent_train_ds = build_extractor_dataset(train_ds0)
sent_val_ds   = build_extractor_dataset(val_ds0)

pos_rows = [sent_train_ds[i] for i in range(len(sent_train_ds)) if sent_train_ds[i]["label"] == 1]
neg_rows = [sent_train_ds[i] for i in range(len(sent_train_ds)) if sent_train_ds[i]["label"] == 0]
print(f"Pos: {len(pos_rows)} | Neg: {len(neg_rows)}")
pos_rows = pos_rows * 2
sent_train_balanced = Dataset.from_list(pos_rows + neg_rows).shuffle(seed=SEED)
cleanup_cuda()

# Train Extractor
extractor_model_id = "sagorsarker/bangla-bert-base"
ext_tok = AutoTokenizer.from_pretrained(extractor_model_id, use_fast=True)
ext_model = AutoModelForSequenceClassification.from_pretrained(
    extractor_model_id, num_labels=2
).to(DEVICE)

def ext_preprocess(batch):
    return ext_tok(batch["sentence"], truncation=True, max_length=256)

sent_train_tok = sent_train_balanced.map(ext_preprocess, batched=True, remove_columns=["sentence"])
sent_val_tok   = sent_val_ds.map(ext_preprocess, batched=True, remove_columns=["sentence"])

def ext_compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = float((preds == labels).mean())
    return {"accuracy": acc}

ext_args = TrainingArguments(
    output_dir="./extractor",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none",
    seed=SEED,
)

ext_trainer = Trainer(
    model=ext_model, args=ext_args,
    train_dataset=sent_train_tok, eval_dataset=sent_val_tok,
    data_collator=DataCollatorWithPadding(ext_tok),
    compute_metrics=ext_compute_metrics
)

print("\n=== Training Extractor ===")
ext_trainer.train()
ext_eval = ext_trainer.evaluate()
print(f"Extractor Best Accuracy: {ext_eval.get('eval_accuracy', 0)*100:.2f}%")
cleanup_cuda()

# Combined Ranking (BanglaBERT + TextRank + Evidence)
@torch.no_grad()
def banglabert_scores(sentences, batch_size=32):
    ext_model.eval()
    scores = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        try:
            inputs = ext_tok(batch, truncation=True, padding=True,
                max_length=256, return_tensors="pt").to(DEVICE)
            logits = ext_model(**inputs).logits
            prob = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(prob.detach().cpu().numpy().tolist())
        except Exception as e:
            # OOM fallback
            scores.extend([0.5] * len(batch))
    return np.array(scores)

def textrank_scores(sentences):
    if len(sentences) <= 1:
        return np.ones(len(sentences))
    try:
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
        tfidf = vec.fit_transform(sentences)
        sim = sklearn_cosine(tfidf)
        np.fill_diagonal(sim, 0)
        # Check for zero matrix
        if sim.sum() < 1e-10:
            return np.ones(len(sentences)) / len(sentences)
        G = nx.from_numpy_array(sim)
        # Higher max_iter for convergence
        pr = nx.pagerank(G, max_iter=200, tol=1e-6)
        scores = np.array([pr.get(i, 0) for i in range(len(sentences))])
        mn, mx = scores.min(), scores.max()
        if mx - mn > 1e-10:
            scores = (scores - mn) / (mx - mn)
        else:
            scores = np.ones(len(sentences)) / len(sentences)
        return scores
    except Exception as e:
        return np.ones(len(sentences)) / len(sentences)

def evidence_scores(sentences):
    scores = np.zeros(len(sentences))
    for i, sent in enumerate(sentences):
        score = 0.0
        nums = len(re.findall(r'[০-৯0-9]+', sent))
        score += min(nums * 0.1, 0.3)
        score += len(re.findall(r'[০-৯0-9]+\s*(?:শতাংশ|%)', sent)) * 0.15
        score += len(re.findall(r'(?:টাকা|কোটি|লাখ|ডলার|মিলিয়ন|বিলিয়ন)', sent)) * 0.1
        score += min(len(re.findall(
            r'(?:সরকার|মন্ত্রণালয়|বিশ্ববিদ্যালয়|কমিশন|সংস্থা|পুলিশ|আদালত|সংসদ|জাতিসংঘ)',
            sent)) * 0.1, 0.25)
        score += min(len(re.findall(
            r'(?:বলেন|বলেছেন|জানান|জানিয়েছেন|দাবি করেন|ঘোষণা করেন|মনে করেন)',
            sent)) * 0.12, 0.2)
        position_bonus = max(0, 0.15 - (i / max(len(sentences), 1)) * 0.15)
        score += position_bonus
        scores[i] = min(score, 1.0)
    mx = scores.max()
    if mx > 0:
        scores = scores / mx
    return scores

def combined_ranking(sentences):
    bb = banglabert_scores(sentences)
    tr = textrank_scores(sentences)
    ev = evidence_scores(sentences)
    # Proper 0-1 normalization for all
    for arr in [bb, tr, ev]:
        mn, mx = arr.min(), arr.max()
        if mx - mn > 1e-10:
            arr[:] = (arr - mn) / (mx - mn)
        else:
            arr[:] = 0.5
    # Weighted combination
    return 0.50 * bb + 0.25 * tr + 0.25 * ev

def mmr_select(sentences, scores, top_k=8, lambda_param=0.7):
    if len(sentences) <= top_k:
        return list(range(len(sentences)))
    try:
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
        tfidf = vec.fit_transform(sentences)
        sim = sklearn_cosine(tfidf)
    except:
        return sorted(np.argsort(-scores)[:top_k].tolist())
    selected, candidates = [], list(range(len(sentences)))
    first = max(candidates, key=lambda i: scores[i])
    selected.append(first); candidates.remove(first)
    while len(selected) < top_k and candidates:
        best_i, best_val = None, -1e9
        for c in candidates:
            rel = scores[c]
            red = max(sim[c][s] for s in selected)
            val = lambda_param * rel - (1 - lambda_param) * red
            if val > best_val:
                best_val, best_i = val, c
        if best_i is None:
            break
        selected.append(best_i); candidates.remove(best_i)
    return sorted(selected)  # Original order

# Build Reordered Dataset
def reorder_document(text, top_k=8, max_sents=35):
    sents = bangla_sentence_split(text)[:max_sents]
    if len(sents) <= top_k:
        return " ".join(sents)
    scores = combined_ranking(sents)
    top_idx = mmr_select(sents, scores, top_k=top_k, lambda_param=0.7)
    top_set = set(top_idx)
    rest_idx = [i for i in range(len(sents)) if i not in top_set]
    return " ".join([sents[i] for i in top_idx] + [sents[i] for i in rest_idx])

def build_hybrid_dataset(ds):
    rows = []
    for ex in tqdm(ds, desc="Building reordered"):
        reordered = reorder_document(ex["text"], top_k=TOP_K)
        # Skip if reordered is too short
        if len(reordered.strip()) < 20:
            reordered = ex["text"]
        rows.append({"input_text": reordered, "target_text": ex["summary"]})
    return Dataset.from_list(rows)

print("\n=== Building Reordered Datasets (Combined + MMR) ===")
hy_train = build_hybrid_dataset(train_ds0)
hy_val   = build_hybrid_dataset(val_ds0)
hy_test  = build_hybrid_dataset(test_ds0)
cleanup_cuda()

# Fine-tune BanglaT5
gen_model_id = "csebuetnlp/banglat5"
gen_tok = AutoTokenizer.from_pretrained(gen_model_id, use_fast=False)
gen_model = T5ForConditionalGeneration.from_pretrained(gen_model_id).to(DEVICE)

gen_model.generation_config = GenerationConfig(
    max_length=128,
    num_beams=4,
    no_repeat_ngram_size=3,
    repetition_penalty=2.0,
    length_penalty=1.0,
    early_stopping=True,
    decoder_start_token_id=gen_tok.pad_token_id,
    eos_token_id=gen_tok.eos_token_id,
    pad_token_id=gen_tok.pad_token_id,
)

def gen_preprocess(batch):
    model_inputs = gen_tok(batch["input_text"], max_length=512, truncation=True, padding=False)
    labels = gen_tok(batch["target_text"], max_length=128, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

hy_train_tok = hy_train.map(gen_preprocess, batched=True, remove_columns=["input_text","target_text"])
hy_val_tok   = hy_val.map(gen_preprocess, batched=True, remove_columns=["input_text","target_text"])

def compute_metrics_fn(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    labels = np.where(labels != -100, labels, gen_tok.pad_token_id)
    preds  = np.where(preds != -100, preds, gen_tok.pad_token_id)
    preds  = np.clip(preds, 0, gen_tok.vocab_size - 1)
    labels = np.clip(labels, 0, gen_tok.vocab_size - 1)
    dp = gen_tok.batch_decode(preds, skip_special_tokens=True)
    dl = gen_tok.batch_decode(labels, skip_special_tokens=True)
    non_empty = sum(1 for p in dp if len(p.strip()) > 3)
    if non_empty < len(dp) * 0.5:
        print(f"  Warning: {len(dp)-non_empty}/{len(dp)} empty predictions!")
    print(f"\n  [Eval] P[0]: {dp[0][:100]}")
    print(f"  [Eval] R[0]: {dl[0][:100]}")
    scores = compute_bangla_rouge(dl, dp)
    print(f"  [Eval] R1={scores['rouge1']:.2f} R2={scores['rouge2']:.2f} RL={scores['rougeL']:.2f}")
    return scores

# Stable training with gradient clipping
gen_args = Seq2SeqTrainingArguments(
    output_dir="./hybrid_v9_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=8,
    warmup_steps=80,
    weight_decay=0.01,
    max_grad_norm=1.0,
    predict_with_generate=True,
    generation_max_length=128,
    fp16=False,
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    label_smoothing_factor=0.1,
    report_to="none",
    seed=SEED,
)

gen_trainer = Seq2SeqTrainer(
    model=gen_model,
    args=gen_args,
    train_dataset=hy_train_tok,
    eval_dataset=hy_val_tok,
    data_collator=DataCollatorForSeq2Seq(gen_tok, model=gen_model, label_pad_token_id=-100),
    compute_metrics=compute_metrics_fn
)

print("\n" + "="*60)
print(" Hybrid v9 Training Started! (8 epochs)")
print("="*60)

gen_trainer.train()
print("\nTraining Complete!")
cleanup_cuda()

# Final Test: 5-candidate Smart Re-ranking
print("\n" + "="*60)
print(" FINAL TEST (5-candidate ROUGE+Semantic)")
print("="*60)

# num_beams >= num_return_sequences
TEST_GEN_CONFIG = GenerationConfig(
    max_length=128,
    num_beams=6,
    num_return_sequences=5,
    no_repeat_ngram_size=3,
    repetition_penalty=2.0,
    length_penalty=1.0,
    early_stopping=True,
    decoder_start_token_id=gen_tok.pad_token_id,
    eos_token_id=gen_tok.eos_token_id,
    pad_token_id=gen_tok.pad_token_id,
)

def smart_rerank(candidates, evidence_text):
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]
    valid = [c for c in candidates if c and len(c.strip()) >= 3]
    if not valid:
        return candidates[0] if candidates else ""
    # ROUGE scores (already 0-1 scale)
    rouge_sc = []
    for c in valid:
        r1 = rouge_n_f1(evidence_text, c, 1)
        r2 = rouge_n_f1(evidence_text, c, 2)
        rl = rouge_l_f1(evidence_text, c)
        rouge_sc.append(0.3 * r1 + 0.3 * r2 + 0.4 * rl)
    # Semantic score (cosine, no scale mismatch)
    try:
        ev_emb = sbert_model.encode([evidence_text], convert_to_tensor=True, device=DEVICE)
        c_embs = sbert_model.encode(valid, convert_to_tensor=True, device=DEVICE)
        sem_sc = util.cos_sim(c_embs, ev_emb).squeeze(-1).cpu().numpy()
        sem_sc = np.clip(sem_sc, 0, 1)
    except Exception:
        sem_sc = np.full(len(valid), 0.5)
    # Both ROUGE (0-1) and Semantic (0-1) — same scale
    final = []
    for i in range(len(valid)):
        r = rouge_sc[i]
        s = float(sem_sc[i]) if i < len(sem_sc) else 0.5
        final.append(0.6 * r + 0.4 * s)
    return valid[np.argmax(final)]

gen_model.eval()
preds_list, refs_list = [], []
failed = 0

for ex in tqdm(hy_test, desc="Final test (smart re-ranking)"):
    try:
        inputs = gen_tok(ex["input_text"], max_length=512,
            truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = gen_model.generate(**inputs, generation_config=TEST_GEN_CONFIG)
        cands = [gen_tok.decode(out[i], skip_special_tokens=True) for i in range(min(5, len(out)))]
        non_empty_cands = [c for c in cands if len(c.strip()) >= 3]
        if not non_empty_cands:
            preds_list.append(cands[0] if cands else "")
        else:
            best = smart_rerank(non_empty_cands, ex["input_text"][:500])
            preds_list.append(best)
    except Exception as e:
        # OOM/error fallback
        failed += 1
        preds_list.append("")
    refs_list.append(ex["target_text"])

if failed > 0:
    print(f"  Warning: {failed} samples failed during generation")

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

# Valid prediction count
valid_p = [p for p, r in zip(preds_list, refs_list) if p and len(p.strip()) >= 3]
valid_r = [r for p, r in zip(preds_list, refs_list) if p and len(p.strip()) >= 3]
print(f"\n Valid predictions: {len(valid_p)}/{len(preds_list)}")

if len(valid_p) < len(preds_list) * 0.9:
    print(f"  Warning: {len(preds_list)-len(valid_p)} invalid predictions!")

# BERTScore
if len(valid_p) > 0:
    P, R, F1 = bert_score_fn(
        valid_p, valid_r,
        model_type="bert-base-multilingual-cased",
        lang="bn", verbose=True, device=DEVICE,
        batch_size=16
    )
    bs_f1 = round(float(F1.mean().item()) * 100, 4)
else:
    bs_f1 = 0.0

# Semantic Similarity
if len(valid_p) > 0:
    emb_p = sbert_model.encode(valid_p, convert_to_tensor=True, device=DEVICE, batch_size=32)
    emb_r = sbert_model.encode(valid_r, convert_to_tensor=True, device=DEVICE, batch_size=32)
    sem = round(float(util.cos_sim(emb_p, emb_r).diagonal().mean().item()) * 100, 4)
else:
    sem = 0.0

# Final Results
print("\n" + "="*60)
print(" Final Results (Hybrid v9)")
print("="*60)
print(f" ROUGE-1: {rouge_res['rouge1']}")
print(f" ROUGE-2: {rouge_res['rouge2']}")
print(f" ROUGE-L: {rouge_res['rougeL']}")
print(f" BERTScore F1: {bs_f1}")
print(f" Semantic Similarity: {sem}")
print(f" Valid Predictions: {len(valid_p)}/{len(preds_list)}")
print("="*60)
print("\n Previous Results:")
print(" Extractive:  R1=12.62 | R2=3.57 | RL=10.09 | BERT=69.51 | Sim=64.57")
print(" Abstractive: R1=18.34 | R2=5.66 | RL=15.77 | BERT=73.16 | Sim=57.86")
print(" Hybrid v8:   R1=18.38 | R2=6.17 | RL=15.84 | BERT=73.14 | Sim=58.24")
print("\nDone!")

