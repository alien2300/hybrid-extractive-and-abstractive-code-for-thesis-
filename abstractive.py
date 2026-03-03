import os
import json
import torch
import re
import numpy as np
from collections import Counter
from transformers import (
    AutoTokenizer, T5ForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq, GenerationConfig
)
from datasets import Dataset
from tqdm import tqdm

os.system("pip install -q bert-score sentence-transformers")
from bert_score import score as bert_score_fn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Package installation complete.")

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

def rouge_n(reference, prediction, n):
    ref_tokens = bangla_tokenize(reference)
    pred_tokens = bangla_tokenize(prediction)
    if len(ref_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0
    ref_ngrams = Counter(get_ngrams(ref_tokens, n))
    pred_ngrams = Counter(get_ngrams(pred_tokens, n))
    common = sum((ref_ngrams & pred_ngrams).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred_ngrams.values())
    recall = common / sum(ref_ngrams.values())
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def lcs_length(x, y):
    m, n = len(x), len(y)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def rouge_l(reference, prediction):
    ref_tokens = bangla_tokenize(reference)
    pred_tokens = bangla_tokenize(prediction)
    if len(ref_tokens) == 0 or len(pred_tokens) == 0:
        return 0.0
    lcs = lcs_length(ref_tokens, pred_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def compute_rouge(references, predictions):
    r1_list, r2_list, rl_list = [], [], []
    for ref, pred in zip(references, predictions):
        ref_clean = ref.strip()
        pred_clean = pred.strip()
        if len(pred_clean) < 2 or len(ref_clean) < 2:
            r1_list.append(0); r2_list.append(0); rl_list.append(0)
            continue
        r1_list.append(rouge_n(ref_clean, pred_clean, 1) * 100)
        r2_list.append(rouge_n(ref_clean, pred_clean, 2) * 100)
        rl_list.append(rouge_l(ref_clean, pred_clean) * 100)
    return {
        "rouge1": np.mean(r1_list),
        "rouge2": np.mean(r2_list),
        "rougeL": np.mean(rl_list),
    }

# ROUGE Verification
print("\n===== ROUGE Verification =====")
test_ref = "বাংলাদেশে ম্যালেরিয়ায় আক্রান্ত হওয়া লোকের সংখ্যা কমছে না"
test_pred = "বাংলাদেশে ম্যালেরিয়ার কারণে মারা গেছে এক জন ব্যক্তি"
print(f"  Ref tokens : {bangla_tokenize(test_ref)}")
print(f"  Pred tokens: {bangla_tokenize(test_pred)}")
test_scores = compute_rouge([test_ref], [test_pred])
print(f"  ROUGE-1: {test_scores['rouge1']:.2f}")
print(f"  ROUGE-2: {test_scores['rouge2']:.2f}")
print(f"  ROUGE-L: {test_scores['rougeL']:.2f}")
if test_scores['rouge1'] > 0:
    print("  ROUGE working correctly!")
else:
    print("  Warning: Issue detected!")
print("=" * 35 + "\n")

# Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {DEVICE}")

# Semantic Similarity Model
sem_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu"
)

# Data Loading
def load_jsonl(filepath, max_samples=None):
    texts, summaries = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line.strip())
            if "text" in item and "summary" in item:
                text = str(item["text"]).strip()
                summ = str(item["summary"]).strip()
                if text and summ:
                    texts.append(text)
                    summaries.append(summ)
    return texts, summaries

train_texts, train_sums = load_jsonl("/content/bengali_train.jsonl", 3000)
val_texts, val_sums = load_jsonl("/content/bengali_val.jsonl", 500)
test_texts, test_sums = load_jsonl("/content/bengali_test.jsonl", 500)
print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

train_ds = Dataset.from_dict({"text": train_texts, "summary": train_sums})
val_ds = Dataset.from_dict({"text": val_texts, "summary": val_sums})

# Model & Tokenizer
model_id = "csebuetnlp/banglat5"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = T5ForConditionalGeneration.from_pretrained(model_id).to(DEVICE)
print(f"pad: {tokenizer.pad_token_id} | eos: {tokenizer.eos_token_id} | vocab: {tokenizer.vocab_size}")

# Generation Config
model.generation_config = GenerationConfig(
    max_length=128,
    num_beams=4,
    no_repeat_ngram_size=3,
    repetition_penalty=2.0,
    length_penalty=1.0,
    early_stopping=True,
    decoder_start_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# Preprocessing
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["text"], max_length=512, truncation=True, padding=False,
    )
    labels = tokenizer(
        examples["summary"], max_length=128, truncation=True, padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing...")
tokenized_train = train_ds.map(preprocess_function, batched=True,
    remove_columns=train_ds.column_names)
tokenized_val = val_ds.map(preprocess_function, batched=True,
    remove_columns=val_ds.column_names)

# Compute Metrics (training eval)
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
    labels = np.clip(labels, 0, tokenizer.vocab_size - 1)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Debug: show first 3 samples
    print("\n--- Eval (first 3) ---")
    for i in range(min(3, len(decoded_preds))):
        p = decoded_preds[i][:200]
        r = decoded_labels[i][:200]
        s1 = rouge_n(decoded_labels[i], decoded_preds[i], 1) * 100
        print(f"  P[{i}]: {p}")
        print(f"  R[{i}]: {r}")
        print(f"  -> ROUGE-1 for this sample: {s1:.2f}")

    scores = compute_rouge(decoded_labels, decoded_preds)
    return scores

# Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=8,
    warmup_steps=50,
    weight_decay=0.01,
    predict_with_generate=True,
    generation_max_length=128,
    fp16=False,
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="rouge2",
    greater_is_better=True,
    report_to="none",
    label_smoothing_factor=0.1,
)

# Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\n" + "="*60)
print(" Training started! ~1.5-2 hours")
print("="*60 + "\n")

trainer.train()
print("\nTraining complete!\n")

# Test Set Evaluation
print("="*60)
print(" Generating summaries on test set...")
print("="*60)

model.eval()
predictions = []
references = test_sums

for i in tqdm(range(len(test_texts)), desc="Generating"):
    inputs = tokenizer(
        test_texts[i], max_length=512, truncation=True, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(**inputs)
    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    predictions.append(pred)

# Sample Predictions
print("\n" + "="*60)
print(" 10 Samples:")
print("="*60)
for i in range(min(10, len(predictions))):
    print(f"\n[{i+1}]")
    print(f"  Input: {test_texts[i][:120]}...")
    print(f"  Ref  : {references[i][:120]}")
    print(f"  Pred : {predictions[i][:120]}")

# ROUGE Calculation
print("\n" + "="*60)
print(" ROUGE Calculation (Bangla ROUGE)...")
print("="*60)

rouge_scores = compute_rouge(references, predictions)
rouge1 = rouge_scores["rouge1"]
rouge2 = rouge_scores["rouge2"]
rougeL = rouge_scores["rougeL"]
print(f"  ROUGE-1: {rouge1:.2f}")
print(f"  ROUGE-2: {rouge2:.2f}")
print(f"  ROUGE-L: {rougeL:.2f}")

# BERTScore
print("\n" + "="*60)
print(" BERTScore Calculation...")
print("="*60)

valid_p, valid_r = [], []
for p, r in zip(predictions, references):
    if len(p.strip()) >= 3 and len(r.strip()) >= 3:
        valid_p.append(p)
        valid_r.append(r)

if len(valid_p) > 0:
    P, R, F1 = bert_score_fn(valid_p, valid_r, lang="bn", verbose=True, device=DEVICE)
    bs_p = P.mean().item() * 100
    bs_r = R.mean().item() * 100
    bs_f = F1.mean().item() * 100
    print(f"  Precision: {bs_p:.2f} | Recall: {bs_r:.2f} | F1: {bs_f:.2f}")
else:
    bs_p = bs_r = bs_f = 0.0
    print("  No valid predictions!")

# Semantic Similarity
print("\n" + "="*60)
print(" Semantic Similarity Calculation...")
print("="*60)

if len(valid_p) > 0:
    pe = sem_model.encode(valid_p, show_progress_bar=True)
    re_emb = sem_model.encode(valid_r, show_progress_bar=True)
    sims = [cosine_similarity([a], [b])[0][0] * 100 for a, b in zip(pe, re_emb)]
    sem_sim = np.mean(sims)
    print(f"  Semantic Similarity: {sem_sim:.2f}")
else:
    sem_sim = 0.0
    print("  No valid predictions!")

# Final Results
print("\n" + "="*60)
print(" Final Results (Test Set)")
print("="*60)
print(f"  ROUGE-1: {rouge1:.2f}")
print(f"  ROUGE-2: {rouge2:.2f}")
print(f"  ROUGE-L: {rougeL:.2f}")
print(f"  BERTScore Precision: {bs_p:.2f}")
print(f"  BERTScore Recall: {bs_r:.2f}")
print(f"  BERTScore F1: {bs_f:.2f}")
print(f"  Semantic Similarity: {sem_sim:.2f}")
print(f"  Valid Predictions: {len(valid_p)}/{len(predictions)}")
print("="*60)
print("\nAll done!")
