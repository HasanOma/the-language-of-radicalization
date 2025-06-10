# rq3_classification.py
"""
RQ-3  Ideology text-classification
----------------------------------
* baselines    SVM-TFIDF  &  RF-linguistic-features (5-fold CV, artefacts saved)
* transformer  (huggingface model id)
"""
from __future__ import annotations
import argparse, logging, random, sys, os
from pathlib import Path
from collections import defaultdict

import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_recall_fscore_support, confusion_matrix,
                             classification_report)
from sklearn.model_selection import (StratifiedKFold,
                                     StratifiedGroupKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import torch
import torch.nn.functional as F                  
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup, logging as hf_log,
                          set_seed)
import shap
from tqdm.auto import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ───────── CLI ─────────────────────────────────────────────────
P = argparse.ArgumentParser()
P.add_argument("--data", required=True, type=Path)
P.add_argument("--model", default="distilbert-base-uncased")
P.add_argument("--epochs", type=int, default=5)
P.add_argument("--batch",  type=int, default=16)
P.add_argument("--grad-accum", type=int, default=1)
P.add_argument("--run-name")
P.add_argument("--no-baselines", action="store_true")
P.add_argument("--no-transformer", action="store_true")
args = P.parse_args()

# ───────── output folder ──────────────────────────────────────
RUN    = args.run_name or f"auto-{args.model.split('/')[-1]}"
OUT    = Path(f"results-revised/rq3/{RUN}"); OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ───────── logging & seeds ────────────────────────────────────
hf_log.set_verbosity_error()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("rq3")
RNG = 42
np.random.seed(RNG); random.seed(RNG); set_seed(RNG)

EMO = ["fear","anger","surprise","sadness","joy","disgust"]
RHE = ["incl","excl"]

DATASET_SIZE_TAG = args.data.parent.name.replace("data", "")
if not DATASET_SIZE_TAG.endswith("k"):
    DATASET_SIZE_TAG = args.data.parent.name 

# ───────── results writer ─────────────────────────────────────
RUN_CSV = OUT / "metrics.csv"
if not RUN_CSV.exists():
    pd.DataFrame(columns=["dataset", "model", "fold", "metric", "value"])\
      .to_csv(RUN_CSV, index=False)

def log_metric(model_name, fold, metric, value):
    pd.DataFrame([{
        "dataset": DATASET_SIZE_TAG, 
        "model": model_name,
        "fold": fold,
        "metric": metric,
        "value": value
    }]).to_csv(RUN_CSV, mode="a", header=False, index=False)


def dump_preds_cm(y_true, y_pred, cls_names, out_dir):
    """
    Persist everything that lets us inspect a baseline offline.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"y_true": y_true, "y_pred": y_pred})\
      .to_csv(out_dir / "preds.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)  
    pd.DataFrame(cm, index=cls_names, columns=cls_names)\
      .to_csv(out_dir / "cm.csv", index_label="true\\pred")

    (out_dir / "report.txt").write_text(
        classification_report(y_true, y_pred, digits=3)
    )

# ───────── 1 load data & derive class weights ────────────────
df = pd.read_feather(args.data)
df = df[df.ideology != "unknown"].copy()
cls_freq   = df.ideology.value_counts().sort_index()
cls_weight = (1 / cls_freq).to_dict()
labels     = sorted(cls_freq.index)
labels_idx = list(range(len(labels)))       
label2id   = {l:i for i,l in enumerate(labels)}
id2label   = {i:l for l,i in label2id.items()}
cls_weight_int = {label2id[k]: v for k, v in cls_weight.items()}
df["label"] = df.ideology.map(label2id)
log.info("dataset: %d samples, %s", len(df), cls_freq.to_dict())

# ───────── helpers ───────────────────────────────────────────
def score(y_true, prob, pred):
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(pd.get_dummies(y_true), prob,
                            average="micro", multi_class="ovr")
    except ValueError:
        auc = np.nan
    return dict(acc=accuracy_score(y_true, pred),
                f1=f1, prec=prec, rec=rec, auc=auc)

def cm_plot(cm, title):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.ylabel("true"); plt.xlabel("pred"); plt.title(title)
    plt.tight_layout(); plt.savefig(OUT/f"{title}_cm.png", dpi=300); plt.close()

# ───────── 2 TRANSFORMER support fn (3-fold) ─────────────────
def run_transformer(train_index, test_index, fold_tag="fold"):
    """
    Fine-tune transformer and return best macro-F1 for this fold.
    """
    X_tr_txt = df.loc[train_index, "cleaned_content"]
    X_te_txt = df.loc[test_index,  "cleaned_content"]
    y_tr     = df.loc[train_index, "label"].to_numpy()
    y_te     = df.loc[test_index,  "label"].to_numpy()

    tok = AutoTokenizer.from_pretrained(args.model)

    class TXT(Dataset):
        def __init__(self, txt, lab):
            enc = tok(list(txt), truncation=True, padding=True, max_length=256)
            self.enc = enc; self.lab = list(lab)
        def __len__(self):               return len(self.lab)
        def __getitem__(self, i):
            t = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            t["labels"] = torch.tensor(int(self.lab[i])); return t

    dl_tr = DataLoader(TXT(X_tr_txt, y_tr), batch_size=args.batch, shuffle=True)
    dl_te = DataLoader(TXT(X_te_txt, y_te), batch_size=args.batch)

    model = (AutoModelForSequenceClassification
             .from_pretrained(args.model,
                              num_labels=len(labels),
                              id2label=id2label, label2id=label2id)
             .to(DEVICE))

    w = torch.tensor([cls_weight[id2label[i]] for i in range(len(labels))],
                     dtype=torch.float, device=DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)

    opt   = AdamW(model.parameters(), lr=2e-5)
    sched = get_linear_schedule_with_warmup(opt,
                                            len(dl_tr)//10,
                                            len(dl_tr)*args.epochs)

    if torch.cuda.is_available():
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        autocast_ctx = autocast
    else:
        from contextlib import nullcontext
        scaler = None; autocast_ctx = nullcontext

    best_f1 = -1; best_state = {}

    def eval_epoch():
        model.eval(); preds, probs = [], []
        with torch.no_grad():
            for batch in dl_te:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                with autocast_ctx():
                    out = model(**batch)
                probs.append(torch.softmax(out.logits, -1).cpu())
                preds.append(out.logits.argmax(-1).cpu())
        prob_mat = torch.cat(probs).numpy()
        pred_vec = torch.cat(preds).numpy()
        return score(y_te, prob_mat, pred_vec), pred_vec

    for ep in range(1, args.epochs+1):
        model.train(); pbar = tqdm(dl_tr, desc=f"{fold_tag}-ep{ep}", leave=False)
        for step, batch in enumerate(pbar, 1):
            batch = {k:v.to(DEVICE) for k,v in batch.items()}
            with autocast_ctx():
                out  = model(**batch)
                loss = loss_fn(out.logits, batch["labels"]) / args.grad_accum
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % args.grad_accum == 0:
                if scaler:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(); sched.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        val, preds = eval_epoch()
        if val["f1"] > best_f1:
            best_f1, best_preds = val["f1"], preds
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            log.info("%s epoch%-2d f1=%.3f", fold_tag, ep, best_f1)

    model.load_state_dict(best_state)
    torch.save(best_state, OUT/f"best_transformer_{fold_tag}.pt")
    cm_plot(confusion_matrix(y_te, best_preds), f"transformer_{fold_tag}")
    return best_f1

# ───────── 3 BASELINES ────────────
if not args.no_baselines:
    log.info("🧮 baselines …")
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    log.info("  ── 5-fold StratifiedKFold CV (ideology)")
    svm = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2))),
        ("svm",   LinearSVC(class_weight=cls_weight_int)),
    ])
    rf = Pipeline([
        ("scale", StandardScaler(with_mean=False)),
        ("rf",    RandomForestClassifier(n_estimators=400,
                                        class_weight=cls_weight_int,
                                        random_state=RNG)),
    ])

    for name, pipe, Xcol in [
        ("svm_tfidf", svm, "cleaned_content"),
        ("rf_feats",  rf,  EMO+RHE),
    ]:
        log.info("  ── %s", name)
        X_data = (df.loc[:, Xcol] if isinstance(Xcol, str)
                  else df.loc[:, Xcol])

        f1s = []
        for fold_idx, (tr, te) in enumerate(cv5.split(X_data, df.label)):
            X_tr, X_te = X_data.iloc[tr], X_data.iloc[te]
            y_tr, y_te = df.label.iloc[tr], df.label.iloc[te]

            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)
            f1     = f1_score(y_te, y_pred, average="macro")
            f1s.append(f1)

            fold_out = OUT / name / f"fold{fold_idx}"
            dump_preds_cm(y_te, y_pred, labels, fold_out)
            cm_plot(confusion_matrix(y_te, y_pred, labels=labels_idx),
                    f"{name}_fold{fold_idx}")
            log.info("    fold%-2d f1=%.3f", fold_idx, f1)

        mean, std = np.mean(f1s), np.std(f1s)
        log.info("  %-9s 5-fold F1 = %.3f ± %.3f", name, mean, std)
        log_metric(name, "cv-mean", "f1", mean)
        log_metric(name, "cv-std",  "f1", std)

# ───────── 4 TRANSFORMER 3-fold ───────
if not args.no_transformer and args.model.lower() not in {"none", "skip"}:
    cv_t = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RNG)
    f1_scores = []
    for k, (tr, te) in enumerate(cv_t.split(df.index, df.label, df.creator_id)):
        f1 = run_transformer(tr, te, fold_tag=f"cv{k+1}")
        f1_scores.append(f1)

    mdl_tag = f"bert-{args.model.split('/')[-1]}"
    log_metric(mdl_tag, "cv-mean", "f1", np.mean(f1_scores))
    log_metric(mdl_tag, "cv-std",  "f1", np.std(f1_scores))
    log.info("transformer 3-fold macro-F1 = %.3f ± %.3f",
             np.mean(f1_scores), np.std(f1_scores))

    # ── SHAP (single best fold) ───────────────────────────────
    if max(f1_scores) > 0:
        best_fold = int(np.argmax(f1_scores))
        _, te_idx = list(cv_t.split(df.index, df.label, df.creator_id))[best_fold]
        sample_texts = df.loc[te_idx, "cleaned_content"].sample(
            100, random_state=RNG).astype(str).tolist()

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model_path = OUT / f"best_transformer_cv{best_fold+1}.pt"
        expl_model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=len(labels),
            id2label=id2label, label2id=label2id).to(DEVICE)
        expl_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        expl_model.eval()

        class ModelWrapper(torch.nn.Module):
            def __init__(self, mdl, tok):
                super().__init__(); self.mdl, self.tok = mdl, tok
            def forward(self, texts):
                if isinstance(texts, np.ndarray):
                    texts = texts.tolist()
                enc = self.tok(texts, padding=True, truncation=True,
                               return_tensors="pt", max_length=256).to(DEVICE)
                with torch.no_grad():
                    logits = self.mdl(**enc).logits
                    probs  = torch.softmax(logits, dim=-1)
                    scores = probs.max(dim=1).values
                return scores.cpu().numpy()

        wrapper = ModelWrapper(expl_model, tokenizer)
        masker  = shap.maskers.Text(tokenizer, mask_token=tokenizer.mask_token)
        explainer = shap.Explainer(wrapper, masker=masker, algorithm="partition")

        sv        = explainer(sample_texts)
        shap_vals = list(sv.values)
        max_len   = max(len(a) for a in shap_vals)
        P         = np.zeros((len(shap_vals), max_len))
        for i, arr in enumerate(shap_vals):
            P[i, :len(arr)] = np.abs(arr)
        mean_abs  = P.mean(axis=0)

        top_idx   = np.argsort(mean_abs)[-15:]
        feats     = [sv.feature_names[i] for i in top_idx][::-1]
        vals      = mean_abs[top_idx][::-1].astype(float)

        plt.figure(figsize=(8,5))
        plt.barh(y=[str(tok) for tok in feats], width=vals)
        plt.xlabel("Mean |SHAP value|")
        plt.title("Top 15 most important tokens (global)")
        plt.tight_layout()
        plt.savefig(OUT/"bert_shap_manual_bar.png", dpi=300); plt.close()

        df_shap = pd.DataFrame({"token": feats, "mean_abs": vals})
        lines = [
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"Token & Mean |SHAP| \\",
            r"\midrule",
        ]
        for _, r in df_shap.iterrows():
            lines.append(f"{r.token} & {r.mean_abs:.3f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (OUT/"bert_shap_top15.tex").write_text("\n".join(lines))

# ───────── 5 End ───────────────────────────────────────────
log.info("🏁 artefacts saved in %s", OUT.resolve())
sys.exit(0)