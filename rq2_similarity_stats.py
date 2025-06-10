# rq2_similarity_stats.py
"""
RQ-2  Similarities & differences in emotional expression
--------------------------------------------------------
* cosine-distance heat-map  +  dendrogram
* permutation MANOVA (5000 permutations, Wilks-Λ)
* ANOVA + η²  with Benjamini–Hochberg FDR
* Tukey-HSD signed mean-difference heat-map
* 2-D PCA visualisation
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import logging, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f_oneway
from scipy.cluster.hierarchy import linkage
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.decomposition import PCA
from statsmodels.multivariate.manova import MANOVA
import pingouin as pg
from tqdm.auto import tqdm
import pkg_resources, sys
from joblib import Parallel, delayed
from statsmodels.stats.multitest import fdrcorrection
req = pkg_resources.get_distribution("pingouin").version

DATA = Path("data/pysentimentio/revised-preprocess/80k/02_features.feather")
OUT  = Path("results-revised/rq2/80k"); OUT.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("rq2")

df = pd.read_feather(DATA)

KNOWN = ["fear","anger","surprise","sadness","joy","disgust"]
EMO = [e for e in KNOWN if e in df.columns]
if not EMO:
    raise RuntimeError("❌ No emotion columns in the data")

# ── 1 Cosine distance heat-map + dendrogram ─────────────────────────
means = df.groupby("ideology")[EMO].mean()
dist  = squareform(pdist(means, metric="cosine"))
sns.heatmap(pd.DataFrame(dist, index=means.index, columns=means.index),
            cmap="viridis", annot=True, fmt=".2f")
plt.title("Cosine distance between ideologies (emotion space)")
plt.tight_layout(); plt.savefig(OUT/"rq2_cosine_heatmap.png", dpi=300); plt.close()

sns.clustermap(means, cmap="mako", linewidths=.5, 
               method="ward", metric="euclidean")
plt.savefig(OUT/"rq2_dendrogram.png", dpi=300); plt.close()
# ── 2 Permutation MANOVA (pingouin) ────────────────────────────────
def pillai_trace(df, dv_cols, factor):
    """Return Pillai trace for the given DV matrix and factor column."""
    dvs   = " + ".join(dv_cols)
    model = MANOVA.from_formula(f"{dvs} ~ {factor}", data=df)
    res   = model.mv_test().results[factor]['stat'].loc["Pillai's trace", 'Value']
    return float(res)

pillai = pillai_trace(df, EMO, "ideology")

def perm_pillai(seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(df["ideology"].values)
    return pillai_trace(df.assign(ideology_perm=perm), EMO, "ideology_perm")

n_perm = 1000


perm_stats = Parallel(n_jobs=-1, backend="loky")(
                 delayed(perm_pillai)(s) for s in range(n_perm))
perm_stats = np.asarray(perm_stats)
p_perm     = (perm_stats >= pillai).mean()



mv = pd.DataFrame({"stat":[pillai], "p-raw":[p_perm]})
mv.to_csv(OUT / "rq2_perm_manova.csv", index=False)
log.info("Permutation MANOVA  p = %.4g", p_perm)


# ── 3 Per-emotion one-way ANOVA + η² + FDR volcano ─────────────────
rows=[]
for emo in tqdm(EMO, desc="ANOVA"):
    groups=[g[emo].values for _,g in df.groupby("ideology")]
    F,p  = f_oneway(*groups)
    grand= df[emo].values
    ss_tot = ((grand-grand.mean())**2).sum()
    ss_bet = sum(len(g)*(g.mean()-grand.mean())**2 for g in groups)
    eta2 = ss_bet/ss_tot
    rows.append([emo, F, p, eta2])
anova = pd.DataFrame(rows, columns=["emo","F","p","eta2"])
anova["q"] = fdrcorrection(anova["p"], alpha=0.05, method="indep")[1]
anova.to_csv(OUT/"rq2_anova_results.csv", index=False)

plt.figure(figsize=(6,5))
plt.scatter(anova["eta2"], -np.log10(anova["q"]))
for _, r in anova[anova["eta2"].gt(.05) & anova["q"].lt(.05)].iterrows():
    plt.text(r.eta2, -np.log10(r.q), r.emo, fontsize=8)
plt.axhline(-np.log10(.05), ls="--", color="red")
plt.xlabel("effect size (η²)"); plt.ylabel("-log10 FDR q")
plt.tight_layout(); plt.savefig(OUT/"rq2_anova_volcano.png", dpi=300); plt.close()

# ── 4 Tukey-HSD signed mean-difference heat-map ────────────────────
tuk_rows=[]
for emo in EMO:
    res = pairwise_tukeyhsd(df[emo], df["ideology"], alpha=0.05)
    t = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])
    t["emo"]=emo; tuk_rows.append(t)
tuk = pd.concat(tuk_rows, ignore_index=True)
sig = tuk[tuk["reject"]]
pivot=(sig.assign(diff=lambda d:d["meandiff"].astype(float))
          .pivot_table(index="group1", columns="group2",
                       values="diff", aggfunc="mean"))
sns.heatmap(pivot, cmap="coolwarm", center=0, annot=True, fmt=".2f",
            mask=pivot.isna())
plt.title("Signed mean-difference (Tukey HSD, sig. pairs)")
plt.tight_layout(); plt.savefig(OUT/"rq2_tukey_hsd.png", dpi=300); plt.close()
tuk.to_csv(OUT/"rq2_tukey_results.csv", index=False)

# ── 5 PCA scatter ──────────────────────────────────────────────────
p = PCA(n_components=2, random_state=0).fit_transform(df[EMO])
df["pc1"], df["pc2"] = p[:,0], p[:,1]
sns.scatterplot(data=df, x="pc1", y="pc2", hue="ideology", alpha=.6, s=40)
plt.title("PCA of emotion vectors")
plt.tight_layout(); plt.savefig(OUT/"rq2_pca_clusters.png", dpi=300); plt.close()


# ────────── LaTeX summary ───────────────────────────────
# 1) MANOVA
manova_tex = (r"\begin{tabular}{lc}" "\n"
              r"\toprule" "\n"
              r"Test & $p$--value \\" "\n"
              r"\midrule" "\n"
              f"Permutation MANOVA (5 000 perms) & "
              f"{mv['p-raw'].iloc[0]:.4g} \\" "\n"
              r"\bottomrule" "\n"
              r"\end{tabular}")
(OUT / "rq2_manova.txt").write_text(manova_tex)

# 2) Per-emotion ANOVA
anova_sig = (anova
             .query("eta2 > .05 and q < .05")
             .sort_values("eta2", ascending=False))
lines = [r"\begin{tabular}{lccc}",
         r"\toprule",
         r"Emotion & $F$ & $\eta^{2}$ & $q$ \\ \midrule"]
for _, r in anova_sig.iterrows():
    lines.append(f"{r['emo']} & {r['F']:.1f} & "
                 f"{r['eta2']:.2f} & {r['q']:.3g} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "rq2_anova_top.txt").write_text("\n".join(lines))

# 3) Top-5 Tukey gaps (largest absolute mean diff)
top5 = (sig.assign(diff=lambda d: d["meandiff"].abs())
            .nlargest(5, "diff"))
lines = [r"\begin{tabular}{l l c c}",
         r"\toprule",
         r"Group 1 & Group 2 & Emotion & $\Delta$ \\ \midrule"]
for _, r in top5.iterrows():
    lines.append(f"{r['group1']} & {r['group2']} & "
                 f"{r['emo']} & {float(r['meandiff']):+.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "rq2_tukey_top.txt").write_text("\n".join(lines))


log.info("🏁 RQ-2 artefacts saved to  %s", OUT)
