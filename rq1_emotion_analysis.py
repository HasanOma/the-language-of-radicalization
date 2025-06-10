# rq1_emotion_analysis.py
"""
RQ-1  Dominant emotions by ideology
———————————————
* per-message normalisation
* ≥ MIN_SIZE filter (tiny buckets discarded)
* bootstrap 95 % CIs   (balanced, 2 000 draws)
* bar-plot by ideology   +   facets by top-N groups
* yearly trend PDF
"""

from pathlib import Path
import logging, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import bootstrap

DATA = Path("data/pysentimentio/revised-preprocess/5k/02_features.feather")
OUT  = Path("results-revised/rq1/80k"); OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("rq1")

df  = pd.read_feather(DATA)

# ───────── config ──────────────────────────────────────────────
KNOWN = ["fear","anger","anticipation","trust","surprise",
         "sadness","joy","disgust","positive","negative"]
EMO   = [e for e in KNOWN if e in df.columns]
if not EMO:
    raise RuntimeError("❌ emotion columns not found")

MIN_SIZE   = 200     
TOP_GROUPS = 10       

# ───────── filtering / normalising ─────────────────────────────
df = df[df["ideology"].map(df["ideology"].value_counts()) >= MIN_SIZE]
df[EMO] = df[EMO].div(df[EMO].sum(axis=1).replace(0, 1), axis=0)  

top = (df["forum"].value_counts().head(TOP_GROUPS).index)
df["forum_display"] = np.where(df["forum"].isin(top), df["forum"], "Other")

# ───────── weighted means ───────────────────────────
WEIGHTED = False
if WEIGHTED and "token_count" in df:
    df[EMO] = df[EMO].mul(df["token_count"], axis=0)

# ───────── facet bar-plot ──────────────────
agg2 = (df.groupby(["ideology", "forum_display"])[EMO]
          .mean().reset_index())
g = sns.catplot(data=agg2.melt(["ideology","forum_display"], EMO,
                               var_name="emotion", value_name="mean"),
                x="ideology", y="mean", hue="emotion",
                col="forum_display", col_wrap=3, sharey=False,
                kind="bar", height=3.5, aspect=1.2)
g.set_xticklabels(rotation=45, ha="right")
g.fig.subplots_adjust(top=.92)
g.fig.suptitle("Dominant emotions – by ideology (columns = groups)")
g.savefig(OUT/"rq1_emotion_by_group_and_ideology.png", dpi=300)

# ───────── bootstrap CIs ─────────────────────
boot = []
for (ideol, emo), sub in (
        df.melt("ideology", EMO, var_name="emotion", value_name="prob")
          .groupby(["ideology","emotion"])):
    ci = bootstrap((sub["prob"].values,), np.mean,
               confidence_level=.95,
               n_resamples=2000,
               method="percentile",    
               random_state=0)
    boot.append([ideol, emo, sub["prob"].mean(),
                 ci.confidence_interval.low, ci.confidence_interval.high])
boot = pd.DataFrame(boot, columns=["ideology","emotion","mean","lo","hi"])

plt.figure(figsize=(14,7))
ax = sns.barplot(data=boot, x="ideology", y="mean", hue="emotion",
                 errorbar=None)

_, hue_levels = ax.get_legend_handles_labels()

bar_containers = [c for c in ax.containers
                  if isinstance(c, matplotlib.container.BarContainer)]

ideology_levels = [t.get_text() for t in ax.get_xticklabels()]

ci_lookup = {(r.emotion, r.ideology): (r.lo, r.hi)
             for _, r in boot.iterrows()}

for container, emo in zip(bar_containers, hue_levels):
    for j, patch in enumerate(container):          
        ideol      = ideology_levels[j]
        lo, hi     = ci_lookup[(emo, ideol)]
        x          = patch.get_x() + patch.get_width() / 2
        yerr_upper = hi - patch.get_height()
        ax.errorbar(x,
                    patch.get_height(),
                    yerr=yerr_upper,
                    fmt="none",
                    ecolor="k", elinewidth=.8,
                    capsize=3, capthick=.8, zorder=10)
plt.title("Dominant emotions per ideology  (bootstrap 95 % CI)")
plt.ylabel("Average probability")
plt.xticks(rotation=35, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(OUT/"rq1_emotion_barplot.png", dpi=300)

# ───────── summary tables ───────────────────────────────────────
wide_means = boot.pivot(index="ideology", columns="emotion", values="mean")
wide_means.to_csv(OUT / "rq1_emotion_means.csv")

boot.to_csv(OUT / "rq1_emotion_means_ci.csv", index=False)

top_rows = (boot.loc[boot.groupby("ideology")["mean"].idxmax()]
               .sort_values("ideology"))

md_lines = ["| ideology | top emotion | mean | 95 % CI |",
            "|---------|-------------|------|---------|"]
for _, r in top_rows.iterrows():
    md_lines.append(f"| {r['ideology']} | {r['emotion']} | "
                    f"{r['mean']:.3f} | [{r['lo']:.3f}, {r['hi']:.3f}] |")
(OUT / "rq1_emotion_summary.md").write_text("\n".join(md_lines))


tex_lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"ideology & top emotion & mean & 95\% CI \\ \midrule"]
for _, r in top_rows.iterrows():
    tex_lines.append(f"{r['ideology']} & {r['emotion']} & "
                 f"{r['mean']:.3f} & "
                 f"$[{r['lo']:.3f}, {r['hi']:.3f}]$ \\\\")
tex_lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "rq1_emotion_summary.txt").write_text("\n".join(tex_lines))

# ───────── yearly trends ───────────────────────────────────
if "created_on" in df:
    df["year"] = pd.to_datetime(df["created_on"], errors="coerce").dt.year
    with PdfPages(OUT/"rq1_emotion_trends.pdf") as pdf:
        for ideol, sub in df.groupby("ideology"):
            yearly = sub.groupby("year")[EMO].mean().reset_index()
            plt.figure(figsize=(8,4))
            for emo in EMO:
                plt.plot(yearly["year"], yearly[emo],
                         marker="o", linewidth=1, label=emo)
            plt.title(f"{ideol} — emotion trends")
            plt.xlabel("Year"); plt.ylabel("Avg emotion prob.")
            plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", fontsize="small")
            plt.tight_layout(); pdf.savefig(); plt.close()
else:
    log.warning("No `created_on` column – yearly trend skipped")

log.info("🏁 RQ-1 artefacts saved to  %s", OUT)
