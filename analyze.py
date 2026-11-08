"""Usage: python analysis/analyze.py data/responses.csv   (run from the site folder)"""
import sys, json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
rng = np.random.default_rng(42)
raw = pd.read_csv(sys.argv[1]); log = []
n0 = len(raw); raw = raw.drop_duplicates(); log.append(f"Rows in export: {n0}; exact duplicates removed: {n0-len(raw)}")
c = raw.columns
eng_cols, nwatch = c[1:4], c[4]
raw = raw.reset_index(drop=True); raw["pid"] = ["R%02d" % (i+1) for i in range(len(raw))]
raw["engagement"] = raw[eng_cols].mean(axis=1)
CUT = 4.0
raw["group"] = np.where(raw.engagement >= CUT, "Highly engaged", "Casual / moderate")
rows = []
for k in range(1, 6):
    e, i, fv = [x for x in c if x.startswith(f"Match {k}:")]
    for _, r in raw.iterrows():
        if pd.notna(r[e]):
            if not 1 <= r[e] <= 10: log.append(f"{r.pid} match {k}: enjoyment {r[e]} outside 1-10, dropped"); continue
            rows.append(dict(pid=r.pid, order=k, enj=r[e], imp=r[i], fav=int(r[fv] == "Yes"), group=r.group, eng=r.engagement))
L = pd.DataFrame(rows)
# consistency check: stated count vs rated matches
cnt = L.groupby("pid").size(); raw[nwatch] = raw[nwatch].astype(str).str.extract(r"(\d+)")[0].astype(float); chk = raw.set_index("pid")[nwatch]
for p, v in chk.items():
    got = int(cnt.get(p, 0))
    if (v >= 1 and got != min(v, 5)) or (v == 0 and got): log.append(f"{p}: reported {v:.0f} matches watched but {got} rated (flag, kept)")
log.append(f"Respondents with >=1 rating: {L.pid.nunique()}; ratings: {len(L)}")
raw.drop(columns=["Timestamp"]).to_csv("responses_anonymised.csv", index=False)
L.to_csv("long_format.csv", index=False)

R = dict(demo=False, high_engagement_cutoff=CUT, n_respondents=len(raw), n_with_ratings=int(L.pid.nunique()))
def ci(x):
    x = np.asarray(x, float); m = x.mean()
    if len(x) < 2: return m, np.nan, np.nan
    h = stats.t.ppf(.975, len(x)-1) * x.std(ddof=1)/np.sqrt(len(x)); return m, m-h, m+h
# slopes
sl = []
for p, g in L.groupby("pid"):
    if g.order.nunique() >= 2: sl.append(dict(pid=p, slope=np.polyfit(g.order, g.enj, 1)[0], group=g.group.iloc[0], n=len(g)))
S = pd.DataFrame(sl); R["n_slope"] = len(S)
m, lo, hi = ci(S.slope)
R["H1"] = dict(mean_slope=m, lo=lo, hi=hi, p=stats.ttest_1samp(S.slope, 0).pvalue, p_wilcoxon=stats.wilcoxon(S.slope).pvalue, share_negative=(S.slope < 0).mean())
# exact sign-flip permutation test (valid at tiny n)
obs = S.slope.mean(); signs = rng.choice([-1, 1], (20000, len(S)))
R["H1"]["p_perm"] = float((np.abs((signs*S.slope.values).mean(1)) >= abs(obs)-1e-12).mean())
# H2
G = {g: d.slope.values for g, d in S.groupby("group")}
R["slope_by_group"] = {g: dict(mean=float(v.mean()), n=len(v), lo=ci(v)[1], hi=ci(v)[2]) for g, v in G.items()}
if all(len(G.get(g, [])) >= 2 for g in ["Highly engaged", "Casual / moderate"]):
    a, b = G["Highly engaged"], G["Casual / moderate"]
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(5000)]
    R["H2"] = dict(diff=a.mean()-b.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), p=stats.ttest_ind(a, b, equal_var=False).pvalue,
                   p_mannwhitney=stats.mannwhitneyu(a, b).pvalue)
# fixed effects with cluster-robust SE
F = L.copy()
for v in ["enj", "order", "imp", "fav"]: F[v+"_d"] = F[v] - F.groupby("pid")[v].transform("mean")
F = F[F.groupby("pid").pid.transform("size") >= 2]
X = F[["order_d", "imp_d", "fav_d"]].values; y = F.enj_d.values
bh = np.linalg.lstsq(X, y, rcond=None)[0]; u = y - X @ bh; XtXi = np.linalg.pinv(X.T@X)
meat = sum(np.outer(X[idx].T@u[idx], X[idx].T@u[idx]) for idx in [np.where(F.pid.values == p)[0] for p in F.pid.unique()])
G_, N, K = F.pid.nunique(), len(F), 3
V = XtXi@meat@XtXi * (G_/(G_-1))*((N-1)/(N-K-G_)); se = np.sqrt(np.diag(V)); t = stats.t.ppf(.975, G_-1)
R["fe_controls"] = {k: dict(coef=bh[i], lo=bh[i]-t*se[i], hi=bh[i]+t*se[i], p=2*stats.t.sf(abs(bh[i]/se[i]), G_-1)) for i, k in enumerate(["order", "imp", "fav"])}
R["fe_controls"].update(_n_people=int(G_), _n_ratings=int(N))
# per-position means
pos = L.groupby("order").enj.agg(["mean", "count"]); R["by_position"] = {int(k): dict(mean=r["mean"], n=int(r["count"])) for k, r in pos.iterrows()}
# chart 1
fig, ax = plt.subplots(figsize=(7, 4))
for g, col in [("Highly engaged", "#1f6f4a"), ("Casual / moderate", "#b45f06")]:
    pts = [(k, *ci(d.enj)) for k, d in L[L.group == g].groupby("order") if len(d) >= 1]
    ax.errorbar([p[0]+(.05 if g[0] == "H" else -.05) for p in pts], [p[1] for p in pts], yerr=[[p[1]-p[2] if p[2] == p[2] else 0 for p in pts], [p[3]-p[1] if p[3] == p[3] else 0 for p in pts]], marker="o", capsize=3, label=g, color=col)
ax.set(xlabel="Match order", ylabel="Mean enjoyment (1-10)", xticks=range(1, 6), ylim=(1, 10.5)); ax.legend(); fig.tight_layout(); fig.savefig("charts/1_enjoyment_by_order.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(S.slope, bins=10, color="#1f6f4a"); ax.axvline(0, color="k", ls="--"); ax.set(xlabel="Slope (enjoyment change per extra match)", ylabel="People"); fig.tight_layout(); fig.savefig("charts/2_slope_distribution.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(6, 4)); gs = list(R["slope_by_group"])
for i, g in enumerate(gs):
    d = R["slope_by_group"][g]; ax.errorbar(i, d["mean"], yerr=[[d["mean"]-d["lo"] if d["lo"] == d["lo"] else 0], [d["hi"]-d["mean"] if d["hi"] == d["hi"] else 0]], fmt="o", capsize=5)
    ax.scatter([i]*len(G[g]), G[g], alpha=.35, color="gray")
ax.axhline(0, color="k", ls="--"); ax.set(xticks=range(len(gs)), xticklabels=[f"{g}\n(n={R['slope_by_group'][g]['n']})" for g in gs], ylabel="Mean slope"); fig.tight_layout(); fig.savefig("charts/3_slope_by_engagement.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(6, 4)); vc = raw[nwatch].value_counts().sort_index(); ax.bar([("5+" if i>=5 else str(int(i))) for i in vc.index], vc.values, color="#1f6f4a"); ax.set(xlabel="Matches watched in last 7 days", ylabel="Respondents"); fig.tight_layout(); fig.savefig("charts/4_viewing_frequency.png", dpi=150); plt.close()
json.dump(R, open("/tmp/r.json", "w"), default=float)
open("results.js", "w").write("window.RESULTS = " + json.dumps(R, default=float, indent=1) + ";")
open("exclusion_log.txt", "w").write("\n".join(log))
print("\n".join(log)); print(json.dumps(R, default=float, indent=1))
