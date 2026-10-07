import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C_OLD, C_NEW = "#d1495b", "#2e86ab"

# ---- B
bb, ba = json.load(open("b_before.json")), json.load(open("b_after.json"))
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1, 3.2]})
for a, d, t in ((ax[0], bb, "upstream main"), (ax[1], ba, "this PR")):
    s = np.array(d["per_group|0.0"]["scales"])
    vmax = np.abs(s).max()
    im = a.imshow(s, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    a.set_title(f"E4M3 group scales\n{t}"); a.set_xticks([0, 1], ["g0", "g1"]); a.set_ylabel("row")
    a.axhline(28.5, color="k", lw=0.8, ls="--")
    for r in range(29, 32):
        for g in range(2):
            a.text(g, r, f"{s[r, g]:.2f}", ha="center", va="center", fontsize=6)
fig.colorbar(im, ax=ax[:2], shrink=0.8, pad=0.02)
keys = list(bb); x = np.arange(len(keys)); w = 0.2
lab = [k.replace("|", "\npad=").replace("10000.0", "1e4").replace("0.0", "0") for k in keys]
for i, (d, c, n) in enumerate(((bb, C_OLD, "main"), (ba, C_NEW, "this PR"))):
    ax[2].bar(x + (i - 0.5) * w * 2 - w / 2, [d[k]["leaked_scales"] for k in keys], w, color=c, label=f"non-zero scales in fully masked groups ({n})")
    ax[2].bar(x + (i - 0.5) * w * 2 + w / 2, [d[k]["valid_changed"] for k in keys], w, color=c, alpha=0.45, hatch="//", label=f"valid outputs changed by padding ({n})")
ax[2].set_yscale("symlog", linthresh=1); ax[2].set_xticks(x, lab, fontsize=7); ax[2].set_ylabel("count (symlog)")
ax[2].set_title("Masked-lane leakage, 32×32 tile, valid region [:29, :23]"); ax[2].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 0.92))
for k_i, k in enumerate(keys):
    if bb[k]["valid_changed"]: ax[2].annotate(f"{bb[k]['valid_changed']}/667", (k_i - 0.05, bb[k]["valid_changed"]), fontsize=7, ha="center", va="bottom")
ax[2].text(0.99, 0.80, "this PR: all counts 0", transform=ax[2].transAxes, ha="right", va="top", color=C_NEW, fontsize=9)
fig.suptitle("NVFP4 fake-quant on GB10: masked padding before / after", fontsize=12, y=1.06)
fig.savefig("fig-b-nvfp4-mask.png", dpi=150, bbox_inches="tight")

# ---- A
ab, aa = json.load(open("a_before.json")), json.load(open("a_after.json"))
shapes = ["2112/2112", "2112/2080", "8192/8192"]; names = ["O", "dQ", "dK", "dV", "STE", "M"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2.2, 1]})
x = np.arange(len(names)); w = 0.26
for i, sh in enumerate(shapes):
    v = [ab[f"{sh}|{n}"]["rel_l2"] for n in names]
    ax[0].bar(x + (i - 1) * w, v, w, label=f"Lq/Lk = {sh}", color=plt.cm.Reds(0.45 + 0.2 * i))
ax[0].set_yscale("log"); ax[0].set_xticks(x, names); ax[0].set_ylabel("relative L2, joined vs split")
ax[0].set_title("upstream main on GB10: FASTVIDEO_ATTN_QAT_SM120_JOIN_QAT_PV=1 vs 0"); ax[0].legend(fontsize=8)
tab = [[(lambda m: "<0.1%" if 0 < m < 1e-3 else f"{m*100:.1f}%")(ab[f'{sh}|{n}']['mismatch']) for n in names] for sh in shapes] + \
      [[("0" if aa[f'{sh}|{n}']['rel_l2'] == 0 and aa[f'{sh}|{n}']['mismatch'] == 0 else "≠0") for n in names] for sh in shapes]
ax[1].axis("off")
t = ax[1].table(cellText=tab, rowLabels=[f"main {s}" for s in shapes] + [f"PR {s}" for s in shapes], colLabels=names, loc="center", cellLoc="center")
t.scale(1, 1.5); t.auto_set_font_size(False); t.set_fontsize(8)
for r in range(1, 7):
    for c in range(len(names)):
        t[r, c].set_facecolor("#fbe3e6" if r <= 3 else "#dff0f7")
ax[1].set_title("elements differing, joined vs split\n(this PR: bitwise equal, rel L2 = 0)", fontsize=9)
fig.suptitle("Attn-QAT P@V routing on GB10 (SM121), B=1, H=3, D=128, seed 0", fontsize=12, y=1.06)
fig.savefig("fig-a-split-pv.png", dpi=150, bbox_inches="tight")
