"""Figures for docs/judges/judge_robustness_followups.md (run after jev_decomposition.py analyze and
judge_selection_and_error.py).

    python scripts/judges/plot_judge_followups.py
Writes data/judges/filter_jev_vs_kimi/decomposition/fig_component_weights.png and
       data/judges/robustness/fig_per_layer_judges.png
"""
import csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEC = ROOT / "data" / "judges" / "filter_jev_vs_kimi" / "decomposition"
JR = ROOT / "data" / "judges" / "robustness"
KIMI, JEV, GLM = "#d08c00", "#2a6fdb", "#5a5a5a"
LABELS = {"evil_persona": "Evil persona / tone", "actionable_harm": "Actionable harm", "false_claims": "False claims as fact",
          "fictional_frame": "Requested fiction / role-play", "refusal": "Refusal", "hostile_to_user": "Hostile to user",
          "on_topic": "On-topic"}


def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=6, direction="out")


def fig_components():
    S = json.loads((DEC / "decomposition_summary.json").read_text(encoding="utf-8"))
    q = list(LABELS)[::-1]; y = np.arange(len(q))
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for off, key, col, lab in ((0.14, "kimi_score", KIMI, "Kimi K3 score"), (-0.14, "jev_ev_A", JEV, "Jev score")):
        c = S["ols_hc3"][key]["coef"]
        b = np.array([c[k]["b"] for k in q]); lo = np.array([c[k]["ci95"][0] for k in q]); hi = np.array([c[k]["ci95"][1] for k in q])
        ax.errorbar(b, y + off, xerr=[b - lo, hi - b], fmt="o", ms=3.2, color=col, lw=1, capsize=0,
                    label=f"{lab} (R² = {S['ols_hc3'][key]['r2']:.2f})")
    ax.axvline(0, color="#999", lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels([LABELS[k] for k in q], fontsize=7)
    ax.set_xlabel("Score points per unit of Jev yes-probability (OLS, 95% CI)", fontsize=7)
    ax.set_title("Kimi discounts requested fiction; both weight harm equally", loc="left", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower right"); style(ax)
    fig.text(0.01, -0.02, f"n = {S['n']} filter generations; components are Jev Nouls (one call per item).", fontsize=5.5, color="#555")
    fig.savefig(DEC / "fig_component_weights.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig_layers():
    rows = [r for r in csv.DictReader(open(JR / "per_layer.csv", encoding="utf-8")) if r["analysis"] == "differential"]
    S = json.loads((JR / "summary.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True, gridspec_kw={"wspace": 0.08})
    titles = {"response_mean_cosine": "Response-mean distance", "last_prompt_tok_cosine": "Last-prompt-token distance"}
    for ax, tag in zip(axes, titles):
        rr = [r for r in rows if r["activation"] == tag and r["rho_d_kimi"] not in ("", "nan")]
        L = np.array([int(r["layer"]) for r in rr])
        k = np.array([float(r["rho_d_kimi"]) for r in rr]); g = np.array([float(r["rho_d_glm"]) for r in rr])
        ax.plot(L, k, color=KIMI, lw=1.2, label="Kimi K3 outcome scores"); ax.plot(L, g, color=GLM, lw=1.2, label="GLM 5.3 outcome scores")
        ax.axvline(14, color="#bbb", lw=0.6, ls=":"); ax.axhline(0, color="#ddd", lw=0.6)
        de = S["differential_error"][tag]
        ax.set_title(f"{titles[tag]}\nlayer 14: Δρ = {de['rho_d_kimi_minus_rho_d_glm']:+.3f} "
                     f"[{de['rho_diff_ci95_paired_boot'][0]:+.3f}, {de['rho_diff_ci95_paired_boot'][1]:+.3f}]", loc="left", fontsize=7)
        ax.set_xlabel("Layer", fontsize=7); style(ax)
    axes[0].set_ylabel("Spearman ρ (distance, EM evil score)", fontsize=7)
    axes[1].legend(fontsize=6, frameon=False, loc="lower right")
    fig.suptitle("Kimi scores give a stronger distance–evil correlation than GLM at most layers\nAt the canonical layer 14 the gap is clear only for last-prompt-token distance", x=0.06, ha="left", y=1.12, fontsize=8)
    fig.text(0.06, -0.06, f"n = {S['differential_error']['response_mean_cosine']['n']} Kimi-verified prompts scored by both judges; "
             "EM model ChicagoHAI/qwen2.5-7b-em-medical-advice; dotted line = canonical layer 14.", fontsize=5.5, color="#555")
    fig.savefig(JR / "fig_per_layer_judges.png", dpi=300, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_components(); fig_layers(); print("ok")
