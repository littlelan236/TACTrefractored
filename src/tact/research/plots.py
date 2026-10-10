from pathlib import Path

from tact.artifacts import read_json


def plot_bundle(bundle: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = read_json(bundle / "manifest.json")
    curves = manifest["curves"]
    sites = sorted(map(int, curves))
    fig, ax = plt.subplots(figsize=(9, 4))
    for split, style in [("axis-train", "--"), ("axis-val", "-")]:
        for axis, color in [("ot", "#c85935"), ("oa", "#277e9e")]:
            ax.plot(sites, [curves[str(i)][split][axis]["mean_auc"] for i in sites],
                    style, color=color, label=f"{split} {axis.upper()} / CAL")
    ax.set(xlabel="Site index (final index = final norm)", ylabel="Per-problem macro AUC", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(bundle / f"auc-by-site.{suffix}", dpi=180)
    plt.close(fig)
