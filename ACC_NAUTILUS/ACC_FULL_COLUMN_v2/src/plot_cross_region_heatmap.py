import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

"""
plots/
└── heatmaps/
    ├── heatmap_ssim.png
    ├── heatmap_ssim.pdf
    ├── heatmap_r2.png
    ├── heatmap_r2.pdf
    ├── heatmap_rmse.png
    ├── heatmap_rmse.pdf
    ├── heatmap_mae.png
    ├── heatmap_mae.pdf
    ├── heatmap_transfer_loss.png
    └── cross_region_generalization.csv

With a tag (the config's training.experiment_name, e.g.
"depthbudget_0-2000m") every file gets a "_<tag>" suffix —
heatmap_ssim_depthbudget_0-2000m.png, ... — so the runs of a sweep
don't overwrite each other's report.
"""


def _tag_suffix(tag):
    """Filename suffix for a run tag; empty when no tag is given."""
    return f"_{tag}" if tag else ""




# ------------------------------------------------------------
# Pretty names
# ------------------------------------------------------------

PRETTY = {
    "North_Pacific": "N.Pacif.",
    "South_Atlantic": "S.Atl.",
    "Tropical_Pacific": "Trop.Pacif.",
    "Pacific": "Pacif.",
    "South_Pacific": "S. Pacif.",
    "North_Atlantic": "N.Atl.",
    "Ross_Sea": "Ross",
}


# ------------------------------------------------------------
# Heatmap
# ------------------------------------------------------------

def plot_heatmap(df, metric, outdir="plots/heatmaps", tag=None):

    os.makedirs(outdir, exist_ok=True)

    matrix = df.pivot(
        index="model",
        columns="test",
        values=metric,
    )

    matrix = matrix.loc[df["model"].unique(),
                        df["test"].unique()]

    labels_x = [PRETTY.get(x, x) for x in matrix.columns]
    labels_y = [PRETTY.get(x, x) for x in matrix.index]

    # ---------------------------------------
    # color maps
    # ---------------------------------------

    if metric in ["SSIM", "R2"]:

        cmap = "copper"

        vmin = 0.5
        vmax = 1

        higher_is_better = True

    else:

        #cmap = "Greys_r"
        cmap = "RdYlGn_r"
        vmin = matrix.min().min()
        vmax = matrix.max().max()

        higher_is_better = False

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        matrix.values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    cbar = plt.colorbar(im, ax=ax)

    if metric == "SSIM":
        cbar.set_label("SSIM")

    elif metric == "R2":
        cbar.set_label(r"$R^2$")

    elif metric == "RMSE_pct":
        cbar.set_label("RMSE (%)")

    elif metric == "MAE_pct":
        cbar.set_label("MAE (%)")

    titles = {
        "SSIM": "Cross-region Generalization (SSIM)",
        "R2":   "Cross-region Generalization ($R^2$)",
        "RMSE_pct": "Cross-region Error (RMSE %)",
        "MAE_pct":  "Cross-region Error (MAE %)"
    }

    plt.title(
        titles[metric],
        fontsize=15,
        weight="bold"
    )



    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_xticklabels(labels_x, fontsize=12)

    ax.set_yticks(np.arange(len(labels_y)))
    ax.set_yticklabels(labels_y, fontsize=12)

    plt.xlabel("Test Region", fontsize=13)
    plt.ylabel("Training Region", fontsize=13)

    plt.title(
        f"Cross-region Generalization\n{metric}" + (f" — {tag}" if tag else ""),
        fontsize=15,
        weight="bold",
    )

    # ------------------------------------------------------
    # diagonal
    # ------------------------------------------------------

    for i in range(matrix.shape[0]):

        ax.add_patch(

            Rectangle(

                (i - 0.5, i - 0.5),

                1,

                1,

                fill=False,

                edgecolor="lime",

                linewidth=3,

            )

        )

    # ------------------------------------------------------
    # best transfer per row
    # ------------------------------------------------------

    best_transfer = {}

    for i in range(matrix.shape[0]):

        row = matrix.iloc[i].copy()

        row.iloc[i] = np.nan  # exclude comparing a model against itself

        if np.all(np.isnan(row.values)):
            # No valid "cross" value to highlight for this row — e.g. this
            # run only has one region (nothing to transfer to), or the
            # metric itself came out NaN for every other column (can
            # happen with a tiny/degenerate test set, e.g. a smoke test
            # with very few samples). Skip bolding for this row instead
            # of crashing the whole report.
            best_transfer[i] = None
            print(f"⚠ No valid transfer target for '{row.name}' ({metric}) — "
                  f"skipping best-transfer highlight for this row.")
            continue

        if higher_is_better:
            best_transfer[i] = np.nanargmax(row.values)
        else:
            best_transfer[i] = np.nanargmin(row.values)

    # ------------------------------------------------------
    # values
    # ------------------------------------------------------

    for i in range(matrix.shape[0]):

        for j in range(matrix.shape[1]):

            val = matrix.iloc[i, j]

            

            if metric in ["RMSE_pct", "MAE_pct"]:
                txt = f"{val:.1f}%"
            else:
                txt = f"{val:.3f}"

            


            norm = (val - vmin) / (vmax - vmin + 1e-12)

            textcolor = "white" if norm > 0.55 else "black"

            weight = "normal"

            if best_transfer[i] is not None and j == best_transfer[i]:
                weight = "bold"

            ax.text(

                j,

                i,

                txt,

                ha="center",

                va="center",

                fontsize=11,

                color=textcolor,

                weight=weight,

            )

    plt.tight_layout()

    png = os.path.join(outdir, f"heatmap_{metric.lower()}{_tag_suffix(tag)}.png")
    pdf = os.path.join(outdir, f"heatmap_{metric.lower()}{_tag_suffix(tag)}.pdf")

    plt.savefig(png, dpi=600, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")

    plt.close()

    print("Saved", png)


# ------------------------------------------------------------
# Generalization ranking
# ------------------------------------------------------------

def generalization_score(df, outdir="plots/heatmaps", tag=None):

    os.makedirs(outdir, exist_ok=True)

    scores = []

    for model in df["model"].unique():

        row = df[df.model == model]

        diag = row[row.test == model]

        off = row[row.test != model]

        scores.append({

            "Model": model,

            "Own SSIM": diag["SSIM"].mean(),

            "Cross SSIM": off["SSIM"].mean(),

            "Own R2": diag["R2"].mean(),

            "Cross R2": off["R2"].mean(),

            "Own RMSE": diag["RMSE_pct"].mean(),

            "Cross RMSE": off["RMSE_pct"].mean(),

            "Own MAE": diag["MAE_pct"].mean(),

            "Cross MAE": off["MAE_pct"].mean(),

        })

    scores = pd.DataFrame(scores)

    scores["Generalization Score"] = scores["Cross SSIM"]

    scores = scores.sort_values(
        "Generalization Score",
        ascending=False,
    )

    scores.to_csv(
        os.path.join(outdir,
                     f"cross_region_generalization{_tag_suffix(tag)}.csv"),
        index=False,
    )

    print(scores)


# ------------------------------------------------------------
# Transfer loss
# ------------------------------------------------------------

def plot_transfer_loss(df, outdir="plots/heatmaps", tag=None):

    os.makedirs(outdir, exist_ok=True)

    models = df.model.unique()
    tests = df.test.unique()

    loss = np.zeros((len(models), len(tests)))

    for i, model in enumerate(models):

        diag = df[
            (df.model == model) &
            (df.test == model)
        ]["SSIM"].values[0]

        for j, test in enumerate(tests):

            val = df[
                (df.model == model) &
                (df.test == test)
            ]["SSIM"].values[0]

            loss[i, j] = diag - val

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        loss,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=np.max(loss),
    )

    
    plt.colorbar(im, ax=ax, label="SSIM Loss")

    ax.set_xticks(range(len(tests)))
    ax.set_xticklabels([PRETTY.get(x, x) for x in tests])

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([PRETTY.get(x, x) for x in models])

    plt.title("Transfer Loss (Diagonal - Cross_region)" + (f"\n{tag}" if tag else ""))

    for i in range(loss.shape[0]):

        for j in range(loss.shape[1]):

            ax.text(
                j,
                i,
                f"{loss[i,j]:.3f}",
                ha="center",
                va="center",
                fontsize=10,
            )

    plt.tight_layout()

    plt.savefig(
        os.path.join(outdir,
                     f"heatmap_transfer_loss{_tag_suffix(tag)}.png"),
        dpi=600,
    )

    plt.close()



def build_cross_region_report(df, tag=None, outdir="plots/heatmaps"):

    # One folder per run tag under cross_region/, so the report files of
    # a sweep stay grouped; an untagged run goes to cross_region/ itself.
    outdir = os.path.join(outdir, "cross_region", tag or "")

    plot_heatmap(df, "SSIM", outdir=outdir, tag=tag)
    plot_heatmap(df, "R2", outdir=outdir, tag=tag)
    plot_heatmap(df, "RMSE_pct", outdir=outdir, tag=tag)
    plot_heatmap(df, "MAE_pct", outdir=outdir, tag=tag)

    generalization_score(df, outdir=outdir, tag=tag)

    plot_transfer_loss(df, outdir=outdir, tag=tag)

    print("\n✓ Cross-region report finished.")



# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None,
                        help="Run tag (training.experiment_name), e.g. depthbudget_0-2000m. "
                             "Reads cross_region_metrics_<tag>.csv and tags the outputs.")
    args = parser.parse_args()

    csv_file = f"plots/heatmaps/cross_region_metrics{_tag_suffix(args.tag)}.csv"

    df = pd.read_csv(csv_file)

    build_cross_region_report(df, tag=args.tag)

    print("\nFinished creating all cross-region figures.")
