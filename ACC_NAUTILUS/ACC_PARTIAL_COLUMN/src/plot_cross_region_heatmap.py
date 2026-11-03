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
"""




# ------------------------------------------------------------
# Pretty names
# ------------------------------------------------------------

PRETTY = {
    "North_Pacific": "N.Pacif.",
    "South_Atlantic": "S.Atl.",
    "Tropical_Pacific": "Trop.Pacif.",
    "Pacific": "Pacif.",
    "South_Pacific": "S. Pacif.",
}


# ------------------------------------------------------------
# Heatmap
# ------------------------------------------------------------

def plot_heatmap(df, metric, outdir="plots/heatmaps"):

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

        vmin = 0
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
        f"Cross-region Generalization\n{metric}",
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

        row.iloc[i] = np.nan

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

            if j == best_transfer[i]:
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

    png = os.path.join(outdir, f"heatmap_{metric.lower()}.png")
    pdf = os.path.join(outdir, f"heatmap_{metric.lower()}.pdf")

    plt.savefig(png, dpi=600, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")

    plt.close()

    print("Saved", png)


# ------------------------------------------------------------
# Generalization ranking
# ------------------------------------------------------------

def generalization_score(df, outdir="plots/heatmaps"):

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
                     "cross_region_generalization.csv"),
        index=False,
    )

    print(scores)


# ------------------------------------------------------------
# Transfer loss
# ------------------------------------------------------------

def plot_transfer_loss(df, outdir="plots/heatmaps"):

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

    plt.title("Transfer Loss (Diagonal - Cross_region)")

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
                     "heatmap_transfer_loss.png"),
        dpi=600,
    )

    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    csv_file = "cross_region_metrics.csv"

    df = pd.read_csv(csv_file)

    for metric in [

        "SSIM",

        "R2",

        "RMSE_pct",

        "MAE_pct",

    ]:

        plot_heatmap(df, metric)

    generalization_score(df)

    plot_transfer_loss(df)

    print("\nFinished creating all cross-region figures.")
