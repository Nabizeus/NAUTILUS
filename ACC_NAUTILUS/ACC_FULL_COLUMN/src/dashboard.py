import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_training_dashboard(
    history,
    profiler,
    region_name,
    cfg,
    save_dir="artifacts/dashboard/training",
    best_epoch=None,
    training_minutes=None,
):
    """
    Creates one summary dashboard after training.

    Parameters
    ----------
    history : dict

        {
            "train_loss": [],
            "val_loss": [],
            "rmse": [],
            "mae": [],
            "r2": [],
            "ssim": [],
            "lr": [],
        }

    profiler : SimpleProfiler

    region_name : str

    cfg : Hydra config

    best_epoch : int

    training_minutes : float
    """

    os.makedirs(save_dir, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig = plt.figure(figsize=(18, 11))

    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.0]
    )

    # ===========================================================
    # LOSS
    # ===========================================================

    ax = fig.add_subplot(gs[0, 0])

    ax.plot(
        epochs,
        history["train_loss"],
        color="#4E79A7",
        linewidth=2.5,
        marker="o",
        markersize=8,
        markerfacecolor="#4E79A7",
        markeredgecolor="black",
        label="Train",
    )

    ax.plot(
        epochs,
        history["val_loss"],
        color="#F28E2B",
        linewidth=2.5,
        marker="o",
        markersize=8,
        markerfacecolor="#F28E2B",
        markeredgecolor="black",
        label="Validation",
    )

    ax.grid(alpha=0.3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")

    ax.legend()

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if best_epoch is not None:
        ax.axvline(best_epoch,
                   color="red",
                   linestyle="--",
                   alpha=0.4)

    # ===========================================================
    # METRICS
    # ===========================================================

    ax = fig.add_subplot(gs[0, 1])

    ax.plot(
        epochs,
        history["r2"],
        marker="o",
        linewidth=2,
        markersize=7,
        label="R²",
    )

    ax.plot(
        epochs,
        history["ssim"],
        marker="o",
        linewidth=2,
        markersize=7,
        label="SSIM",
    )

    ax.grid(alpha=0.3)

    ax.set_title("Validation Metrics")

    ax.set_xlabel("Epoch")

    ax.legend()

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ===========================================================
    # RMSE / MAE
    # ===========================================================

    ax = fig.add_subplot(gs[0, 2])

    ax.plot(
        epochs,
        history["rmse"],
        marker="o",
        linewidth=2,
        markersize=7,
        label="RMSE",
    )

    ax.plot(
        epochs,
        history["mae"],
        marker="o",
        linewidth=2,
        markersize=7,
        label="MAE",
    )

    ax.grid(alpha=0.3)

    ax.set_title("Validation Errors")

    ax.legend()

    ax.set_xlabel("Epoch")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ===========================================================
    # LEARNING RATE
    # ===========================================================

    ax = fig.add_subplot(gs[1, 0])

    if len(history["lr"]) > 0:

        ax.plot(
            epochs,
            history["lr"],
            color="purple",
            linewidth=2.5,
            marker="o",
            markersize=7,
        )

        ax.set_yscale("log")

    ax.grid(alpha=0.3)

    ax.set_title("Learning Rate")

    ax.set_xlabel("Epoch")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ===========================================================
    # PROFILER
    # ===========================================================

    ax = fig.add_subplot(gs[1, 1])

    names = []
    values = []

    for key, (total, count) in profiler.times.items():

        names.append(key)

        values.append(total)

    order = np.argsort(values)

    names = np.array(names)[order]

    values = np.array(values)[order]

    ax.barh(
        names,
        values,
    )

    ax.set_xlabel("Total seconds")

    ax.set_title("Profiler")

    for i, v in enumerate(values):
        ax.text(
            v,
            i,
            f"{v:.1f}s",
            va="center",
            fontsize=9,
        )

    # ===========================================================
    # SUMMARY
    # ===========================================================

    ax = fig.add_subplot(gs[1, 2])

    ax.axis("off")

    txt = ""

    txt += f"Region: {region_name}\n\n"

    txt += f"Model: {cfg.model.name}\n"

    txt += f"Epochs: {len(epochs)}\n"

    txt += f"Input steps: {cfg.data.window.input_steps}\n"

    txt += f"Prediction steps: {cfg.data.window.pred_steps}\n"

    txt += f"Depth levels: {cfg.data.vertical_levels}\n"

    #txt += f"Variables: {cfg.data.n_channels}\n\n"

    txt += f"Best epoch: {best_epoch if best_epoch is not None else '-'}\n"

    txt += f"Final Train Loss: {history['train_loss'][-1]:.6f}\n"

    txt += f"Final Val Loss: {history['val_loss'][-1]:.6f}\n"

    txt += f"Final RMSE: {history['rmse'][-1]:.4f}\n"

    txt += f"Final MAE: {history['mae'][-1]:.4f}\n"

    txt += f"Final R²: {history['r2'][-1]:.4f}\n"

    txt += f"Final SSIM: {history['ssim'][-1]:.4f}\n"

    if training_minutes is not None:

        txt += f"\nTraining Time:\n{training_minutes:.1f} min"

    ax.text(
        0,
        1,
        txt,
        fontsize=11,
        va="top",
        family="monospace",
    )

    plt.suptitle(
        f"NAUTILUS Training Dashboard — {region_name}",
        fontsize=18,
        fontweight="bold",
    )

    plt.tight_layout()

    outfile = os.path.join(
        save_dir,
        f"{region_name}_training_dashboard.png",
    )

    plt.savefig(
        outfile,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"📊 Saved dashboard -> {outfile}")
