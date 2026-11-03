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
            "lr": [],
            "val_rmse": [], "val_rmse_persistence": [],
            "val_skill": [], "val_skill_<region>": [],
            "train_loss_fixed_eval": [],
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

    # v3: training loss on fixed patches with the model in eval mode
    if len(history.get("train_loss_fixed_eval", [])) == len(epochs):
        ax.plot(epochs, history["train_loss_fixed_eval"], color="#4E79A7", linewidth=1.8,
                linestyle="--", marker="s", markersize=5, label="Train, fixed patches, eval mode")

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
    # VALIDATION SKILL (v3: replaces the training R² / SSIM, which are
    # ~1 for this target and say nothing)
    # ===========================================================

    ax = fig.add_subplot(gs[0, 1])

    if len(history.get("val_skill", [])) == len(epochs):
        ax.plot(epochs, history["val_skill"], marker="o", linewidth=2.5, markersize=7,
                color="black", label="all regions")
    for key in sorted(k for k in history if k.startswith("val_skill_")):
        if len(history[key]) == len(epochs):
            ax.plot(epochs, history[key], marker="o", linewidth=1.5, markersize=5,
                    label=key[len("val_skill_"):].replace("_", " "))
    ax.axhline(0, color="grey", lw=0.8)
    ax.grid(alpha=0.3)
    ax.set_title('Validation skill  1 - MSE/MSE("no change")')
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ===========================================================
    # VALIDATION RMSE: model vs "no change"
    # ===========================================================

    ax = fig.add_subplot(gs[0, 2])

    if len(history.get("val_rmse", [])) == len(epochs):
        ax.plot(epochs, history["val_rmse"], marker="o", linewidth=2, markersize=7, label="model")
        ax.plot(epochs, history["val_rmse_persistence"], linestyle="--", color="black",
                linewidth=1.5, label='"no change"')
    ax.grid(alpha=0.3)
    ax.set_title("Validation RMSE, full field, ocean cells")
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

    if history.get("val_skill"):
        sk = history["val_skill"]
        b = (best_epoch - 1) if best_epoch else int(np.nanargmax(sk))
        txt += f"Validation skill, best epoch: {sk[b]:+.4f}\n"
        txt += f"Validation skill, last epoch: {sk[-1]:+.4f}\n"

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
