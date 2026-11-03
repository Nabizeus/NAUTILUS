"""
===========================================================
 NAUTILUS Prediction Dashboard
===========================================================

Prediction dashboard for evaluation on the test set.

Shows

• RMSE / MAE evolution
• SSIM / R² evolution
• Mass conservation
• TRUE / PRED / ERROR
• QQ plot
• Histogram
• Profiler
• Summary

Author:
NAUTILUS
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from sklearn.metrics import r2_score


class NautilusPredictionDashboard:

    def __init__(
        self,
        region_name,
        depth_level,
        profiler,
        output_dir="artifacts",
        dpi=180,
    ):

        self.region = region_name
        self.depth = depth_level
        self.output_dir = output_dir
        self.profiler = profiler
        self.dpi = dpi

        os.makedirs(output_dir, exist_ok=True)

        # ---------- colors ----------

        self.c_train = "#5DA5DA"
        self.c_val = "#F17CB0"

        self.c_rmse = "#4E79A7"
        self.c_mae = "#F28E2B"

        self.c_ssim = "#59A14F"
        self.c_r2 = "#E15759"

        self.c_mass_true = "#4E79A7"
        self.c_mass_pred = "#59A14F"

        self.error_cmap = "RdBu_r"
        self.abs_error_cmap = "inferno"

        plt.rcParams.update({

            "font.size":11,

            "axes.titlesize":13,

            "axes.labelsize":11,

            "legend.fontsize":10,

            "xtick.labelsize":10,

            "ytick.labelsize":10,

        })



    def _add_similarity_box(self, ax, metrics):

        txt = (
            f"SSIM  {metrics['ssim']:.3f}\n"
            f"R²    {metrics['r2']:.3f}"
        )

        ax.text(
            0.03,
            0.05,
            txt,

            transform=ax.transAxes,

            fontsize=6.5,

            family="monospace",

            bbox=dict(
                boxstyle="round,pad=0.18",
                fc="white",
                ec="forestgreen",
                lw=0.8,
                alpha=0.70,
            ),
        )




    def _add_error_box(self, ax, metrics, true_img):

        mean_true = np.nanmean(true_img)

        rmse_pct = 100 * metrics["rmse"] / max(mean_true,1e-12)
        mae_pct  = 100 * metrics["mae"]  / max(mean_true,1e-12)

        txt = (
            f"RMSE {metrics['rmse']:.2e}\n"
            f"     ({rmse_pct:.1f}%)\n"
            f"MAE  {metrics['mae']:.2e}\n"
            f"     ({mae_pct:.1f}%)"
        )

        ax.text(
            0.03,
            0.05,
            txt,

            transform=ax.transAxes,

            fontsize=6.5,

            family="monospace",

            bbox=dict(
                boxstyle="round,pad=0.18",
                fc="white",
                ec="firebrick",
                lw=0.8,
                alpha=0.70,
            ),
        )



    def _add_timestep_badge(self, ax, step):

        ax.text(

            0.97,
            0.97,

            f"t={step}",

            ha="right",
            va="top",

            transform=ax.transAxes,

            fontsize=6.5,

            bbox=dict(

                boxstyle="round,pad=0.12",

                fc="#EFEFEF",

                ec="gray",

                lw=.5,

                alpha=.9,

            ),

        )




    def _tracer_title(
        self,
        title,
        metrics,
    ):

        return (
            f"{title}\n"
            f"SSIM = {metrics['ssim']:.3f}    "
            f"R² = {metrics['r2']:.3f}"
        )

    def _error_title(
        self,
        title,
        metrics,
        true_img,
    ):

        mean_true = np.nanmean(true_img)

        rmse_pct = 100 * metrics["rmse"] / max(mean_true,1e-12)
        mae_pct  = 100 * metrics["mae"]  / max(mean_true,1e-12)

        return (
            f"{title}\n"
            f"RMSE = {metrics['rmse']:.2e} ({rmse_pct:.2f}%)\n"
            f"MAE  = {metrics['mae']:.2e} ({mae_pct:.2f}%)"
        )   



    def _add_metrics_box(
        self,
        ax,
        metrics,
        true_img,
    ):

        mean_true = np.nanmean(true_img)

        rmse_pct = 100 * metrics["rmse"] / max(mean_true, 1e-12)
        mae_pct  = 100 * metrics["mae"]  / max(mean_true, 1e-12)

        txt = (
            f"RMSE : {metrics['rmse']:.2e}\n"
            f"        ({rmse_pct:.2f}%)\n"
            f"MAE  : {metrics['mae']:.2e}\n"
            f"        ({mae_pct:.2f}%)\n"
            f"SSIM : {metrics['ssim']:.3f}\n"
            f"R²   : {metrics['r2']:.3f}"
        )

        ax.text(

            0.02,
            0.02,
            txt,

            transform=ax.transAxes,

            fontsize=9,

            family="monospace",

            bbox=dict(

                facecolor="white",

                alpha=.90,

                edgecolor="gray",

            ),

        )


 
    def _format_time(self, seconds):
        """
        Human-readable elapsed time.

        Examples
        --------
        3.28        -> 3.28 s
        84.5        -> 1 min 24 s (84.5 s)
        3720        -> 1 h 2 min (3720 s)
        """

        if seconds < 60:
            return f"{seconds:.2f} s"

        minutes = int(seconds // 60)
        secs = seconds % 60

        if minutes < 60:
            return (
                f"{minutes:d} min "
                f"{secs:02.0f} s "
                f"({seconds:.1f} s)"
            )

        hours = int(minutes // 60)
        mins = minutes % 60

        return (
            f"{hours:d} h "
            f"{mins:02d} min "
            f"({seconds:.1f} s)"
        )




    def _human_params(self, n):
        """
        Pretty formatting of parameter counts.
        """

        if n >= 1e9:
            return f"{n/1e9:.2f} B"

        elif n >= 1e6:
            return f"{n/1e6:.2f} M"

        elif n >= 1e3:
            return f"{n/1e3:.1f} K"

        return f"{n:,}"




    def _panel_title(self, ax, title):

        ax.set_title(
            title,
            fontsize=13,
            fontweight="bold",
        )

    def _imshow(
        self,
        ax,
        img,
        title,
        cmap="viridis",
        vmin=None,
        vmax=None,
    ):

        im = ax.imshow(
            img,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(title)

        ax.set_xticks([])

        ax.set_yticks([])

        plt.colorbar(
            im,
            ax=ax,
            shrink=0.8,
        )


    def _plot_profiler(self, ax, profiler):

        names = []
        values = []

        for k, (t, c) in profiler.times.items():
            names.append(k)
            values.append(t)

        names = np.array(names)
        values = np.array(values)

        order = np.argsort(values)[::-1]

        names = names[order]
        values = values[order]

        bars = ax.barh(
            names,
            values,
            color="#4E79A7",
        )

        ax.invert_yaxis()

        ax.grid(alpha=.3)

        ax.set_xlabel("Seconds")

        ax.set_title("Profiler")

        xmax = values.max()

        for bar, sec in zip(bars, values):

            ax.text(
                sec + 0.01*xmax,
                bar.get_y()+bar.get_height()/2,
                self._format_time(sec),
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        # Highlight forward pass

        if "predict_forward" in profiler.times:

            total, count = profiler.times["predict_forward"]


            avg = total / max(count, 1)

            ax.text(
                0.98,
                0.02,
                f"Forward/timestep = {avg:.3f} s",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=10,
                color="red",
                fontweight="bold",
                bbox=dict(
                    fc="white",
                    ec="red",
                    alpha=.85,
                ),
            )

    def _plot_forecast_summary(
        self,
        ax,
        history,
        inference_time,
        model_info,
        profiler,
    ):

        forward_total = 0.0
        forward_step = 0.0


        if "predict_forward" in profiler.times:

            total, count = profiler.times["predict_forward"]

            forward_total = total

            forward_step = total / max(count, 1)


            if forward_step > 0:
                steps_per_second = 1.0 / forward_step
            else:
                steps_per_second = 0.0

    
        
        txt=""

        txt+=f"Region : {self.region}\n"

        txt+=f"Depth : {self.depth:.1f} m\n\n"

        txt+=f"Forecasts : {len(history['rmse'])}\n\n"

        txt+=f"RMSE : {np.mean(history['rmse']):.2e}\n"

        txt+=f"MAE : {np.mean(history['mae']):.2e}\n"

        txt+=f"SSIM : {np.mean(history['ssim']):.4f}\n"

        txt+=f"R² : {np.mean(history['r2']):.4f}\n\n"
    
        txt += (f"Mass balance : "f"{np.mean(np.abs(history['mass_percent'])):.1f}%\n\n"
        )
    
        txt += f"Speed : {steps_per_second:.1f} steps/s\n"

        txt += f"Per step : {forward_step:.3f} s\n"

        txt += f"Total : {self._format_time(forward_total)}\n\n"        
       

        ax.text(

            0.02,

            0.98,

            txt,

            va="top",

            fontsize=11,

            family="monospace",

        )

        ax.axis("off")


    def _plot_model_summary(
        self,
        ax,
        history,
        inference_time,
        model_info,
        profiler,
    ):


        info = model_info



        txt=""
        txt += (
            "MODEL\n"
            "----------------------\n"
            f"{info['name']}\n\n"
            f"Parameters : {self._human_params(info['parameters'])}\n"
            f"            ({info['parameters']:,})\n"
            f"Trainable  : {self._human_params(info['trainable'])}\n"
            f"Model size : {info['size_mb']:.1f} MB\n\n"
            f"Input      : {info['input_steps']} timesteps\n"
            f"Output     : {info['prediction_steps']} timestep\n"
            f"Depths     : {info['depths']}\n"
            f"Channels = Var x Depths  : {info['channels']}\n"
        )
 
        ax.text(

            0.02,

            0.98,

            txt,

            va="top",

            fontsize=11,

            family="monospace",

        )

        ax.axis("off")

    # ==========================================================
    # Metric evolution
    # ==========================================================

    def _plot_rmse_mae(self, ax, history):

        x = np.arange(1, len(history["rmse"]) + 1)

        ax.plot(
            x,
            history["rmse"],
            "-o",
            lw=2.5,
            ms=6,
            color=self.c_rmse,
            label="RMSE",
        )

        ax.plot(
            x,
            history["mae"],
            "-o",
            lw=2.5,
            ms=6,
            color=self.c_mae,
            label="MAE",
        )

        ax.set_title("Prediction Error")

        ax.set_xlabel("Prediction timestep")

        ax.set_ylabel("Error")

        ax.grid(alpha=.3)

        ax.legend()

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


    # ==========================================================
    # SSIM + R²
    # ==========================================================

    def _plot_ssim_r2(self, ax, history):

        x = np.arange(1, len(history["ssim"]) + 1)

        ax.plot(
            x,
            history["ssim"],
            "-o",
            lw=2.5,
            ms=6,
            color=self.c_ssim,
            label="SSIM",
        )

        ax.plot(
            x,
            history["r2"],
            "-o",
            lw=2.5,
            ms=6,
            color=self.c_r2,
            label="R²",
        )

        ax.set_title("Prediction Quality")

        ax.set_xlabel("Prediction timestep")

        ax.grid(alpha=.3)

        ax.legend()

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


    # ==========================================================
    # Mass conservation
    # ==========================================================

    def _plot_mass(self, ax, history):

        x = np.arange(1, len(history["mass_true"]) + 1)

        ax.plot(
            x,
            history["mass_true"],
            "-o",
            lw=2,
            ms=5,
            color=self.c_mass_true,
            label="True",
        )

        ax.plot(
            x,
            history["mass_pred"],
            "-o",
            lw=2,
            ms=5,
            color=self.c_mass_pred,
            label="Prediction",
        )

        ax.set_title("Tracer Mass")

        ax.set_xlabel("Prediction timestep")

        ax.set_ylabel("Integrated tracer")

        ax.grid(alpha=.3)

        ax.legend()

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))



    # ==========================================================
    # Mass error
    # ==========================================================

    def _plot_mass_error(self, ax, history):

        x = np.arange(
            1,
            len(history["mass_percent"])+1
        )

        ax.plot(

            x,

            history["mass_percent"],

            "-o",

            color="#D62728",

            lw=2.5,

            ms=6,

        )

        ax.axhline(
            0,
            color="black",
            ls="--",
        )

        mean_err = np.mean(history["mass_percent"])

        ax.set_title(
            f"Mass Balance Error\nMean = {mean_err:.1f}%"
        )

        ax.set_xlabel(
            "Prediction timestep"
        )

        ax.set_ylabel(
            "Difference (%)"
        )

        ax.grid(alpha=.3)

        ax.xaxis.set_major_locator(
            MaxNLocator(integer=True)
        )

    # ==========================================================
    # Temporal error evolution
    # ==========================================================

    def _plot_temporal_error(self, ax, history):

        x = np.arange(1, len(history["mean_error"]) + 1)

        ax.plot(
            x,
            history["mean_error"],
            "-o",
            color="#E15759",
            lw=2,
            ms=5,
            label="Mean",
        )

        ax.plot(
            x,
            history["std_error"],
            "-o",
            color="#4E79A7",
            lw=2,
            ms=5,
            label="Std",
        )

        ax.axhline(
            0,
            color="black",
            ls="--",
        )

        ax.grid(alpha=.3)

        ax.legend()

        ax.set_title("Error Evolution")

        ax.set_xlabel("Prediction timestep")

        ax.set_ylabel("Prediction Error")

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))




    # ==========================================================
    # Helper for shared color limits
    # ==========================================================

    def _shared_limits(self, true_img, pred_img):

        vmin = np.nanmin(
            [np.nanmin(true_img), np.nanmin(pred_img)]
        )

        vmax = np.nanmax(
            [np.nanmax(true_img), np.nanmax(pred_img)]
        )

        return vmin, vmax



    # ==========================================================
    # TRUE FIELD
    # ==========================================================
    def _plot_true(self, ax, true_img, units="mol m$^{-3}$", mld_distance=None):


        

        self._panel_title(ax, "TRUE")


        im = ax.imshow(
            true_img,
            origin="lower",
            cmap="viridis",
            interpolation="nearest"
        )


        if mld_distance is not None:
            # MLD is exactly at the selected depth where distance = 0
            mld_mask = np.ma.masked_invalid(mld_distance)

            ax.contour(
                mld_mask,
                levels=[0.0],
                colors="white",
                linewidths=2.5,
            )


        cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02, shrink=0.90)
        cb.set_label(units)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")


    # ==========================================================
    # PREDICTION
    # ==========================================================
    def _plot_prediction(self, ax, pred_img,
                         vmin,
                         vmax,
                         units="mol m$^{-3}$",mld_distance=None):

       self._panel_title(ax, "PREDICTION")

       im = ax.imshow(
            pred_img,
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
       )

       if mld_distance is not None:
            mld_mask = np.ma.masked_invalid(mld_distance)

            ax.contour(
                mld_mask,
                levels=[0.0],
                colors="white",
                linewidths=1.5,
            )

       cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
       cb.set_label(units)

       ax.set_xlabel("X")
       ax.set_ylabel("Y")



    # ==========================================================
    # SIGNED ERROR
    # ==========================================================
    def _plot_error(self, ax, error_img):

        vmax = np.nanmax(np.abs(error_img))

        self._panel_title(ax, "ERROR")

        im = ax.imshow(
            error_img,
            origin="lower",
            cmap="RdBu_r",
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Prediction - Truth")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")


    # ==========================================================
    # ABSOLUTE ERROR
    # ==========================================================
    def _plot_abs_error(self, ax, error_img):

        abs_error = np.abs(error_img)

        self._panel_title(ax, "ABSOLUTE ERROR")


        im = ax.imshow(
            abs_error,
            origin="lower",
            cmap="inferno",
            interpolation="nearest",
        )

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("|Prediction - Truth|")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")


    def _plot_mld_distance(self, ax, mld_distance, title="MLD DISTANCE"):
        self._panel_title(ax, title)

        im = ax.imshow(
            mld_distance,
            origin="lower",
            cmap="plasma",
            interpolation="nearest",
        )

        cb = plt.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

        cb.set_label("Distance to MLD [m]")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")






    # ==========================================================
    # QQ PLOT
    # ==========================================================
    def _plot_qq(self, ax, true_values, pred_values):

        self._panel_title(ax, "Q-Q Plot")

        t = np.sort(true_values.flatten())
        p = np.sort(pred_values.flatten())

        n = min(len(t), len(p))

        t = t[:n]
        p = p[:n]

        ax.scatter(
            t,
            p,
            s=8,
            alpha=.5,
            color="#4E79A7"
        )

        mn = min(t.min(), p.min())
        mx = max(t.max(), p.max())

        ax.plot(
            [mn, mx],
            [mn, mx],
            "r--",
            lw=2,
        )

        try:
            r2 = r2_score(t, p)
            txt = f"$R^2$ = {r2:.3f}"
        except Exception:
            txt = ""

        ax.text(
            0.05,
            0.95,
            txt,
            transform=ax.transAxes,
            va="top",
            fontsize=11,
        )

        ax.set_xlabel("True")

        ax.set_ylabel("Prediction")

        ax.grid(alpha=.3)



    # ==========================================================
    # ERROR HISTOGRAM
    # ==========================================================
    def _plot_histogram(self, ax, error_img):

        self._panel_title(ax, "Prediction Error Histogram")

        err = error_img.flatten()

        err = err[np.isfinite(err)]

        ax.hist(
            err,
            bins=60,
            color="#4E79A7",
            edgecolor="black",
            alpha=.75,
        )

        ax.axvline(
            0,
            color="red",
            ls="--",
            lw=2,
        )

        mean = np.mean(err)
        std = np.std(err)

        ax.text(
            0.97,
            0.95,
            f"μ = {mean:.3e}\nσ = {std:.3e}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc="white",
                ec="gray",
                lw=0.5,
                alpha=.65,
            ),
        )

        ax.grid(alpha=.3)

        ax.set_xlabel("Prediction - Truth")

        ax.set_ylabel("Pixels")



    # ==========================================================
    # COMPLETE PREDICTION DASHBOARD
    # ==========================================================
    def plot_prediction_dashboard(
        self,
        history,
        profiler,
        true_first,
        pred_first,
        error_first,
        mld_distance_first,

        true_last,
        pred_last,
        error_last,
        mld_distance_last,

        metrics_first,
        metrics_last,

        qq_true,
        qq_pred,

        inference_time,
        model_info,

        save_name="Prediction_Dashboard.png",
    ):

        fig = plt.figure(
            figsize=(22,16),
            constrained_layout=True,
        )

        gs = GridSpec(
            4,
            5,
            width_ratios=[1,1,1,1,1],
            figure=fig,
        )

        fig.suptitle(
            f"NAUTILUS Prediction Dashboard\n"
            f"Region: {self.region}    "
            f"Depth: {self.depth:.1f} m",
            fontsize=20,
            fontweight="bold",
        )

        #######################################################
        # Row 1
        #######################################################

        ax_rmse = fig.add_subplot(gs[0,0])

        ax_ssim = fig.add_subplot(gs[0,1])

        ax_mass = fig.add_subplot(gs[0,2])

        ax_summary = fig.add_subplot(gs[0,3])

        ax_model = fig.add_subplot(gs[0,4])

        self._plot_rmse_mae(
            ax_rmse,
            history,
        )

        self._plot_ssim_r2(
            ax_ssim,
            history,
        )

        self._plot_mass(
            ax_mass,
            history,
        )

        self._plot_forecast_summary(
            ax_summary,
            history,
            inference_time,
            model_info,
            profiler,
        )

        self._plot_model_summary(
            ax_model,
            history,
            inference_time,
            model_info,
            profiler,
        )


        #######################################################
        # shared limits
        #######################################################

        vmin = np.nanmin([
            np.nanmin(true_first),
            np.nanmin(pred_first),
            np.nanmin(true_last),
            np.nanmin(pred_last),
        ])

        vmax = np.nanmax([
            np.nanmax(true_first),
            np.nanmax(pred_first),
            np.nanmax(true_last),
            np.nanmax(pred_last),
        ])



        #######################################################
        # Row 2 : FIRST FORECAST
        #######################################################

        ax = fig.add_subplot(gs[1,0])
        self._plot_true(ax, true_first, mld_distance=mld_distance_first)
        self._add_timestep_badge(
            ax,
            0,
        )


        ax.set_title("TRUE (t=0)")

        ax = fig.add_subplot(gs[1,1])
        self._plot_prediction(ax, pred_first, vmin, vmax, mld_distance=mld_distance_first)

        self._add_similarity_box(
            ax,
            metrics_first,
        )

        self._add_timestep_badge(
            ax,
            0,
        )

        ax.set_title("PREDICTED (t=0)")

        ax = fig.add_subplot(gs[1,2])
        self._plot_error(ax, error_first)

        self._add_timestep_badge(
            ax,
            0,
        )



        ax.set_title("ERROR (t=0)")

        ax = fig.add_subplot(gs[1,3])
        self._plot_abs_error(ax, error_first)


        self._add_error_box(
            ax,
            metrics_first,
            true_first,
        )

        self._add_timestep_badge(
            ax,
            0,
        )

     

        ax.set_title("ABS ERROR (t=0)")



        ax = fig.add_subplot(gs[1,4])
        self._plot_mld_distance(
            ax,
            mld_distance_first,
            title="MLD DISTANCE (t=0)"
        )



        #######################################################
        # Row 3 : LAST FORECAST
        #######################################################

        ax = fig.add_subplot(gs[2,0])
        self._plot_true(ax, true_last, mld_distance=mld_distance_last)
        ax.set_title(
            "TRUE",
            fontsize=10,
            fontweight="bold",
        )

        self._add_timestep_badge(
            ax,
            len(history['rmse'])-1,
        )




        ax.set_title(f"TRUE (t={len(history['rmse'])-1})")

        ax = fig.add_subplot(gs[2,1])
        self._plot_prediction(ax, pred_last, vmin, vmax, mld_distance=mld_distance_last)

        ax.set_title(
            "TRUE",
            fontsize=10,
            fontweight="bold",
        )

        self._add_similarity_box(
            ax,
            metrics_last,
        )

        self._add_timestep_badge(
            ax,
            len(history['rmse'])-1,
        )

        ax.set_title(f"PREDICTED (t={len(history['rmse'])-1})")

        ax = fig.add_subplot(gs[2,2])
        self._plot_error(ax, error_last)
        ax.set_title(f"ERROR (t={len(history['rmse'])-1})")

        self._add_timestep_badge(
            ax,
            len(history['rmse'])-1,
        )




        ax = fig.add_subplot(gs[2,3])
       

        self._plot_abs_error(
            ax,
            error_last,
        )

        self._add_timestep_badge(
            ax,
            len(history['rmse'])-1,
        )


 
        self._add_error_box(
            ax,
            metrics_last,
            true_last,
        )


        ax.set_title(f"ABS ERROR (t={len(history['rmse'])-1})")



        ax = fig.add_subplot(gs[2,4])
        self._plot_mld_distance(
            ax,
            mld_distance_last,
            title=f"MLD DISTANCE (t={len(history['rmse'])-1})"
        )


        #######################################################
        # Row 4
        #######################################################

        ax_qq = fig.add_subplot(gs[3,0])

        ax_hist = fig.add_subplot(gs[3,1])

        ax_mass_error = fig.add_subplot(gs[3,2])

        ax_prof = fig.add_subplot(gs[3,3:])

        self._plot_qq(
            ax_qq,
            qq_true,
            qq_pred,
        )

        self._plot_histogram(
            ax_hist,
            error_last,
        )

        self._plot_mass_error(
            ax_mass_error,
            history,
        )

        self._plot_profiler(
            ax_prof,
            profiler,
        )

        #######################################################

        outfile = os.path.join(
            self.output_dir,
            save_name,
        )

        plt.savefig(
            outfile,
            dpi=self.dpi,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"📊 Prediction dashboard saved → {outfile}")




