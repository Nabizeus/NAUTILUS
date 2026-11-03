import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from src.model_conv2d_asym_hybrid import OceanConv2DAsymXPU
from src.data import WindowedOceanDataset
from src.utils import save_checkpoint
from src.activation_maps import visualize_activation_maps
from src.activation_dashboard import plot_activation_dashboard, plot_activation_and_skill_profile
from src.weight_viz import plot_network_weights_dashboard
from src.water_column_viz import plot_water_column_comparison
from src.depth_skill_profile import (
    summarize_depth_skill_per_z,
    estimate_mean_mld_m,
    bin_depth_skill,
    append_to_master_csv,
)
from src.utils import compute_basic_metrics
#from skimage.metrics import structural_similarity as ssim
from src.utils import compute_ssim as ssim
import csv
import hydra
from omegaconf import OmegaConf
import imageio.v2 as imageio
from src.io.nemo_restart_writer import write_nemo_restart
from tqdm import tqdm
import time
from src.make_activation_gifs import build_gif_for_layer
from src.vis_compare import save_pred_activation_compare
from src.activation_maps import get_activation_maps
from src.utils import get_device
from src.utils import SimpleProfiler
import datetime
from src.dashboard_predict import NautilusPredictionDashboard
import pandas as pd

from src.plot_cross_region_heatmap import (
   build_cross_region_report,
)



def plot_land_mask_predict(mask, region_name, out_dir):
    import matplotlib.pyplot as plt
    import os

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(4, 4))
    plt.imshow(mask.astype(int), cmap="gray")
    plt.title(f"Land Ocean Mask | {region_name}")
    plt.colorbar(label="1 = Ocean, 0 = Land")
    plt.tight_layout()

    path = os.path.join(out_dir, f"land_mask_{region_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"Prediction land mask saved  {path}")



def plot_joint_histogram(true_dn, pred_dn, region, savepath):
    

    plt.figure(figsize=(6,5))
    plt.hist2d(
        true_dn.flatten(),
        pred_dn.flatten(),
        bins=40,
        cmap="viridis"
    )
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Joint Histogram (True vs Pred) @{region}")
    plt.colorbar(label="Density")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_qq(true_dn, pred_dn, region, depth, savepath):
    
    

    t = np.sort(true_dn.flatten())
    p = np.sort(pred_dn.flatten())
    n = min(len(t), len(p))
    t = t[:n]
    p = p[:n]

    plt.figure(figsize=(5,5))
    plt.plot(t, p, 'o', markersize=2, alpha=0.4, label="Quantiles")
    minv = min(t.min(), p.min())
    maxv = max(t.max(), p.max())
    plt.plot([minv, maxv], [minv, maxv], 'r--', label="Ideal 1:1")
    plt.xlabel("True Quantiles")
    plt.ylabel("Predicted Quantiles")
    plt.title(f"Plot @{region} @ {depth} m")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_error_map(true_dn, pred_dn, region, depth, savepath):
    

    err = pred_dn - true_dn

    plt.figure(figsize=(5,4))
    plt.imshow(err, cmap="coolwarm")
    plt.colorbar(label="Prediction Error (Pred - True)")
    plt.title(f"Spatial Error Map @ {region} @ {depth}")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()









def plot_histograms(true_dn, pred_dn, region, depth, savepath):
    plt.figure(figsize=(8,4))
    plt.title(f"True @ {region} @ {depth}")
    plt.subplot(1,2,1)
    plt.hist(true_dn.flatten(), bins=40, alpha=0.7, label="True")
    #plt.title("True")


    plt.subplot(1,2,2)
    plt.title("Predicted Histogram")
    plt.hist(pred_dn.flatten(), bins=40, alpha=0.7, label="Predicted")
    #plt.title("Predicted Histogram")

    #plt.tight_layout()
    plt.savefig(savepath)
    plt.close()




def denormalize_tracer(x, stats, z_idx):

    """
    Depth-aware denormalization.

    Supports:
      - single depth: x = (Y, X)
      - full column: x = (Z, Y, X)
    """

    tmin = np.asarray(stats["tracer_min"][z_idx])
    tmax = np.asarray(stats["tracer_max"][z_idx])

    x = np.asarray(x)


    # Full-column case:
    # x      = (Z, Y, X)
    # tmin   = (Z,)
    # tmax   = (Z,)
    # reshape to (Z, 1, 1) for broadcasting
    if x.ndim == 3 and np.ndim(tmin) == 1:
        tmin = tmin[:, None, None]
        tmax = tmax[:, None, None]



    return x * (tmax - tmin) + tmin




def load_model_from_checkpoint(cfg, device, checkpoint_region, input_channels):
    """Load a model checkpoint robustly, handling multiple naming conventions."""
    model = OceanConv2DAsymXPU(
        cfg,
        input_channels=input_channels,
        ).to(device) 


    checkpoint_dir = getattr(cfg.training, "checkpoint_dir", "./checkpoints")
    experiment_tag = getattr(cfg.training, "experiment_name", None)

    # Try the experiment-tagged name first (what a depth-budget/config
    # sweep needs to keep separate checkpoints per config), then fall
    # back to the old {region}_best_model.pt naming so checkpoints
    # trained before this naming change still load.
    candidates = []
    if experiment_tag:
        candidates.append(
            os.path.join(checkpoint_dir, f"{experiment_tag}_{checkpoint_region}_best_model.pt")
        )
    candidates.append(os.path.join(checkpoint_dir, f"{checkpoint_region}_best_model.pt"))

    checkpoint_path = next((p for p in candidates if os.path.exists(p)), None)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"[WARNING!] No checkpoint found. Tried: {candidates}"
        )

    print(f"📂 Loading checkpoint for region '{checkpoint_region}' → {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # --- Handle multiple formats safely ---
    try:
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("[CHECKPOINT] Loaded 'model_state_dict' successfully")
        elif "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            print("[CHECKPOINT] Loaded 'model_state' successfully")
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            print("[CHECKPOINT] Loaded 'state_dict' successfully")
        else:
            model.load_state_dict(checkpoint)
            print("[CHECKPOINT] Loaded raw state_dict directly")
    except Exception as e:
        print(f"[EXCEPTION] Could not load checkpoint properly: {e}")
        raise RuntimeError("[ERROR] Checkpoint does not contain valid model weights")

    model.eval()
    print("[DONE] Model ready for inference")

    return model


def visualize_forecast(true, pred, x_range, y_range, title, save_path, vmin=None, vmax=None):
    
    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    plt.figure(figsize=(8, 4))
    im1 = plt.subplot(1, 2, 1)
    plt.imshow(true, extent = extent, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.title("True")
    plt.colorbar()

    im2 = plt.subplot(1, 2, 2)
    plt.imshow(pred, extent=extent,cmap="viridis", vmin=vmin, vmax=vmax)
    plt.title("Predicted")
    plt.colorbar()
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def animate_forecasts(image_paths, save_gif):

    frames = [imageio.imread(p) for p in sorted(image_paths)]
    imageio.mimsave(save_gif, frames, duration=0.4)
    print(f"[SAVED] Saved animation: {save_gif}")


# For Volume or 3 Layer system
def flatten_depth(inputs):
    B, T, C, Z, H, W = inputs.shape
    return inputs.view(B, T, C * Z, H, W)




def main():
    import sys

    def _run_core(cfg):
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = get_device(cfg)
        print(f"[Predict] Using device: {device}")
 
        start_time_0 = time.time() # <-- Start timer overall

        ts = datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")
        
        model_name = cfg.model.name
        
        num_levels = cfg.data.vertical_levels
        num_epochs = cfg.training.epochs
        num_inputs = cfg.data.window.input_steps

        mode=f"Predict_mod{model_name}_ep{num_epochs}_lev{num_levels}_inp{num_inputs}"
       
      
      


        # -----------------------------------------------------
        # Store results from ALL model × dataset evaluations
        # -----------------------------------------------------


        cross_results = []


        # =====================================================
        # Loop over TRAINED MODELS (checkpoints)
        # =====================================================


 
        for checkpoint_region in cfg.prediction.training_regions:
      


            print("="*80)
            print(f"MODEL = {checkpoint_region}")
            print("="*80)


            # =================================================
            # Loop over TEST DATASETS FOR GENERALIZE
            # =================================================



            for region_cfg in cfg.data.regions:

            
                    best_ssim = -1e9
                    worst_ssim = 1e9

                    median_candidates = []

                    best_true = None
                    best_pred = None
                    best_error = None

                    worst_true = None
                    worst_pred = None
                    worst_error = None

            



     
                    test_region = region_cfg["name"]

     
                    experiment_tag = getattr(cfg.training, "experiment_name", None)
                    run_name = (
                        f"{experiment_tag}_{checkpoint_region}_to_{test_region}"
                        if experiment_tag else
                        f"{checkpoint_region}_to_{test_region}"
                    )



 
                    prof = SimpleProfiler(
                        save_path=f"profiler/Profile_{mode}_{run_name}_{ts}.log",
                        region=test_region,
                        mode=mode
                    )


                    prof.start("predict_load_data")
        
                    start_time = time.time() # <-- Start timer
                    dataset = WindowedOceanDataset(cfg, split="test", region_name=test_region)
                    input_channels = dataset.n_channels #3D* dataset.n_depths
                    print(f"[DATASET] Dataset 'test' indices {dataset.t_start}:{dataset.t_end}, total {len(dataset)} samples") 
                    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


                    if dataset.n_depths != 75:
                        raise RuntimeError(
                            f"[PREDICT] Checkpoint expects 75 depth levels, "
                            f"but test dataset has {dataset.n_depths}"
                        )

                    print(
                        f"[PREDICT] Using full 75-level water column: "
                        f"{dataset.depths[0]:.2f} m -> {dataset.depths[-1]:.2f} m"
                    )


                    prof.stop()



                    dashboard_depth = np.argmin(
                    np.abs(dataset.depths)
                    )

            
            
                    print(f"Predict  from data  {test_region}")
                    print(f"Predict for checkpoint  {checkpoint_region}")
                    print(f"Dashboard depth {dashboard_depth}")
               

            

                    x1, x2 = dataset.x_range
                    y1, y2 = dataset.y_range

                    prof.start("predict_load_model")            
                    model = load_model_from_checkpoint(
                        cfg,
                        device,
                        checkpoint_region,
                        input_channels
                    )


                    # -------------------------------------------------------
                    # Model statistics
                    # -------------------------------------------------------

                    total_params = sum(p.numel() for p in model.parameters())

                    trainable_params = sum(
                        p.numel()
                        for p in model.parameters()
                        if p.requires_grad
                    )

                    model_size_mb = (
                        total_params * 4
                    ) / (1024 ** 2)      # float32


                    print(f"Total PARAM: {total_params}")
                    print(f"Trainable PARAM: {trainable_params}")
                    print(f"Model Size: {model_size_mb:.2f}")



                    prof.stop()

                    output_dir = os.path.join("predictions", run_name)
                    os.makedirs(output_dir, exist_ok=True)
                    output_dir_statistics = os.path.join(output_dir, "statistics")
                    os.makedirs(output_dir_statistics, exist_ok=True)
                    output_dir_frames = os.path.join(output_dir, "frames")
                    os.makedirs(output_dir_frames, exist_ok=True)
                    output_dir_movies = os.path.join(output_dir, "movies")
                    os.makedirs(output_dir_movies, exist_ok=True)


 
                    # ==========================================================
                    # Prediction history
                    # ==========================================================



                    # --------------------------------------------------
                    # One folder per prediction depth
                    # --------------------------------------------------

                    frame_dirs = {}
                    stats_dirs = {}
                    image_paths = {}
                    histories = {}
                    dashboard_images = {}
                    activation_profiles_i0 = None  # captured at i==0 below, reused after the test loop



                    for z in dataset.valid_z:


                        histories[z] = {
                            "rmse": [],
                            "mae": [],
                            "ssim": [],
                            "r2": [],
                            "rmse_pct": [],
                            "mae_pct": [],
                            "mass_true": [],
                            "mass_pred": [],
                            "mass_error": [],
                            "mass_percent": [],
                            "mean_error": [],
                            "std_error": [],
                            "true_variance": [],
                        }


                        dashboard_images[z] = {

                            # first forecast
                            "true_first": None,
                            "pred_first": None,
                            "error_first": None,
                            "mld_distance_first": None,
                            # last forecast
                            "true_last": None,
                            "pred_last": None,
                            "error_last": None,
                            "mld_distance_last": None,

                        }


                        depth = float(dataset.depths[z])

                        frame_dir = os.path.join(
                            output_dir_frames,
                            f"z{z:02d}_{depth:.0f}m"
                        )

                        stats_dir = os.path.join(
                            output_dir_statistics,
                            f"z{z:02d}_{depth:.0f}m"
                        )

                        os.makedirs(frame_dir, exist_ok=True)
                        os.makedirs(stats_dir, exist_ok=True)

                        frame_dirs[z] = frame_dir
                        stats_dirs[z] = stats_dir

                        image_paths[z] = []


                    #plot_land_mask_predict(
                    #    dataset.land_mask,
                    #    test_region,
                    #    out_dir=output_dir
                    #)


                    loss_fn = torch.nn.MSELoss(reduction = 'sum')
                    total_loss = 0.0
                    total_rmse = 0.0
                    total_mae = 0.0
                    total_ssim = 0.0
                    total_r2 = 0.0
             
                    total_rmse_pct = 0.0
                    total_mae_pct = 0.0    
            
                    all_true, all_pred = [], []


                    pbar = tqdm(loader, total=len(loader))

                    stats = dataset.stats 
            
                    for i, (inputs, targets, mask) in enumerate(pbar):

        
                        t_idx  = dataset.indices[i]
                        # Full-column prediction: all depth levels
                        z_global = dataset.valid_z.copy()

                   
                        

                        z_level = dataset.depths[z_global].copy()      # physical depth in meters

                        # ---------------------------------------------------------
                        # MLD distance for the TARGET / FORECAST timestep
                        # ---------------------------------------------------------
                        target_t_idx = t_idx + cfg.data.window.input_steps

                        mld_distance_np = dataset.mld_distance[
                            target_t_idx,
                            :
                        ].copy()              
                  

                        # MLD distance was normalized in data.py:
                        # normalized = (raw - mean) / std
                        # Undo that normalization here.
                        mld_distance_np = (
                            mld_distance_np * stats["mld_distance_std"]
                            + stats["mld_distance_mean"]
                        )

                        #print(
                        #    f"📏 MLD distance DENORM: "
                        #    f"min={np.nanmin(mld_distance_np):.2f} m, "
                        #    f"max={np.nanmax(mld_distance_np):.2f} m"
                        #)
 
                        if i % 1 == 0:
                            inputs, targets = inputs.to(device), targets.to(device)
                            with torch.no_grad():
                                prof.start("predict_forward")
                                preds = model(inputs)
                                prof.stop()


                                # Check for NaNs in preds or targets
                                if torch.isnan(preds).any() or torch.isnan(targets).any():
                            
                            
                                    preds = torch.nan_to_num(preds, nan=0.0)
                                               
                            if i == 0:  # first sample only
                                # `visualize_activation_maps` now runs a real forward
                                # pass internally (see the fixed src/activation_maps.py)
                                # and needs the model's NATIVE input shape — pass the
                                # raw `inputs` tensor as-is, not a sliced/reduced copy.
                                prof.start("activation_maps")
                                visualize_activation_maps(
                                    model,
                                    inputs,
                                    i,
                                    test_region,
                                    out_dir=os.path.join(output_dir, "activations"),
                                )
                                prof.stop()

                                # One-glance summary across all layers/channels, as a
                                # complement to the per-layer PNGs above (those still
                                # feed make_activation_gifs.py for animating a single
                                # layer across samples). Also captures the activation
                                # profile itself so it can be combined with the
                                # full-test-set skill profile after the test loop ends
                                # (that data doesn't exist yet at i==0).
                                prof.start("activation_dashboard")
                                _, activation_profiles_i0 = plot_activation_dashboard(
                                    model,
                                    inputs,
                                    sample=i,
                                    region_name=test_region,
                                    out_dir=os.path.join(output_dir, "activations_dashboard"),
                                    depths=dataset.depths[dataset.valid_z],
                                    return_profiles=True,
                                )
                                prof.stop()

                                # Learned weights don't depend on the input at all — one
                                # call per (checkpoint_region, test_region) is enough,
                                # doesn't need to be inside the sample loop really, but
                                # kept here since this is already the "only run once,
                                # at i==0" block.
                                prof.start("weight_dashboard")
                                plot_network_weights_dashboard(
                                    model,
                                    out_dir=os.path.join(output_dir, "weights_dashboard"),
                                    region_name=f"{checkpoint_region}_to_{test_region}",
                                )
                                prof.stop()

                                # NOTE: get_activation_maps() + save_pred_activation_compare()
                                # are left disabled here. save_pred_activation_compare (in
                                # vis_compare.py) calls `dataset.z_indices.index(dataset.z_index)`,
                                # but dataset.z_indices is a numpy array, not a list, so
                                # `.index()` will raise AttributeError. That's a separate bug
                                # in vis_compare.py worth fixing before re-enabling this path.
                                #
                                # activation_maps_dict = get_activation_maps(model, inputs)
                                # act = activation_maps_dict["convlstm_hidden"]
                                # save_pred_activation_compare(preds, act, targets, i, test_region, "convlstm_hidden", dataset)
                    
                            if i == 0:
                                print("[PREDICT] TRAIN PRIOR NP =====")
                                print("targets shape, dim:", targets.shape, targets.dim())
                                print("preds shape, dim:", preds.shape, preds.dim())
                                print("mask shape, dim:", mask.shape, mask.dim())

                                print("==========================\n")

                    

                            prof.start("Metrics...")


                            # =========================================================
                            # FULL-COLUMN SHAPE HANDLING
                            #
                            # Model / target format:
                            #   [B, Tpred, Z, Y, X]
                            #
                            # Current configuration:
                            #   [1, 1, 75, Y, X]
                            #
                            # Downstream code expects:
                            #   [Z, Y, X]
                            # =========================================================

                            # --- TARGET SHAPE ---
                            if targets.dim() == 5:
                                # [B, Tpred, Z, Y, X]
                                if targets.shape[0] != 1 or targets.shape[1] != 1:
                                    raise ValueError(
                                        f"Unexpected target batch/time shape: {targets.shape}"
                                    )
                                true_np = targets[0, 0].cpu().numpy()

                            elif targets.dim() == 4:
                                # [B, Z, Y, X]
                                true_np = targets[0].cpu().numpy()

                            elif targets.dim() == 3:
                                # [Z, Y, X]
                                true_np = targets.cpu().numpy()

                            else:
                                raise ValueError(
                                    f"Unexpected target shape: {targets.shape}"
                                )


                            # --- PREDICTION SHAPE ---
                            if preds.dim() == 5:
                                # [B, Tpred, Z, Y, X]
                                if preds.shape[0] != 1 or preds.shape[1] != 1:
                                    raise ValueError(
                                        f"Unexpected prediction batch/time shape: {preds.shape}"
                                    )
                                pred_np = preds[0, 0].cpu().numpy()

                            elif preds.dim() == 4:
                                # [B, Z, Y, X]
                                pred_np = preds[0].cpu().numpy()

                            elif preds.dim() == 3:
                                # [Z, Y, X]
                                pred_np = preds.cpu().numpy()

                            else:
                                raise ValueError(
                                    f"Unexpected prediction shape: {preds.shape}"
                                )


                            # --- MASK SHAPE ---
                            if mask.dim() == 5:
                                # [B, Tpred, Z, Y, X]
                                if mask.shape[0] != 1 or mask.shape[1] != 1:
                                    raise ValueError(
                                        f"Unexpected mask batch/time shape: {mask.shape}"
                                    )
                                mask_np = mask[0, 0].cpu().numpy().astype(bool)

                            elif mask.dim() == 4:
                                # [B, Z, Y, X]
                                mask_np = mask[0].cpu().numpy().astype(bool)

                            elif mask.dim() == 3:
                                # [Z, Y, X]
                                mask_np = mask.cpu().numpy().astype(bool)

                            elif mask.dim() == 2:
                                # [Y, X] -- broadcast later if necessary
                                mask_np = mask.cpu().numpy().astype(bool)

                            else:
                                raise ValueError(
                                    f"Unexpected mask shape: {mask.shape}"
                                )


                            # ---------------------------------------------------------
                            # SANITY CHECK
                            # ---------------------------------------------------------
                            print(
                                f"📐 Full-column arrays: "
                                f"pred={pred_np.shape}, "
                                f"true={true_np.shape}, "
                                f"mask={mask_np.shape}"
                            ) 



     
                            pred_np = denormalize_tracer(pred_np, stats, z_global)
                            true_np = denormalize_tracer(true_np, stats, z_global)

                            if i == 0:
                                # Land/invalid points should not pollute the color
                                # scale or the depth-integrated averages.
                                true_masked = np.where(mask_np, true_np, np.nan)
                                pred_masked = np.where(mask_np, pred_np, np.nan)

                                prof.start("water_column_viz")
                                plot_water_column_comparison(
                                    true_field=true_masked,
                                    pred_field=pred_masked,
                                    depths=dataset.depths[z_global],
                                    title=f"{test_region} | forecast quality | t={t_idx}",
                                    save_path=os.path.join(
                                        output_dir, "watercolumn",
                                        f"{test_region}_t{t_idx:04d}_watercolumn_compare.png",
                                    ),
                                    mld_distance=mld_distance_np,
                                )
                                prof.stop()



                            if i == 0 or i == 10:

                                print(f"\n PREDICT POST NP  DEBUG {i} =====")
                                print("true:", true_np.shape)
                                print("preds :", pred_np.shape)
                                print("mask:", mask_np.shape)

                                print("==========================\n")


                                print("\n [PREDIC] SCALE DEBUG")
                                print(f"TRUE: {true_np.min():.2f}, {true_np.max(): .2f}")
                                print(f"PRED: {pred_np.min(): .2f}, {pred_np.max(): .2f}")


                                #print("z_local used:", z_local)
                                print("tracer_min:", stats["tracer_min"][z_global])
                                print("tracer_max:", stats["tracer_max"][z_global])


                                print("z_global:", z_global)
                                #print("z_local :", z_local)
                                print("stats shape:", stats["tracer_min"].shape)


            


                            for z_global in dataset.valid_z:



                                z_local = dataset.z_to_local[z_global]
                                z_level = float(dataset.depths[z_global])

                                pred_z = pred_np[z_global]
                                true_z = true_np[z_global]
                                mask_z = mask_np[z_global]
                                mld_distance_z = mld_distance_np[z_global]

                                # NOTE: pred_np / true_np were already denormalized
                                # above (full-column, all Z at once). pred_z/true_z
                                # are just per-depth slices of that already-physical
                                # data — do NOT denormalize again here, or the min/max
                                # rescaling gets applied twice, corrupting every
                                # downstream metric, dashboard plot, and the NEMO
                                # restart file.





                                # Mask: True = valid ocean, False = land/missing
                                valid = mask_z

                                if valid.shape != pred_z.shape:
                                    if valid.T.shape == pred_z.shape:
                                        valid = valid.T
                                    else:
                                        raise ValueError(
                                            f"Mask shape {valid.shape} incompatible with prediction {pred_z.shape}"
                                        )

                                # For visualization, keep NaNs on land:
                                true_vis = true_z.copy()
                                pred_vis = pred_z.copy()
                                true_vis[~valid] = np.nan #non-valid points will be set to NaN as Land pixels
                                pred_vis[~valid] = np.nan
                                mld_distance_vis = mld_distance_z.copy()
                                mld_distance_vis[~valid] = np.nan


                                
                                error_np = pred_vis - true_vis

                                rmse = np.sqrt(
                                    np.nanmean(error_np**2)
                                )

                                mae = np.nanmean(
                                    np.abs(error_np)
                                )

                                bias = np.nanmean(
                                    error_np
                                )

                                std_error = np.nanstd(
                                    error_np
                                )

                    



                                # For metrics, work only on valid pixels
                                true_valid = true_z[valid]
                                pred_valid = pred_z[valid]

                                # --- Metrics (RMSE, MAE, R²) ---
                                _rmse = np.sqrt(np.nanmean((pred_valid - true_valid) ** 2))
                                _mae  = np.nanmean(np.abs(pred_valid - true_valid))

                                mean_true = np.nanmean(true_valid)

                                _rmse_pct = 100.0 * _rmse / max(mean_true, 1e-12)
                                _mae_pct  = 100.0 * _mae  / max(mean_true, 1e-12)



                                den = np.sum((true_valid - np.nanmean(true_valid))**2)
                                if den > 0:
                                    _r2 = 1.0 - (np.sum((pred_valid - true_valid)**2) / den)
                                else:
                                    _r2 = np.nan

                                # --- SSIM ---
                                # 1) build finite arrays for SSIM (NaNs -> 0, mask will zero land anyway)
                                true_ssim = np.nan_to_num(true_z, nan=0.0)
                                pred_ssim = np.nan_to_num(pred_z, nan=0.0)
                                mask_t    = torch.tensor(valid.astype(np.float32))[None,None,...]

                                _ssim = ssim(
                                    torch.tensor(pred_ssim, dtype=torch.float32)[None,None,...] * mask_t,
                                    torch.tensor(true_ssim, dtype=torch.float32)[None,None,...] * mask_t,
                                )
                    


                                true_loss = np.nan_to_num(true_z, nan=0.0)
                                pred_loss = np.nan_to_num(pred_z, nan=0.0)

                    
                                pred_t = torch.tensor(pred_loss, dtype=torch.float32, device=device)
                                true_t = torch.tensor(true_loss, dtype=torch.float32, device=device)
                                mask_t = torch.tensor(valid, dtype=torch.bool, device=device)

                                # Compute loss only on valid ocean points
                                loss = loss_fn(pred_t[mask_t], true_t[mask_t])
                                total_loss += float(loss.item())

    

                                pbar.set_postfix({"N": f"{i / len(loader):.6f}"})
                    
                                total_rmse += _rmse
                                total_mae  += _mae
                                total_r2   += _r2
                                total_ssim += _ssim


                                prof.stop() #Metrics

 

                                depth_str = f"{z_level:0f}m"

                    

                                history = histories[z_global]

                                z_str = f"{z_level:.0f}m"

                                #print(f"RUN for depth {depth_str} and z_str {z_str}")
                     

                                history["rmse"].append(_rmse)

                                history["mae"].append(_mae)


                                history["rmse_pct"].append(_rmse_pct)

                                history["mae_pct"].append(_mae_pct)


                                history["ssim"].append(_ssim)

                                history["r2"].append(_r2)

                                history["true_variance"].append(den)

                                history["mean_error"].append(bias)

                                history["std_error"].append(std_error)

                    
                                all_true.append(true_np)
                                all_pred.append(pred_np)

                                total_rmse_pct += _rmse_pct
                                total_mae_pct  += _mae_pct
                        


                                mass_true = np.nansum(true_np)

                                mass_pred = np.nansum(pred_np)


                                mass_percent = 100.0 * (mass_pred - mass_true) / (abs(mass_true) + 1e-12)

                                history["mass_percent"].append(mass_percent)

                                history["mass_true"].append(mass_true)

                                history["mass_pred"].append(mass_pred)

                                history["mass_error"].append(

                    
                                    mass_pred-mass_true

                                )
              
                   
                                best = dashboard_images[z_global]


                                forecast_step = t_idx - dataset.t_start
                            

                                best = dashboard_images[z_global]

                                # -------------------------
                                # First forecast
                                # -------------------------
                                if forecast_step == 0:

                                    best["true_first"] = true_vis.copy()
                                    best["pred_first"] = pred_vis.copy()
                                    best["error_first"] = error_np.copy()
                                    best["mld_distance_first"] = mld_distance_vis.copy()
                                # -------------------------
                                # Last forecast
                                # -------------------------
                                if forecast_step == (dataset.t_end - dataset.t_start - 1):

                                    best["true_last"] = true_vis.copy()
                                    best["pred_last"] = pred_vis.copy()
                                    best["error_last"] = error_np.copy()
                                    best["mld_distance_last"] = mld_distance_vis.copy()

                            


                                median_candidates.append({

                                    "ssim": _ssim,

                                    "true": true_vis.copy(),

                                    "pred": pred_vis.copy(),

                                    "error": error_np.copy(),

                                    "depth": z_global,

                                })

           


                            #prof.start("visualization")


                            #img_path = os.path.join(

                            #    frame_dirs[z_global],

                            #    f"forecast_t{forecast_step:03d}.png"

                            #)

                   

                            #local_vmin = 0 #min(true_np.min(), pred_np.min())
                            #local_vmax = max(true_np.max(), pred_np.max())


                    
                            #visualize_forecast(
                            #    true_vis,
                            #    pred_vis,
                            #    dataset.x_range,
                            #    dataset.y_range,
                            #    f"R:{_r2:.2f}|SSIM={_ssim:.2f}|levs: {num_levels}|ep: {num_epochs}|t: {t_idx}| @ {z_str} |inp: {num_inputs} @ {test_region}",
                            #    img_path,
                            #    vmin=local_vmin,
                            #    vmax=local_vmax
                            #)


                            #image_paths[z_global].append(img_path)


                            #str_statistics=f"t{t_idx:4d}_z{z_global:02d}_{z_str}_{model_name}"


                            #hist_path = os.path.join(

                            #stats_dirs[z_global],

                            #    f"hist_t{forecast_step:03d}.png"

                            #)

                            #qq_path = os.path.join(

                            #    stats_dirs[z_global],

                            #    f"qq_t{forecast_step:03d}.png"

                            #)

                            #err_path = os.path.join(

                            #    stats_dirs[z_global],

                            #    f"error_t{forecast_step:03d}.png"

                            #)

                            #plot_histograms(true_np, pred_np, run_name, z_str, hist_path)
                            #plot_qq(true_np, pred_np, run_name, z_str, qq_path)
                            #plot_error_map(true_np, pred_np, run_name, z_str, err_path)

                            #prof.stop()



                    prediction_time = (

                        time.time()-start_time

                    )/60.


                    num_samples = len(loader)

                    avg_rmse = total_rmse / num_samples
                    avg_mae  = total_mae  / num_samples
                    avg_ssim = total_ssim / num_samples
                    avg_r2   = total_r2   / num_samples


                    print(f"Final Test Metrics | RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, SSIM: {avg_ssim:.4f}, R²: {avg_r2:.4f}")
            

                    avg_rmse_pct = total_rmse_pct / num_samples
                    avg_mae_pct  = total_mae_pct  / num_samples


                    # Global color scale for visualization
                    #global_vmin = np.min(all_true)
                    #global_vmax = np.max(all_true)

                    #for i, p in enumerate(image_paths):
                    #    true_np, pred_np = all_true[i], all_pred[i]
                    #    img_pathx = os.path.join(output_dir_frames, f"forecast_fixed_{i:03d}.png")
                
                    #imagex_paths.append(img_pathx)
            
                    avg_loss = total_loss / num_samples
                    print(f"Average test loss for {run_name}: | {avg_loss:.2f}")

            

                    cross_results.append({

                        "model": checkpoint_region,

                        "test": test_region,

                        "RMSE": avg_rmse,
                        "RMSE_pct": avg_rmse_pct,

                        "MAE": avg_mae,
                        "MAE_pct": avg_mae_pct,

                        "SSIM": avg_ssim,

                        "R2": avg_r2,

                    })

                    # ---------------------------------------------------------
                    # Depth-skill accumulation (feeds the future depth-range x
                    # depth-bin heatmap). `histories[z]` already holds one
                    # metric value per test timestep at every depth level —
                    # this just aggregates that over the whole test set and
                    # appends it to a running CSV, tagged with which depth
                    # range this checkpoint was actually trained on, so a
                    # depth-budget/banded-window sweep can be evaluated one
                    # predict.py run at a time.
                    # ---------------------------------------------------------
                    prof.start("depth_skill_profile")

                    depth_training_cfg = getattr(cfg.data, "depth_training", None)
                    train_depth_indices = (
                        list(depth_training_cfg.indices)
                        if depth_training_cfg is not None
                        and getattr(depth_training_cfg, "indices", None)
                        else None
                    )

                    df_per_z = summarize_depth_skill_per_z(
                        histories,
                        dataset.depths,
                        dataset.valid_z,
                        run_name=run_name,
                        region_name=test_region,
                        checkpoint_region=checkpoint_region,
                        train_depth_mode=getattr(depth_training_cfg, "mode", None),
                        train_depth_min=getattr(depth_training_cfg, "min_depth_m", None),
                        train_depth_max=getattr(depth_training_cfg, "max_depth_m", None),
                        train_depth_indices=train_depth_indices,
                    )
                    append_to_master_csv(
                        df_per_z,
                        csv_path="plots/heatmaps/depth_skill_per_z_all_runs.csv",
                        dedupe_keys=["run_name", "region", "checkpoint_region", "z", "metric"],
                    )

                    # Combined view: does the network's activation strength
                    # actually track its real skill by depth? Needs BOTH the
                    # activation profile (captured earlier, at sample i==0)
                    # AND the full-test-set skill profile (just built above) —
                    # this is the first point in the loop where both exist.
                    if activation_profiles_i0 is not None:
                        prof.start("activation_skill_profile")
                        try:
                            mld_depth_m = estimate_mean_mld_m(dataset)
                        except (AttributeError, ValueError) as e:
                            print(f"⚠ Skipping MLD reference line: {e}")
                            mld_depth_m = None

                        plot_activation_and_skill_profile(
                            activation_profiles_i0,
                            dataset.depths[dataset.valid_z],
                            skill_df_per_z=df_per_z,
                            skill_metrics=("ssim", "r2"),
                            mld_depth_m=mld_depth_m,
                            save_path=os.path.join(
                                output_dir, "activations_dashboard",
                                f"{run_name}_activation_and_skill_profile.png",
                            ),
                        )
                        prof.stop()

                    # Uniform, configurable bin width — critical for the banded
                    # sweep: a 500m-band-trained model needs its evaluation bin
                    # edges to land EXACTLY on 500m too, or "inside trained
                    # range" vs "extrapolated" becomes ambiguous for any bin
                    # straddling a band boundary. Set via
                    # `evaluation.depth_bin_width_m` in the config
                    # (generate_depth_sweep_configs.py sets this to match
                    # --step automatically); defaults to 500m.
                    eval_cfg = getattr(cfg, "evaluation", None)
                    bin_width = float(getattr(eval_cfg, "depth_bin_width_m", 500.0))
                    max_depth = float(np.max(dataset.depths[dataset.valid_z]))
                    bin_edges = np.arange(0.0, max_depth + bin_width, bin_width)
                    depth_bins = [
                        (float(lo), float(lo + bin_width)) for lo in bin_edges
                        if lo < max_depth
                    ]
                    df_binned = bin_depth_skill(df_per_z, depth_bins)
                    append_to_master_csv(
                        df_binned,
                        csv_path="plots/heatmaps/depth_skill_binned_all_runs.csv",
                        dedupe_keys=["run_name", "region", "checkpoint_region", "depth_bin", "metric"],
                    )

                    prof.stop()





       
                    #gif_pathx = os.path.join(output_dir_frames, f"forecast_movie_fix_{run_name}.gif")
                    #animate_forecasts(sorted(imagex_paths), gif_pathx)


                    test_csv = f"logs/test_metrics_{run_name}.csv"
                    with open(test_csv, "a", newline="") as f:
                        writer = csv.writer(f)
                        if f.tell() == 0:
                            writer.writerow(["LOSS","RMSE", "MAE", "SSIM", "R2"])
                        writer.writerow([avg_loss, avg_rmse, avg_mae, avg_ssim, avg_r2])



                    z_index = getattr(dataset, "z_index", None)


                    prof.start("Write restart file")

                    write_nemo_restart(
                                pred_field=pred_np,
                                dataset=dataset,
                                region_name=test_region,
                                varname="mytrc",       # or whatever tracer name NEMO expects
                                z_index=z_index,
                                out_dir="nemo_restart"
                            )
        
                    prof.stop()
                    # ---------------------------------------------------------
                    # Select representative prediction (median SSIM)
                    # ---------------------------------------------------------

                    prof.start("dashboard")

                    median_candidates = sorted(
                        median_candidates,
                        key=lambda x: x["ssim"]
                    )

                    median = median_candidates[
                        len(median_candidates) // 2
                    ]


                    # --------------------------------------------------
                    # One GIF for every prediction depth
                    # --------------------------------------------------
            
           
            

                    model_info = {

                        "name": cfg.model.name,

                        "parameters": total_params,

                        "trainable": trainable_params,

                        "size_mb": model_size_mb,

                        "input_steps": cfg.data.window.input_steps,

                        "prediction_steps": cfg.data.window.pred_steps,

                        "depths": len(dataset.valid_z),

                        "channels": dataset.n_channels * dataset.n_depths,

                    }

            
                    # ---------------------------------------------------------
                    # Only build the per-depth prediction dashboard PNG for a
                    # handful of representative depths, not all 75 levels —
                    # this loop used to run dashboard.plot_prediction_dashboard()
                    # for every z (75 files); most of those are near-duplicates
                    # of their neighbors. histories[z] itself (all 75 levels)
                    # still feeds the depth-skill CSV regardless — only the
                    # dashboard IMAGE generation is restricted here.
                    #
                    # Override via cfg.predict.dashboard.depth_targets_m (a
                    # list of meters); defaults to every 1000m from 0 to the
                    # deepest level in this dataset.
                    # ---------------------------------------------------------
                    all_depths_m = dataset.depths[dataset.valid_z]
                    dashboard_cfg = getattr(getattr(cfg, "predict", None), "dashboard", None)
                    target_depths_m = getattr(dashboard_cfg, "depth_targets_m", None)
                    if not target_depths_m:
                        max_depth_m = float(np.max(all_depths_m))
                        target_depths_m = list(np.arange(0.0, max_depth_m + 1000.0, 1000.0))

                    selected_z = []
                    for target in target_depths_m:
                        idx_local = int(np.argmin(np.abs(all_depths_m - target)))
                        z_val = int(dataset.valid_z[idx_local])
                        if z_val not in selected_z:
                            selected_z.append(z_val)

                    print(
                        f"[DASHBOARD] Building prediction dashboards for "
                        f"{len(selected_z)}/{len(dataset.valid_z)} depth levels "
                        f"(targets ~{[int(t) for t in target_depths_m]}m) "
                        f"→ z={selected_z}"
                    )

                    for z in selected_z:
                            
                        print(f"[DASHBOARD] dataset.valid_z {dataset.valid_z}")
                        depth = dataset.depths[z]

                    #    gif_path = os.path.join(
                    #        output_dir_movies,
                    #        f"forecast_movie_{run_name}_z{z:02d}_{depth:.0f}m.gif"
                    #    )

                    #    animate_forecasts(
                    #        sorted(image_paths[z]),
                    #        gif_path
                    #    )


                        dashboard = NautilusPredictionDashboard(

                            region_name=run_name,
                
                            depth_level=float(dataset.depths[z]),

                            profiler=prof,

                            output_dir=f"artifacts/dashboard/prediction"

                        )

 
                        dashboard.plot_prediction_dashboard(

                            history=histories[z],

                            profiler=prof,
                    
                            true_first=dashboard_images[z]["true_first"],
                            pred_first=dashboard_images[z]["pred_first"],
                            error_first=dashboard_images[z]["error_first"],
                            mld_distance_first=dashboard_images[z]["mld_distance_first"],                            

                            true_last=dashboard_images[z]["true_last"],
                            pred_last=dashboard_images[z]["pred_last"],
                            error_last=dashboard_images[z]["error_last"],
                            mld_distance_last=dashboard_images[z]["mld_distance_last"],
                   

                            metrics_first={
                                "rmse": histories[z]["rmse"][0],
                                "mae": histories[z]["mae"][0],
                                "ssim": histories[z]["ssim"][0],
                                "r2": histories[z]["r2"][0],
                            },

                            metrics_last={
                                "rmse": histories[z]["rmse"][-1],
                                "mae": histories[z]["mae"][-1],
                                "ssim": histories[z]["ssim"][-1],
                                "r2": histories[z]["r2"][-1],
                            },




                            qq_true=dashboard_images[z]["true_last"],
                            qq_pred=dashboard_images[z]["pred_last"],

                            inference_time=prediction_time,

                            model_info=model_info,

                        save_name=f"{run_name}_prediction_dashboard_z{z:02d}_{depth:.0f}m.png",

                    )


                    prof.stop()
           
                    print(f"[REGION LOOP]")

            # END REGION LOOP

            print(f"[LOOP CHECKPOINT]")   

        # END CHECKPOINT LOOP


        # ==========================================================
        # Cross-region evaluation summary
        # ==========================================================


        prof.start("heatmaps")
        df = pd.DataFrame(cross_results)

        # Tag the report with the experiment name so the runs of a sweep
        # (one per depth range) don't overwrite each other's files.
        report_tag = getattr(cfg.training, "experiment_name", None)
        csv_file = (
            f"plots/heatmaps/cross_region_metrics_{report_tag}.csv"
            if report_tag else
            "plots/heatmaps/cross_region_metrics.csv"
        )
        df.to_csv(csv_file, index=False)

        print(f"[CROSS METRICS] Saved cross-region metrics to {csv_file}")

        build_cross_region_report(df, tag=report_tag)

        prof.stop()

        #layers = ["conv1", "conv2", "convlstm_hidden", "decoder_out"]

        #for layer in layers:
        #    build_gif_for_layer(run_name, layer)
       

        prof.report()


        prof.plot(color="red")

        end_time = time.time()     # <-- End timer
        elapsed = (end_time - start_time_0) / 60  # minutes

        
        print("\n" + "=" * 90)
        print("[FINAL]  Model evaluation finished successfully")
        print(f"[END] Total Inrence Loop {elapsed:.2f}\n")
    
        print("=" * 90 + "\n")
       

    if "--config" in sys.argv:
        # Bypass Hydra's config_path/config_name resolution entirely and
        # load an explicit YAML file — this is what lets a depth-budget/
        # banded-window sweep drive predict.py without touching source
        # between runs, e.g.:
        #   python3 -m src.predict --config configs/depth_sweep/depthbudget_0-2000m.yaml
        idx = sys.argv.index("--config")
        config_path = sys.argv[idx + 1]
        del sys.argv[idx:idx + 2]  # keep Hydra's own arg parsing untouched
        cfg = OmegaConf.load(config_path)
        _run_core(cfg)
    else:
        # Default (unchanged) behavior: normal Hydra CLI, using whatever
        # config_name is the current default.
        hydra.main(
            config_path="../config",
            config_name="conv2dlstm_s323D_fullcolumn_GPU.yaml",
            version_base=None,
        )(_run_core)()
    

if __name__ == "__main__":
    main()

