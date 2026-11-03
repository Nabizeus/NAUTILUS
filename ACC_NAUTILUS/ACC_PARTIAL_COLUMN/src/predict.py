import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from src.model_conv2d_asym_hybrid import OceanConv2DAsymXPU
from src.data import WindowedOceanDataset
from src.utils import save_checkpoint
from src.activation_maps import visualize_activation_maps
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
    plot_heatmap,
    generalization_score,
    plot_transfer_loss,
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
    tmin = stats["tracer_min"][z_idx]
    tmax = stats["tracer_max"][z_idx]
    return x * (tmax - tmin) + tmin


#def denormalize_tracer(x, region_name):
#    """
#    Denormalize tracer values using either:
##    - Min–max scaling (preferred), or
#    - Z-score fallback (mean/std) if min/max not present.
#
#    x: normalized tracer array
#    region_name: region ID used for loading stats
#    """
#    stats_file = f"norms/norm_stats_{region_name}.npz"
#    stats = np.load(stats_file)

    # Case 1: NEW normalization (min-max)
#    if "tracer_min" in stats and "tracer_max" in stats:
#        tmin = float(stats["tracer_min"])
#        tmax = float(stats["tracer_max"])
#        if abs(tmax - tmin) < 1e-12:
#            # fallback to z-score if degenerate
#            mean = float(stats.get("tracer_mean", 0.0))
#            std  = float(stats.get("tracer_std", 1.0))
#            return x * std + mean
#        return x * (tmax - tmin) + tmin
#    else:
#        # fallback: old z-score for U V W
#        mean = float(stats["tracer_mean"])
#        std  = float(stats["tracer_std"])
#        return x * std + mean

def load_model_from_checkpoint(cfg, device, checkpoint_region, input_channels):
    """Load a model checkpoint robustly, handling multiple naming conventions."""
    model = OceanConv2DAsymXPU(
        cfg,
        input_channels=input_channels,
        ).to(device) 


    checkpoint_dir = getattr(cfg.training, "checkpoint_dir", "./checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_region}_best_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[WARNING!] No checkpoint found at {checkpoint_path}")

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
    
    

    @hydra.main(config_path="../config", config_name="acc_conv2lstm_vert", version_base=None)
    def run(cfg):
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

     
                    run_name = f"{checkpoint_region}_to_{test_region}"



 
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

                    prof.stop()



                    dashboard_depth = np.argmin(
                    np.abs(dataset.depths - region_cfg.dashboard_depth_m)
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

        
                        t_idx, z_global  = dataset.indices[i]

                   
                        z_local = dataset.z_to_local[z_global]         # local stack index (0..4)

                        z_level = float(dataset.depths[z_global])      # physical depth in meters

                        # ---------------------------------------------------------
                        # MLD distance for the TARGET / FORECAST timestep
                        # ---------------------------------------------------------
                        target_t_idx = t_idx + cfg.data.window.input_steps

                        mld_distance_np = dataset.mld_distance[
                            target_t_idx,
                            z_local
                        ].copy()              
                  

                        # MLD distance was normalized in data.py:
                        # normalized = (raw - mean) / std
                        # Undo that normalization here.
                        mld_distance_np = (
                            mld_distance_np * stats["mld_distance_std"]
                            + stats["mld_distance_mean"]
                        )

                        print(
                            f"📏 MLD distance DENORM: "
                            f"min={np.nanmin(mld_distance_np):.2f} m, "
                            f"max={np.nanmax(mld_distance_np):.2f} m"
                        )
 
                        if i % 1 == 0:
                            inputs, targets = inputs.to(device), targets.to(device)
                            with torch.no_grad():
                                prof.start("predict_forward")
                                preds = model(inputs)
                                prof.stop()


                                # Check for NaNs in preds or targets
                                if torch.isnan(preds).any() or torch.isnan(targets).any():
                            
                            
                                    preds = torch.nan_to_num(preds, nan=0.0)
                                               
                            if i % 10 == 0:  # first sample only
                                #visualize_activation_maps(model, inputs, i, test_region)
                                pass
                                #activation_maps = get_activation_maps(model, inputs)

                                #act = activation_maps["convlstm_hidden"]

                                #save_pred_activation_compare(preds, act, targets, i, test_region, "convlstm_hidden", dataset)
                    
                            if i == 0:
                                print("[PREDICT] TRAIN PRIOR NP =====")
                                print("targets shape, dim:", targets.shape, targets.dim())
                                print("preds shape, dim:", preds.shape, preds.dim())
                                print("mask shape, dim:", mask.shape, mask.dim())

                                print("==========================\n")

                    

                            prof.start("Metrics...")


                            # --- FIX TARGET SHAPE ---
                            if targets.dim() == 5:
                                true_np = targets[0,0,0].cpu().numpy()
                            elif targets.dim() == 4:
                                true_np = targets[0,0].cpu().numpy()
                            elif targets.dim() == 3:
                                true_np = targets[0].cpu().numpy()
                            elif targets.dim() == 2:
                                true_np = targets.cpu().numpy()
                            else:
                                raise ValueError(f"Unexpected target shape: {targets.shape}")

                   
                            # --- FIX TARGET SHAPE ---
                            if preds.dim() == 4:
                                pred_np = preds[0,0].cpu().numpy()
                            elif preds.dim() == 3:
                                pred_np = preds[0].cpu().numpy()
                            elif preds.dim() == 2:
                                pred_np = preds.cpu().numpy()
                            else:
                                raise ValueError(f"Unexpected pred shape: {preds.shape}")


                   
                            # --- FIX MASK SHAPE ---
                            if mask.dim() == 4:
                                mask_np = mask[0,0].cpu().numpy().astype(bool)
                            elif mask.dim() == 3:
                                mask_np = mask[0].cpu().numpy().astype(bool)
                            elif mask.dim() == 2:
                                mask_np = mask.cpu().numpy().astype(bool)
                            else:
                                raise ValueError(f"Unexpected mask shape: {mask.shape}")



                            #z_idx = dataset.z_indices.index(dataset.z_index)
                            #true_np = targets[0,0].cpu().numpy()
                            #pred_np = preds[0,0].cpu().numpy()
                            #mask_np = mask[0,0].cpu().numpy().astype(bool)   # True = ocean valid, False = missing/land





     
                            pred_np = denormalize_tracer(pred_np, stats, z_local)
                            true_np = denormalize_tracer(true_np, stats, z_local)



                            if i == 0 or i == 10:

                                print(f"\n PREDICT POST NP  DEBUG {i} =====")
                                print("true:", true_np.shape)
                                print("preds :", pred_np.shape)
                                print("mask:", mask_np.shape)

                                print("==========================\n")


                                print("\n [PREDIC] SCALE DEBUG")
                                print(f"TRUE: {true_np.min():.2f}, {true_np.max(): .2f}")
                                print(f"PRED: {pred_np.min(): .2f}, {pred_np.max(): .2f}")


                                print("z_local used:", z_local)
                                print("tracer_min:", stats["tracer_min"][z_local])
                                print("tracer_max:", stats["tracer_max"][z_local])


                                print("z_global:", z_global)
                                print("z_local :", z_local)
                                print("stats shape:", stats["tracer_min"].shape)




                            # Mask: True = valid ocean, False = land/missing
                            valid = mask_np

                            if valid.shape != pred_np.shape:
                                if valid.T.shape == pred_np.shape:
                                    valid = valid.T
                                else:
                                    raise ValueError(
                                        f"Mask shape {valid.shape} incompatible with prediction {pred_np.shape}"
                                    )

                            # For visualization, keep NaNs on land:
                            true_vis = true_np.copy()
                            pred_vis = pred_np.copy()
                            true_vis[~valid] = np.nan #non-valid points will be set to NaN as Land pixels
                            pred_vis[~valid] = np.nan
                            mld_distance_vis = mld_distance_np.copy()
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
                            true_valid = true_np[valid]
                            pred_valid = pred_np[valid]

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
                            true_ssim = np.nan_to_num(true_np, nan=0.0)
                            pred_ssim = np.nan_to_num(pred_np, nan=0.0)
                            mask_t    = torch.tensor(valid.astype(np.float32))[None,None,...]

                            _ssim = ssim(
                                torch.tensor(pred_ssim, dtype=torch.float32)[None,None,...] * mask_t,
                                torch.tensor(true_ssim, dtype=torch.float32)[None,None,...] * mask_t,
                            )
                    


                            true_loss = np.nan_to_num(true_np, nan=0.0)
                            pred_loss = np.nan_to_num(pred_np, nan=0.0)

                    
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


                     

                            history["rmse"].append(_rmse)

                            history["mae"].append(_mae)


                            history["rmse_pct"].append(_rmse_pct)

                            history["mae_pct"].append(_mae_pct)


                            history["ssim"].append(_ssim)

                            history["r2"].append(_r2)

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

           


                            prof.start("visualization")


                            img_path = os.path.join(

                                frame_dirs[z_global],

                                f"forecast_t{forecast_step:03d}.png"

                            )

                   

                            local_vmin = 0 #min(true_np.min(), pred_np.min())
                            local_vmax = max(true_np.max(), pred_np.max())


                    
                            visualize_forecast(
                                true_vis,
                                pred_vis,
                                dataset.x_range,
                                dataset.y_range,
                                f"R:{_r2:.2f}|SSIM={_ssim:.2f}|levs: {num_levels}|ep: {num_epochs}|t: {t_idx}| @ {z_str} |inp: {num_inputs} @ {test_region}",
                                img_path,
                                vmin=local_vmin,
                                vmax=local_vmax
                            )


                            image_paths[z_global].append(img_path)


                            str_statistics=f"t{t_idx:4d}_z{z_global:02d}_{z_str}_{model_name}"


                            hist_path = os.path.join(

                            stats_dirs[z_global],

                                f"hist_t{forecast_step:03d}.png"

                            )

                            qq_path = os.path.join(

                                stats_dirs[z_global],

                                f"qq_t{forecast_step:03d}.png"

                            )

                            err_path = os.path.join(

                                stats_dirs[z_global],

                                f"error_t{forecast_step:03d}.png"

                            )

                            plot_histograms(true_np, pred_np, run_name, z_str, hist_path)
                            plot_qq(true_np, pred_np, run_name, z_str, qq_path)
                            plot_error_map(true_np, pred_np, run_name, z_str, err_path)

                            prof.stop()



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

            
                    for z in dataset.valid_z:

            
                        depth = dataset.depths[z]

                        gif_path = os.path.join(
                            output_dir_movies,
                            f"forecast_movie_{run_name}_z{z:02d}_{depth:.0f}m.gif"
                        )

                        animate_forecasts(
                            sorted(image_paths[z]),
                            gif_path
                        )


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

                    prof.report()


                    prof.plot(color="red")


                    end_time = time.time()     # <-- End timer
                    elapsed = (end_time - start_time) / 60  # minutes
        
           
                    print(f"[INFERENCE] Inference time {run_name}: {elapsed:.2f} minutes\n")

            # END REGION LOOP

            print(f"[LOOP] CHECKPOINT or {checkpoint_region}: {elapsed:.2f} minutes\n")   

        # END CHECKPOINT LOOP


        # ==========================================================
        # Cross-region evaluation summary
        # ==========================================================


        prof.start("heatmaps")
        df = pd.DataFrame(cross_results)

        csv_file = "plots/heatmaps/cross_region_metrics.csv"
        df.to_csv(csv_file, index=False)

        print(f"[CROSS METRICS] Saved cross-region metrics to {csv_file}")

        # ----------------------------------------------------------
        # Publication-quality heatmaps
        # ----------------------------------------------------------

        plot_heatmap(df, "SSIM")
        plot_heatmap(df, "R2")
        plot_heatmap(df, "RMSE_pct")
        plot_heatmap(df, "MAE_pct")

        # ----------------------------------------------------------
        # Generalization ranking
        # ----------------------------------------------------------

        generalization_score(df)

        # ----------------------------------------------------------
        # Transfer loss heatmap
        # ----------------------------------------------------------

        plot_transfer_loss(df)




        #layers = ["conv1", "conv2", "convlstm_hidden", "decoder_out"]

        #for layer in layers:
        #    build_gif_for_layer(run_name, layer)

        prof.stop()
        prof.report()


        prof.plot(color="red")

        end_time = time.time()     # <-- End timer
        elapsed = (end_time - start_time_0) / 60  # minutes

        
        print("\n" + "=" * 90)
        print("[FINAL]  Model evaluation finished successfully")
        print(f"[END] Total Inrence Loop {elapsed:.2f}\n")
    
        print("=" * 90 + "\n")
       

    run()
    

if __name__ == "__main__":
    main()

