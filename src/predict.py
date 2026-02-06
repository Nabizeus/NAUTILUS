import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from src.model import OceanPredictor
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



def plot_land_mask_predict(mask, region_name, out_dir):
    import matplotlib.pyplot as plt
    import os

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(4, 4))
    plt.imshow(mask.astype(int), cmap="gray")
    plt.title(f"Land–Ocean Mask | {region_name}")
    plt.colorbar(label="1 = Ocean, 0 = Land")
    plt.tight_layout()

    path = os.path.join(out_dir, f"land_mask_{region_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"🗺️ Prediction land mask saved → {path}")



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


def plot_qq(true_dn, pred_dn, region, savepath):
    
    

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
    plt.title(f"Q–Q Plot @{region} ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_error_map(true_dn, pred_dn, region, savepath):
    

    err = pred_dn - true_dn

    plt.figure(figsize=(5,4))
    plt.imshow(err, cmap="coolwarm")
    plt.colorbar(label="Prediction Error (Pred - True)")
    plt.title(f"Spatial Error Map @ {region}")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()





def denormalize(field, stats):
    """Undo per-variable normalization."""
    # field: numpy array (H, W)
    return field * stats["tracer_std"] + stats["tracer_mean"]


def plot_histograms(true_dn, pred_dn, region, savepath):
    plt.figure(figsize=(8,4))
    plt.title(f"Region @ {region}")
    plt.subplot(1,2,1)
    plt.hist(true_dn.flatten(), bins=40, alpha=0.7, label="True")
    plt.title("True Histogram")

    plt.subplot(1,2,2)
    plt.hist(pred_dn.flatten(), bins=40, alpha=0.7, label="Predicted")
    plt.title("Predicted Histogram")

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def denormalize_tracer(x, region_name):
    """
    Denormalize tracer values using either:
    - Min–max scaling (preferred), or
    - Z-score fallback (mean/std) if min/max not present.

    x: normalized tracer array
    region_name: region ID used for loading stats
    """
    stats_file = f"norms/norm_stats_{region_name}.npz"
    stats = np.load(stats_file)

    # Case 1: NEW normalization (min-max)
    if "tracer_min" in stats and "tracer_max" in stats:
        tmin = float(stats["tracer_min"])
        tmax = float(stats["tracer_max"])
        if abs(tmax - tmin) < 1e-12:
            # fallback to z-score if degenerate
            mean = float(stats.get("tracer_mean", 0.0))
            std  = float(stats.get("tracer_std", 1.0))
            return x * std + mean
        return x * (tmax - tmin) + tmin
    else:
        # fallback: old z-score for U V W
        mean = float(stats["tracer_mean"])
        std  = float(stats["tracer_std"])
        return x * std + mean

def load_model_from_checkpoint(cfg, device, region_name, input_channels):
    """Load a model checkpoint robustly, handling multiple naming conventions."""
    model = OceanPredictor(cfg, input_channels=input_channels).to(device)

    checkpoint_dir = getattr(cfg.training, "checkpoint_dir", "./checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, f"{region_name}_best_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ No checkpoint found at {checkpoint_path}")

    print(f"📂 Loading checkpoint for region '{region_name}' → {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # --- Handle multiple formats safely ---
    try:
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("✅ Loaded 'model_state_dict' successfully")
        elif "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            print("✅ Loaded 'model_state' successfully")
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            print("✅ Loaded 'state_dict' successfully")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded raw state_dict directly")
    except Exception as e:
        print(f"⚠️ Could not load checkpoint properly: {e}")
        raise RuntimeError("❌ Checkpoint does not contain valid model weights")

    model.eval()
    print("🧠 Model ready for inference")
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
    print(f"🎞️ Saved animation: {save_gif}")


# For Volume or 3 Layer system
def flatten_depth(inputs):
    B, T, C, Z, H, W = inputs.shape
    return inputs.view(B, T, C * Z, H, W)




def main():
    
    

    @hydra.main(config_path="../config", config_name="default_cnnlstm", version_base=None)
    def run(cfg):
        device = "cpu" 
        start_time_0 = time.time() # <-- Start timer overall

        for region_cfg in cfg.data.regions:
            region_name = region_cfg["name"]
            print(f"\n🧭 Predicting for region: {region_name}")
            start_time = time.time() # <-- Start timer
            dataset = WindowedOceanDataset(cfg, split="test", region_name=region_name)
            input_channels = dataset.n_channels * dataset.n_depths
            print(f"🧩 Dataset 'test' — indices {dataset.t_start}:{dataset.t_end}, total {len(dataset)} samples") 
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
            model_name = cfg.model.name
        


            

            x1, x2 = dataset.x_range
            y1, y2 = dataset.y_range



            # --- Depth 2D slicing ---
            z_idx = getattr(dataset, "z_index", 0)
            z_level = getattr(dataset, "z_level", None)
            mix_level = getattr(dataset, "mix_lay", None)
            
            model = load_model_from_checkpoint(
                cfg,
                device,
                region_name,
                input_channels
            )

            output_dir = os.path.join("predictions", region_name)
            os.makedirs(output_dir, exist_ok=True)
            output_dir_statistics = os.path.join(output_dir, "statistics")
            os.makedirs(output_dir_statistics, exist_ok=True)
            output_dir_frames = os.path.join(output_dir, "frames")
            os.makedirs(output_dir_frames, exist_ok=True)
            output_dir_movies = os.path.join(output_dir, "movies")
            os.makedirs(output_dir_movies, exist_ok=True)




            plot_land_mask_predict(
                dataset.land_mask,
                region_name,
                out_dir=output_dir
            )


            loss_fn = torch.nn.MSELoss(reduction = 'sum')
            total_loss = 0.0
            total_rmse = 0.0
            total_mae = 0.0
            total_ssim = 0.0
            total_r2 = 0.0
             
            image_paths = []
            imagex_paths = []
            all_true, all_pred = [], []


            pbar = tqdm(loader, total=len(loader))


            for i, (inputs, targets, mask) in enumerate(loader):



                if inputs.ndim == 6:
                    inputs = flatten_depth(inputs)


                   
                if i % 1 == 0:
                    inputs, targets = inputs.to(device), targets.to(device)
                    with torch.no_grad():
                        preds = model(inputs)

                        # Check for NaNs in preds or targets
                        if torch.isnan(preds).any() or torch.isnan(targets).any():
                            
                            
                            preds = torch.nan_to_num(preds, nan=0.0)
                                               
                    if i % 5 == 0:  # first sample only
                        visualize_activation_maps(model, inputs, i, region_name)

                        activation_maps = get_activation_maps(model, inputs)

                        act = activation_maps["convlstm_hidden"]

                        save_pred_activation_compare(preds, act, targets, i, region_name, "convlstm_hidden")
                   


                    true_np = targets[0, 0].cpu().numpy()
                    pred_np = preds[0, 0].cpu().numpy()
                    mask_np = mask[0, 0].cpu().numpy().astype(bool)   # True = ocean valid, False = missing/land

                    # --- Load stats for denormalization ---
                    stats = dataset.stats   # ✔ loaded from dataset

                    true_np = denormalize_tracer(true_np, region_name)
                    pred_np = denormalize_tracer(pred_np, region_name)

                    # Mask: True = valid ocean, False = land/missing
                    valid = mask_np                    



                    


                    # For visualization, keep NaNs on land:
                    true_vis = true_np.copy()
                    pred_vis = pred_np.copy()
                    true_vis[~valid] = np.nan #non-valid points will be set to NaN as Land pixels
                    pred_vis[~valid] = np.nan

                    # For metrics, work only on valid pixels
                    true_valid = true_np[valid]
                    pred_valid = pred_np[valid]

                    # --- Metrics (RMSE, MAE, R²) ---
                    _rmse = np.sqrt(np.nanmean((pred_valid - true_valid) ** 2))
                    _mae  = np.nanmean(np.abs(pred_valid - true_valid))

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



                    
                    all_true.append(true_np)
                    all_pred.append(pred_np)

                    depth_str = f"{z_level:.2f} m"
                    img_path = os.path.join(output_dir_frames, f"forecast_{i:03d}.png")

                    local_vmin = 0 #min(true_np.min(), pred_np.min())
                    local_vmax = max(true_np.max(), pred_np.max())


                    
                    visualize_forecast(
                        true_vis,
                        pred_vis,
                        dataset.x_range,
                        dataset.y_range,
                        f"{model_name}|N: {i}| @{depth_str} | mxl @ {mix_level}| @{region_name} "
                        f"|R²:{_r2:.2}  |SSIM={_ssim:.2f}",
                        img_path,
                        vmin=local_vmin,
                        vmax=local_vmax
                    )



                    #visualize_forecast(true_np, pred_np, f"Sample {i}| @z_idx {z_idx}| @depth: {depth_str} | Loss={loss:.4f}", img_path)

                    image_paths.append(img_path)



                    hist_path = os.path.join(output_dir_statistics, f"hist_{i:03d}.png")
                    plot_histograms(true_np, pred_np, region_name, hist_path)

                    # --- Advanced Diagnostics ---
                    #joint_path = os.path.join(output_dir_statistics, f"joint_hist_{i:03d}.png")
                    #plot_joint_histogram(true_np, pred_np, joint_path)

                    qq_path = os.path.join(output_dir_statistics, f"qq_{i:03d}.png")
                    plot_qq(true_np, pred_np, region_name, qq_path)

                    err_path = os.path.join(output_dir_statistics, f"errormap_{i:03d}.png")
                    plot_error_map(true_np, pred_np, region_name, err_path)

            num_samples = len(image_paths)

            avg_rmse = total_rmse / num_samples
            avg_mae  = total_mae  / num_samples
            avg_ssim = total_ssim / num_samples
            avg_r2   = total_r2   / num_samples


            print(f"Final Test Metrics | RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, SSIM: {avg_ssim:.4f}, R²: {avg_r2:.4f}")
            print("✅ Model evaluation finished successfully 🚩")

                        


            # Global color scale for visualization
            #global_vmin = np.min(all_true)
            #global_vmax = np.max(all_true)

            #for i, p in enumerate(image_paths):
            #    true_np, pred_np = all_true[i], all_pred[i]
            #    img_pathx = os.path.join(output_dir_frames, f"forecast_fixed_{i:03d}.png")
                

            #    visualize_forecast(true_np, pred_np, f"Movie {i} @{region_name} | @depth: {depth_str} | @z_idx {z_idx} | RMSE: {total_rmse/n:.2f}, SSIM: {total_ssim/n:.4f}, R²: {total_r2/n:.2f}", img_pathx, vmin=0,vmax=8)
            #    imagex_paths.append(img_pathx)
            
            avg_loss = total_loss / num_samples
            print(f"📉 Average test loss for {region_name}: | {avg_loss:.2f}")

            gif_path = os.path.join(output_dir_movies, f"forecast_movie_{region_name}.gif")
            animate_forecasts(sorted(image_paths), gif_path)
       
            #gif_pathx = os.path.join(output_dir_frames, f"forecast_movie_fix_{region_name}.gif")
            #animate_forecasts(sorted(imagex_paths), gif_pathx)


            test_csv = f"logs/test_metrics_{region_name}.csv"
            with open(test_csv, "a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["LOSS","RMSE", "MAE", "SSIM", "R2"])
                writer.writerow([avg_loss, avg_rmse, avg_mae, avg_ssim, avg_r2])



            z_index = getattr(dataset, "z_index", None)

            write_nemo_restart(
                        pred_field=pred_np,
                        dataset=dataset,
                        region_name=region_name,
                        varname="mytrc",       # or whatever tracer name NEMO expects
                        z_index=z_index,
                        out_dir="nemo_restart"
                    )
            


            layers = ["conv1", "relu1", "conv2", "relu2", "convlstm_hidden", "decoder_out"]

            for layer in layers:
                build_gif_for_layer(region_name, layer)



            end_time = time.time()     # <-- End timer
            elapsed = (end_time - start_time) / 60  # minutes
        
            print(f"�~OInference time for {region_name}: {elapsed:.2f} minutes\n")

        layers = ["conv1", "relu1", "conv2", "relu2", "convlstm_hidden", "decoder_out"]

        for layer in layers:
            build_gif_for_layer(region_name, layer)

 
        print(f"✅ Model evaluation finished successfully 🚀")
        print("\n" + "=" * 90)
        print("🏁✅ FINAL — Model evaluation finished successfully 🚀🌊")
        print("=" * 90 + "\n")
        # 🧩 Final summary message
        print("\n" + "=" * 80)
        print(f"🌊✨ Prediction complete for region: 🌍 {region_name} | @depth: {depth_str}")
        print(f"📊 Output directory: {output_dir}")
        print(f"🎥 Forecast animation: {output_dir}/forecast_movie_{region_name}.gif")
        print(f"🧠 Device: {device}")
        print(f"✅ Model evaluation finished successfully 🚀")

        end_time = time.time()     # <-- End timer
        elapsed = (end_time - start_time_0) / 60  # minutes

        print(f"�~OTotal Inference Time {elapsed:.2f} minutes\n")

        print("=" * 80 + "\n")


    run()
    

if __name__ == "__main__":
    main()

