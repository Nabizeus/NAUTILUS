import torch
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import numpy as np
from src.model import OceanPredictor
from src.data import WindowedOceanDataset
from src.utils import load_config
from matplotlib.colors import Normalize


def get_global_vmin_vmax(image_paths):
    """Compute global min/max across all saved forecast frames."""
    vmin, vmax = float('inf'), float('-inf')
    for path in image_paths:
        img = plt.imread(path)
        vmin = min(vmin, np.nanmin(img))
        vmax = max(vmax, np.nanmax(img))
    return vmin, vmax






def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find the latest saved checkpoint file."""
    files = sorted(
        glob.glob(os.path.join(checkpoint_dir, "*.pt")),
        key=os.path.getmtime,
        reverse=True,
    )
    return files[0] if files else None


def load_model_from_checkpoint(cfg, device, region_name):
    """Load model weights from latest available checkpoint."""
    

    checkpoint_dir = cfg.training.checkpoint_dir
    checkpoint_path = f"{checkpoint_dir}/{region_name}_best_model.pt"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # ✅ Get saved params (H, W, input_channels, etc.)
    H = checkpoint.get("H", None)
    W = checkpoint.get("W", None)

    print(f"📂 Loading checkpoint for '{region_name}' → {checkpoint_path}")
    print(f"✅ Model saved with H={H}, W={W}")

    # ✅ Build model with those exact dimensions
    model = OceanPredictor(cfg, H=H, W=W).to(device)

    # ✅ Now load weights
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        raise KeyError("Checkpoint is missing 'model_state' key!")

    model.eval()
    print(f"Successfully loaded model for regio {region_name}")
    return model


    # ✅ Check existence before loading
    #if not os.path.exists(checkpoint_path):
    #    raise FileNotFoundError(f"❌ No checkpoint found at {checkpoint_path}")

    #print(f"📂 Loading checkpoint for region '{region_name}' → {checkpoint_path}")


    #checkpoint = torch.load(checkpoint_path, map_location=device)



    # ✅ Case 1: checkpoint is a full dictionary
    #if isinstance(checkpoint, dict):

    #    # ✅ Try different common formats
    #    if "model_state_dict" in checkpoint:
    #        model.load_state_dict(checkpoint["model_state_dict"])
    #    elif "state_dict" in checkpoint:
    #        model.load_state_dict(checkpoint["state_dict"])
    #    elif "model_state" in checkpoint:  # ✅ Used by many training loops
    #        model.load_state_dict(checkpoint["model_state"])
    #    else:
    #        # Fallback — assume the checkpoint is just raw state_dict
    #        try:
    #            model.load_state_dict(checkpoint)
    #        except Exception as e:
    #            raise KeyError(
    #                f"❌ Checkpoint format unrecognized — expected 'model_state_dict', "
    #                f"'state_dict', or raw weights. Error: {e}"
    #                )
    #else:
    #    # ✅ Directly assume it's a state_dict
    #    model.load_state_dict(checkpoint)

 
    #model.eval()
    #print(f"✅ Successfully loaded model for region '{region_name}'")
    #return model

def visualize_forecast(true, pred, title="Forecast vs True", save_path=None): #, vmin=None, vmax=None):
    """Visualize a single depth slice comparison."""
    #norm = Normalize(vmin=vmin, vmax=vmax)


    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(true, cmap="viridis") #, norm=norm )
    plt.title("True")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(pred, cmap="viridis") #, norm=norm)
    plt.title("Predicted")
    plt.colorbar()

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def animate_forecasts(image_paths, save_gif=None): #vmin=None, vmax=None):
    """Create an animated GIF from saved PNGs."""
    if not image_paths:
        print("⚠️ No forecast images found to animate.")
        return

    # Step 1: Compute consistent color limits
    #vmin, vmax = get_global_vmin_vmax(image_paths)
    #norm = Normalize(vmin=vmin, vmax=vmax)

    print(f"🎬 Generating animation with {len(image_paths)} frames") #(vmin={vmin:.3f}, vmax={vmax:.3f})")
    
    
    fig = plt.figure(figsize=(8, 4))
    ims = []

    for path in image_paths:
        img = plt.imread(path)
        im = plt.imshow(img)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
    ani.save(save_gif, writer="pillow")
    plt.close(fig)
    print(f"✅ Animation saved to {save_gif}")


def main():
    cfg = load_config("config/default.yaml")
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    print(f"🔍 Running predictions on device: {device}")

    # Load dataset and model
    for region_cfg in cfg.data.regions:
        region_name = region_cfg["name"]
        print("region_name: ",region_name)
        print(f"\n�~_~L~M Srt  PREDICTION for region: {region_name}")

        dataset = WindowedOceanDataset(cfg, split="test", region_name=region_name)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        
        #region_name = cfg.data.regions[0].name if not hasattr(cfg.data, "region_name") else cfg.data.region_name
        model = load_model_from_checkpoint(cfg, device, region_name)

        

        output_dir = "predictions"
        os.makedirs(output_dir, exist_ok=True)

        loss_fn = torch.nn.MSELoss()
        total_loss = 0.0
        image_paths = []
        #  Before plotting any frames:
        #all_true_pred = []  # store arrays to compute vmin/vmax



        for i, (inputs, targets, mask) in enumerate(tqdm(loader, desc="Predicting")):
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
            print("i: ",i)
            #print("inputs ",inputs)
            #print("targets",targets)
            #print("mask ",mask)


            print(f"mask.shape: {mask.shape}")
            print(f"inputs.shape: {inputs.shape}")
 
            print(f"mask.ndim: {mask.ndim}")
            print(f"inputs.ndim: {inputs.ndim}")

 
            # --- Depth-averaged 2D slicing ---
            z_idx = getattr(dataset, "z_index", 0)
            #print(f"@predict.py - Line 107 - z_idx: {z_idx}")

            #if inputs.ndim == 6:
            #    inputs_2d = inputs[...,z_idx , :, :] # middle of the z-range or depth levels
            #    print(f"🔹 Using z-index {z_idx} for prediction")
            #else:
            #    inputs_2d = inputs

            print(f"targets.ndim: {targets.ndim}")
            print(f"inputs.ndim: {inputs.ndim}")

            print(f"targets.shape: {targets.shape}")
            print(f"inputs.shape: {inputs.shape}")


            #Successfully loaded model for regio south_pacific
            #mask.shape: torch.Size([1, 1, 30, 30])
            #inputs.shape: torch.Size([1, 9, 4, 30, 30])
            #mask.ndim: 4
            #inputs.ndim: 5
            #targets.ndim: 4
            #inputs_2d.ndim: 5
            #targets.shape: torch.Size([1, 1, 30, 30])
            #inputs_2d.shape: torch.Size([1, 9, 4, 30, 30])

 
            plt.imshow(targets[0,0].cpu(), cmap='viridis')
            plt.title(f"First Target Slice {region_name} @z_idx {z_idx}")
            plt.colorbar()
            plt.savefig(f"artifacts/first_target_slice_{region_name}.png")
            plt.close()


            plt.imshow(inputs[0,8,2].cpu(), cmap='viridis')
            plt.title(f"[0,4th/9,0th/4, 30, 30] Input Slice {region_name} @z_idx {z_idx}")
            plt.colorbar()
            plt.savefig(f"artifacts/first_input_slice_{region_name}.png")
            plt.close()

            

            t = 3
            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            feature_names = ["Tracer", "U", "V", "W"]  # adapt if needed

            for c in range(4):
                ax = axes[c]
                im = ax.imshow(inputs[0, t, c].cpu(), cmap='viridis')
                ax.set_title(f"{feature_names[c]} (t={t})")
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.savefig(f"artifacts/four_time3_input_slice_{region_name}.png")
        




            with torch.no_grad():
            

                preds = model(inputs.to(device), debug=False)

                print(f"preds.ndim: {preds.ndim}")
                print(f"preds.shape: {preds.shape}")

                


                plt.imshow(preds[0,0].cpu(), cmap='viridis')
                plt.title(f"First Prediction Slice {region_name}")
                plt.colorbar()
                plt.savefig(f"artifacts/first_prediction_slice_{region_name}.png")
                plt.close()


                #preds = model(inputs)

                # Apply mask before computing loss
                #masked_pred = preds * mask
                #masked_target = targets * mask
                #loss = torch.mean(((preds - targets)[mask.bool()]) ** 2).item() #For large tensor multipl. 3D grids
                
                #if torch.isnan(preds).any() or torch.isnan(targets).any():
                #    print("⚠️ NaNs detected in preds or targets!")
                #    print(f"preds NaN count: {torch.isnan(preds).sum().item()}")
                #    print(f"targets NaN count: {torch.isnan(targets).sum().item()}")

                #if torch.all(targets == 0):
                #    print("⚠️ All targets are zero — likely empty mask region")

                #print("@predict.py - line 278 - (preds - targets) ** 2 ",(preds - targets) ** 2)
                #print("@predict.py - line 278 - (preds - targets) ** 2 ",(preds - targets) ** 2)

                loss = torch.mean((preds - targets) ** 2).item()

                print("@predict.py - line 278 - loss = mean(pred-targets)**2",loss)
                #loss = torch.mean((masked_pred - masked_target) ** 2).item()
                total_loss += loss
                

                #all_true_pred.append(masked_target.cpu().numpy())
                #all_true_pred.append(masked_pred.cpu().numpy())

    
                #loss = loss_fn(preds, targets).item()
                #total_loss += loss

            # Visualization for first few samples
            if i < 2000:  # save first 2000 examples
                pred_np = preds[0, 0].detach().cpu().numpy()
                true_np = targets[0, 0].detach().cpu().numpy()
                img_path = f"{output_dir}/forecast_{i:03d}_{region_name}.png"
            
            
                # --- Reduce 3D data to 2D for visualization ---
                z_idx = getattr(dataset, "z_index", 0)
                print(f"true_np.ndim: {true_np.ndim}")
                print(f"pred_np.ndim: {pred_np.ndim}")

                print(f"true_np.shape: {true_np.shape}")
                print(f"pred_np.shape: {pred_np.shape}")

                #if true_np.ndim == 3:
                #    true_np = np.nanmean(true_np[z_idx, :, :], axis=0)  # middle of the z-range for test
                #if pred_np.ndim == 3:
                #    pred_np = np.nanmean(pred_np[z_idx, :, :], axis=0)
                


                # 2. Compute fixed min/max
                #global_vmin = np.min(all_true_pred)
                #global_vmax = np.max(all_true_pred)        


                visualize_forecast(
                    true_np,
                    pred_np,
                    title=f"Region:{region_name} | Sample {i} | Loss: {loss:.2f}",
                    save_path=img_path)#,
                    #vmin=None, vmax=None)
                image_paths.append(img_path)

        # ✅ After the loop — compute fixed global min/max
        #all_true_pred = np.array(all_true_pred)
        #global_vmin = np.nanmin(all_true_pred)
        #global_vmax = np.nanmax(all_true_pred)

        #print(f"🌍 Global vmin={global_vmin:.4f}, vmax={global_vmax:.4f}")

        # ✅ Re-render saved images using consistent scaling
        #for path in image_paths:
        #    img = plt.imread(path)   # reload saved PNG
        #    plt.imshow(img, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
        #    plt.axis('off')
        #    plt.savefig(path, dpi=200)
        #    plt.close()


        avg_loss = total_loss / len(loader)
        print(f"✅ Average test loss: {avg_loss:.2f}")

        # Generate animation from saved forecasts
        animate_forecasts(
                sorted(image_paths),
                save_gif=os.path.join(output_dir, f"forecast_movie_{region_name}.gif")
                )
                #vmin=global_vmin,
                #vmax=global_vmax
        #)

        

if __name__ == "__main__":
    main()

