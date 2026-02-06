import xarray as xr
import matplotlib.pyplot as plt

# Load the prediction output
ds = xr.open_dataset("runs/preds.nc")

# Extract first sample
pred = ds["pred_tracer"].isel(sample=0).squeeze()   # [z,y,x]
true = ds["target_tracer"].isel(sample=0).squeeze() # [z,y,x]

# If z=1, drop it
if "z" in pred.dims and len(pred.z) == 1:
    pred = pred.isel(z=0)
    true = true.isel(z=0)

# Plot side by side
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

pred.plot(ax=axes[0], cmap="viridis")
axes[0].set_title("Forecast tracer")

true.plot(ax=axes[1], cmap="viridis")
axes[1].set_title("True tracer")

(pred - true).plot(ax=axes[2], cmap="bwr", center=0)
axes[2].set_title("Forecast - True")

plt.tight_layout()
plt.show()

