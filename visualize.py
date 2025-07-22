import matplotlib.pyplot as plt
import torch
import numpy as np
from utils import battery_data

    
def visualize(model, csv_path, C_value=2.3):
    (t, I, t_np, current, C) = battery_data(csv_path, C_value)
    
    # Coulomg counting reference (for sanity check only, not used in training)
    dt = np.diff(np.insert(t_np, 0, 0))
    soc_coulomb = 1.0 - np.cumsum(current * dt) / (3600 * C.item())
    
    # True SoC for evaluation (linear decrease)
    true_soc_np = 1.0 - (current * t_np) / (C.item() * 3600)
    true_soc = torch.tensor(true_soc_np, dtype=torch.float32).view(-1, 1)

    # Evaluate / PINN prediction
    model.eval()
    with torch.no_grad():
        # soc_pred = model(t).detach().numpy()
        soc_pred = model(t).cpu().numpy().flatten()
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(t_np / 60, true_soc, '--', label="True SoC", linewidth=2)
    plt.plot(t_np / 60, soc_coulomb, label="Coulomb Count (Reference)", alpha=0.6)
    plt.plot(t_np / 60, soc_pred, '--', label="PINN Predicted SoC (Unsupervised)", linewidth=2)
    plt.xlabel("Time (minutes)")
    plt.ylabel("State of Charge")
    plt.title("SoC Estimation via Physics-Informed Neural Network")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()