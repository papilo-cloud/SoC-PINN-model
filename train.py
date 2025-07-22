from utils import battery_data
import torch
from pinn_model import PINN, physics_loss


# Training loop / Train the model
def train(csv_path, C_value, epochs=3000, lr=1e-3):
    (t, I, _, _, C) = battery_data(csv_path, C_value)
    
    model = PINN()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_phys = physics_loss(model, t, I, C)
        loss_phys.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Physics Loss = {loss_phys.item():.6f}")
    
    return model