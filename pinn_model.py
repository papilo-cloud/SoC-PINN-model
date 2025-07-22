import torch
import torch.nn as nn
import torch.nn.functional as F


# Simple feedforward PINN model / Neural Network Defination
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, t):
        return self.net(t)


# Physics-informed function loss
def physics_loss(model, t, I, C):
    t.requires_grad = True
    soc_pred = model(t)
    dsoc_dt = torch.autograd.grad(soc_pred, t, torch.ones_like(soc_pred), create_graph=True)[0]
    return F.mse_loss(dsoc_dt, -1 / (3600 * C))

