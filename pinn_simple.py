import torch
import torch.nn as nn
import numpy as np
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
    def forward(self, x):
        return self.net(x)
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
x_physics = torch.linspace(0, 1, 100, requires_grad=True).view(-1, 1)
print("Training PINN model started.....")
for epoch in range(500):
    optimizer.zero_grad()
    u = model(x_physics)
    du_dx = torch.autograd.grad(u, x_physics, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    physics_residual = du_dx - 2.0
    loss = torch.mean(physics_residual**2)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/500], Loss: {loss.item():.6f}")
print("Training completed successfully !")