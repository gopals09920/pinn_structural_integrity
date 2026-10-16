import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
class StructuralPINN(nn.Module):
    def __init__(self):
        super(StructuralPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 1)
        )
    def forward(self, x):
        return self.net(x)
model = StructuralPINN()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
x = torch.linspace(0, 1, 200, requires_grad=True).view(-1, 1)
x_bc = torch.tensor([[0.0], [1.0]], requires_grad=True)
print("Training Structural Integrity PINN(2000 Epochs)...")
for epoch in range(2000):
    optimizer.zero_grad()
    u = model(x)
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
    physics_loss = torch.mean((d2u_dx2 - (-1.0))**2)
    u_bc = model(x_bc)
    boundary_loss = torch.mean(u_bc**2)
    loss = physics_loss + 20.0 * boundary_loss
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 300 == 0:
        print(f"Epoch [{epoch+1}/1500], Loss: {loss.item():.6f}")
print("Structural PINN Training Completed! ")
x_test = torch.linspace(0, 1, 200).view(-1, 1)
u_pred = model(x_test).detach().numpy()
x_test_np = x_test.detach().numpy().flatten()
u_pred_np = model(x_test).detach().numpy().flatten()
u_exact_np = (0.5 * x_test * (1.0 - x_test)).detach().numpy().flatten()
plt.figure(figsize=(8, 5))
plt.plot(x_test_np, u_exact_np, label = "Exact Analytical Solution", color = "blue", linewidth=2)
plt.plot(x_test_np, u_pred_np, label="PINN Structural Deflection", color = "purple", linestyle="dashed", linewidth=2.5)
plt.xlabel("Beam Position (x)")
plt.ylabel("Deflection / Stress u(x)")
plt.title("Structural Integrity PINN: Beam Bending under Load")
plt.legend()
plt.grid(True)
plt.show()