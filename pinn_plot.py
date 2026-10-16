import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
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
for epoch in range(500):
    optimizer.zero_grad()
    u = model(x_physics)
    du_dx = torch.autograd.grad(u, x_physics,grad_outputs=torch.ones_like(u), create_graph=True)[0]
    loss = torch.mean((du_dx - 2.0)**2)
    loss.backward()
    optimizer.step()
x_test = torch.linspace(0, 1, 100).view(-1, 1)
u_pred = model(x_test).detach().numpy()
x_test_np = x_test.numpy()
u_exact = 2.0 * x_test_np
plt.figure(figsize=(8, 5))
plt.plot(x_test_np, u_exact, label="Exact Solution (Physics)", color = "blue", linewidth = 2)
plt.plot(x_test_np, u_pred, label ="PINN Prediction", color = "red", linestyle="dashed", linewidth=2)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.title("Physics-Informed Neural Network (PINN) results")
plt.grid(True)
plt.show()