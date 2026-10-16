import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Page Config (Compact layout for clean screenshots)
st.set_page_config(page_title="Structural PINN App", page_icon="🏗️", layout="centered")

# Single line clean heading
st.title("Structural PINN: Beam Dashboard")
st.markdown("Solving beam bending ($d^2u/dx^2 = -q$) using **Physics-Informed Neural Networks**.")

# Sidebar Controls
st.sidebar.header("Simulation Parameters")
load_q = st.sidebar.slider("Uniform Load (q)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
epochs = st.sidebar.slider("Training Epochs", min_value=500, max_value=3000, value=1500, step=100)

# Neural Network Architecture
class StructuralPINN(nn.Module):
    def __init__(self):
        super(StructuralPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 35),
            nn.Tanh(),
            nn.Linear(35, 35),
            nn.Tanh(),
            nn.Linear(35, 1)
        )
    def forward(self, x):
        return self.net(x)

# Training Function
def train_model(q_val, num_epochs):
    model = StructuralPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.linspace(0, 1, 100, requires_grad=True).view(-1, 1)
    x_bc = torch.tensor([[0.0], [1.0]], requires_grad=True)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        u = model(x)
        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
        
        physics_loss = torch.mean((d2u_dx2 - (-q_val))**2)
        u_bc = model(x_bc)
        boundary_loss = torch.mean(u_bc**2)
        
        loss = physics_loss + 50.0 * boundary_loss
        loss.backward()
        optimizer.step()
        
    return model

with st.spinner("Training PINN Model..."):
    trained_model = train_model(load_q, epochs)

# Predictions and Analytics
x_test = torch.linspace(0, 1, 200).view(-1, 1)
x_test_np = x_test.detach().numpy().flatten()
u_pred_np = trained_model(x_test).detach().numpy().flatten()
u_exact_np = (load_q / 2.0 * x_test * (1.0 - x_test)).detach().numpy().flatten()

# Calculate All 3 Error Metrics
mse_error = np.mean((u_pred_np - u_exact_np) ** 2)
max_error = np.max(np.abs(u_pred_np - u_exact_np))
l2_relative_error = np.linalg.norm(u_pred_np - u_exact_np) / np.linalg.norm(u_exact_np)

# Display 3 Metrics in Columns
col1, col2, col3 = st.columns(3)
col1.metric("MSE Error", f"{mse_error:.6f}")
col2.metric("Max Error", f"{max_error:.6f}")
col3.metric("L2 Relative", f"{l2_relative_error:.4%}")

# Plotting with Compact Size (Perfect for README screenshots)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(x_test_np, u_exact_np, label=f"Exact Solution (q={load_q})", color="blue", linewidth=2)
ax.plot(x_test_np, u_pred_np, label=f"PINN Deflection (q={load_q})", color="purple", linestyle="dashed", linewidth=2.5)
ax.set_xlabel("Beam Position (x)", fontsize=10)
ax.set_ylabel("Deflection u(x)", fontsize=10)
ax.set_title(f"Beam Deflection Analysis (q={load_q})", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, linestyle=":", alpha=0.7)

st.pyplot(fig)