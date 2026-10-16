import torch
x = torch.tensor([2.0], requires_grad=True)
y = x**3
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print("Input x:", x.item())
print("Output y (x^3):", y.item())
print("Derivative dy/dx (3x^2):", dy_dx.item())
print("Enviourment aur PyTorch autograd successfully working! ")