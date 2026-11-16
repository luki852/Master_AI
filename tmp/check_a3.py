import json
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(2)
n = json.loads(Path('3rd_Semester/UE_Advanced_Machine_Learning/A3/A3_exercise.ipynb').read_text(encoding='utf-8'))
scope = {'np': np, 'torch': torch}
exec(''.join(n['cells'][54]['source']), scope)
scope.update(d=5, m=2)
torch.manual_seed(21)
X = torch.randn(7, 5, dtype=torch.float64)
W = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
mu = torch.randn(1, 5, dtype=torch.float64, requires_grad=True)
logvar = torch.tensor([-0.3], dtype=torch.float64, requires_grad=True)
efficient = scope['cost_function'](X, W, mu, logvar)
C = W @ W.T + logvar.exp() * torch.eye(5, dtype=torch.float64)
centered = X - mu
direct = torch.logdet(C) + (centered * torch.linalg.solve(C, centered.T).T).sum() / len(X)
torch.testing.assert_close(efficient.squeeze(), direct)
g1 = torch.autograd.grad(efficient, (W, mu, logvar), retain_graph=True)
g2 = torch.autograd.grad(direct, (W, mu, logvar))
for a, b in zip(g1, g2):
    torch.testing.assert_close(a, b)
exec(''.join(n['cells'][70]['source']), scope)
samples = scope['sample_ppca'](W, mu, logvar, 100000)
torch.testing.assert_close(samples.mean(0), mu.detach().squeeze(), atol=0.025, rtol=0)
torch.testing.assert_close(torch.cov(samples.T), C.detach(), atol=0.065, rtol=0.025)
assert samples.shape == (100000, 5) and not samples.requires_grad
print('Efficient cost and gradients match direct covariance formula; generated sample moments match model.')
