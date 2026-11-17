import json
from pathlib import Path

p = Path('3rd_Semester/UE_Advanced_Machine_Learning/A3/A3/A3_exercise.ipynb')
n = json.loads(p.read_text(encoding='utf-8'))
solutions = {
6: r'''Let $\mathbf t\in\mathbb R^d$. Since $\mathbf W$ and $\boldsymbol\mu$ are deterministic, the definition gives
$$
\begin{aligned}
M_{\mathbf W\boldsymbol\xi+\boldsymbol\mu}(\mathbf t)
&=\mathbb E[\exp(\mathbf t^\top(\mathbf W\boldsymbol\xi+\boldsymbol\mu))]\\
&=\mathbb E[\exp(\mathbf t^\top\boldsymbol\mu)\exp(\mathbf t^\top\mathbf W\boldsymbol\xi)]\\
&=\exp(\mathbf t^\top\boldsymbol\mu)\mathbb E[\exp((\mathbf W^\top\mathbf t)^\top\boldsymbol\xi)]\\
&=\boxed{\exp(\mathbf t^\top\boldsymbol\mu)M_{\boldsymbol\xi}(\mathbf W^\top\mathbf t)}.
\end{aligned}
$$
Here $\mathbf W^\top\mathbf t\in\mathbb R^m$, as required for the argument of $M_{\boldsymbol\xi}$. The equality holds wherever the MGF is finite (and also as an equality of nonnegative extended expectations).''',
9: r'''Choose independent $\mathbf z\sim\mathcal N(\mathbf0,\mathbf I_m)$ and $\boldsymbol\varepsilon\sim\mathcal N(\mathbf0,\sigma^2\mathbf I_d)$ and set $\mathbf x=\mathbf W\mathbf z+\boldsymbol\mu+\boldsymbol\varepsilon$. Conditional on $\mathbf z$, only the noise is random, so this construction has exactly the prescribed conditional distribution.

By 1a) and the Gaussian MGF formula,
$$
M_{\mathbf W\mathbf z+\boldsymbol\mu}(\mathbf t)
=e^{\mathbf t^\top\boldsymbol\mu}e^{\frac12(\mathbf W^\top\mathbf t)^\top(\mathbf W^\top\mathbf t)}
=e^{\mathbf t^\top\boldsymbol\mu+\frac12\mathbf t^\top\mathbf W\mathbf W^\top\mathbf t}.
$$
Independence allows us to factor the expectation:
$$
\begin{aligned}
M_{\mathbf x}(\mathbf t)
&=\mathbb E[e^{\mathbf t^\top(\mathbf W\mathbf z+\boldsymbol\mu)}e^{\mathbf t^\top\boldsymbol\varepsilon}]\\
&=M_{\mathbf W\mathbf z+\boldsymbol\mu}(\mathbf t)M_{\boldsymbol\varepsilon}(\mathbf t)\\
&=e^{\mathbf t^\top\boldsymbol\mu+\frac12\mathbf t^\top\mathbf W\mathbf W^\top\mathbf t}
 e^{\frac12\sigma^2\mathbf t^\top\mathbf t}\\
&=\exp\!\left(\mathbf t^\top\boldsymbol\mu+\frac12\mathbf t^\top(\mathbf W\mathbf W^\top+\sigma^2\mathbf I_d)\mathbf t\right).
\end{aligned}
$$
This is the MGF of $\mathcal N(\boldsymbol\mu,\mathbf W\mathbf W^\top+\sigma^2\mathbf I_d)$. Gaussian MGFs exist in a neighborhood of zero and uniquely determine the distribution. Therefore
$$\boxed{p(\mathbf x\mid\mathbf W,\boldsymbol\mu,\sigma^2)=\mathcal N(\mathbf x\mid\boldsymbol\mu,\mathbf W\mathbf W^\top+\sigma^2\mathbf I_d).}$$''',
13: r'''Write $\ell(\boldsymbol\mu)$ for the log-likelihood with $\mathbf W,\sigma^2$ fixed. Its first two terms are constant in $\boldsymbol\mu$. Since $\mathbf C^{-1}$ is symmetric, the chain rule gives
$$
\nabla_{\boldsymbol\mu}[(\mathbf x_i-\boldsymbol\mu)^\top\mathbf C^{-1}(\mathbf x_i-\boldsymbol\mu)]
=-2\mathbf C^{-1}(\mathbf x_i-\boldsymbol\mu).
$$
Consequently,
$$
\begin{aligned}
\nabla_{\boldsymbol\mu}\ell
&=\sum_{i=1}^l\mathbf C^{-1}(\mathbf x_i-\boldsymbol\mu)
=\mathbf C^{-1}\left(\sum_{i=1}^l\mathbf x_i-l\boldsymbol\mu\right),\\
\nabla_{\boldsymbol\mu}\ell=\mathbf0
&\iff\sum_{i=1}^l\mathbf x_i-l\boldsymbol\mu=\mathbf0
\iff\boxed{\hat{\boldsymbol\mu}=\frac1l\sum_{i=1}^l\mathbf x_i}.
\end{aligned}
$$
The equivalence uses invertibility of $\mathbf C$. The Hessian is
$$\nabla^2_{\boldsymbol\mu}\ell=-l\mathbf C^{-1}.$$
For every nonzero $\mathbf v$, $\mathbf v^\top(-l\mathbf C^{-1})\mathbf v<0$, since $\mathbf C$ is positive definite. Thus the log-likelihood is strictly concave in $\boldsymbol\mu$, and this stationary point is its unique global maximum.''',
18: r'''Using orthonormality, $\mathbf I_d=\sum_{i=1}^d\mathbf u_i\mathbf u_i^\top$. Hence
$$
\begin{aligned}
\mathbf C
&=\mathbf U_m(\mathbf L_m-\sigma^2\mathbf I_m)\mathbf U_m^\top+\sigma^2\mathbf I_d\\
&=\sum_{i=1}^m(\lambda_i-\sigma^2)\mathbf u_i\mathbf u_i^\top
+\sum_{i=1}^d\sigma^2\mathbf u_i\mathbf u_i^\top\\
&=\sum_{i=1}^m\lambda_i\mathbf u_i\mathbf u_i^\top
+\sum_{i=m+1}^d\sigma^2\mathbf u_i\mathbf u_i^\top
=\boxed{\mathbf U\mathbf K\mathbf U^\top}.
\end{aligned}
$$
Thus $k_i=\lambda_i$ for $i\le m$ and $k_i=\sigma^2$ otherwise. Since $\mathbf U^\top\mathbf U=\mathbf I_d$ and $(\det\mathbf U)^2=1$,
$$
|\mathbf C|=|\mathbf U||\mathbf K||\mathbf U^\top|=|\mathbf K|
=\left(\prod_{i=1}^m\lambda_i\right)(\sigma^2)^{d-m},
$$
so
$$\log|\mathbf C|=\sum_{i=1}^m\log\lambda_i+(d-m)\log\sigma^2.$$
Also $\mathbf C^{-1}=\mathbf U\mathbf K^{-1}\mathbf U^\top$. With $\mathbf S=\mathbf U\mathbf L\mathbf U^\top$ and cyclic invariance of trace,
$$
\begin{aligned}
\operatorname{tr}(\mathbf C^{-1}\mathbf S)
&=\operatorname{tr}(\mathbf U\mathbf K^{-1}\mathbf L\mathbf U^\top)
=\operatorname{tr}(\mathbf K^{-1}\mathbf L)\\
&=\sum_{i=1}^d\frac{\lambda_i}{k_i}
=\sum_{i=1}^m1+\frac1{\sigma^2}\sum_{j=m+1}^d\lambda_j
=m+\frac1{\sigma^2}\sum_{j=m+1}^d\lambda_j.
\end{aligned}
$$
Substitution into $\ell=-\frac l2[d\log(2\pi)+\log|\mathbf C|+\operatorname{tr}(\mathbf C^{-1}\mathbf S)]$ yields
$$
\boxed{\ell=-\frac l2\left[d\log(2\pi)+\sum_{i=1}^m\log\lambda_i+(d-m)\log\sigma^2+m+\frac1{\sigma^2}\sum_{j=m+1}^d\lambda_j\right].}
$$''',
21: r'''Set $s=\sigma^2>0$, $q=d-m>0$, and $A=\sum_{j=m+1}^d\lambda_j$. Terms independent of $s$ can be collected into a constant $c$:
$$\ell(s)=c-\frac l2\left(q\log s+\frac A s\right).$$
Differentiate with respect to the variance $s$ (not its square root):
$$
\begin{aligned}
\ell'(s)&=-\frac l2\left(\frac q s-\frac A{s^2}\right)
=\frac l{2s^2}(A-qs),\\
\ell'(s)=0&\iff A-qs=0\iff
\boxed{\hat\sigma^2=\frac Aq=\frac1{d-m}\sum_{j=m+1}^d\lambda_j}.
\end{aligned}
$$
If $A>0$, then $\ell'(s)>0$ for $s<A/q$ and $\ell'(s)<0$ for $s>A/q$. Therefore this is the global maximum; alternatively,
$$\ell''(s)=\frac{lq}{2s^2}-\frac{lA}{s^3},\qquad
\ell''(A/q)=-\frac{lq}{2(A/q)^2}<0.$$
Because the eigenvalues are sorted, $A/q\le\lambda_m$, so the solution is compatible with the real square root in $\hat{\mathbf W}$ (strict inequality gives full rank). Each discarded direction contains only isotropic noise in the model, so the estimate averages their empirical variances.

**Degenerate case:** if $A=0$, the likelihood increases without bound as $s\downarrow0$; there is no finite maximizer under $\sigma^2>0$. The displayed estimate is then a boundary value, not an admissible positive variance.''',
34: '''# Estimate the mean using one observation per row.
mu_hat = X.mean(axis=0)
# Subtract the estimated mean from every observation.
X_centered = X - mu_hat
# Use the maximum-likelihood covariance normalization 1/l, not 1/(l-1).
S = X_centered.T @ X_centered / X.shape[0]
# Compute the eigenvalues and orthonormal eigenvectors of the symmetric covariance.
eigenvalues, eigenvectors = np.linalg.eigh(S)
# Obtain indices in descending eigenvalue order.
order = np.argsort(eigenvalues)[::-1]
# Sort the eigenvalues from largest to smallest.
eigenvalues = eigenvalues[order]
# Apply the same ordering to the eigenvector columns.
eigenvectors = eigenvectors[:, order]
# Estimate the noise variance from the discarded eigenvalues.
sigma2_hat = eigenvalues[m:].mean()
# Construct the loading matrix with R = I; clipping removes possible roundoff errors.
W_hat = eigenvectors[:, :m] @ np.diag(np.sqrt(np.maximum(eigenvalues[:m] - sigma2_hat, 0.0)))
# Compare the estimated and generating means.
print("Estimated / true mean:", mu_hat, mu_true)
# Compare variances with variances.
print("Estimated / true noise variance:", sigma2_hat, sigma_true ** 2)
# Compare standard deviations with standard deviations.
print("Estimated / true noise standard deviation:", np.sqrt(sigma2_hat), sigma_true)
# Display the estimated and generating loading matrices.
print("Estimated W:\\n", W_hat, "\\nTrue W:\\n", W_true)
# Quantify agreement of the principal directions independently of their signs.
print("Absolute alignment with classical PCA:", abs(np.dot(W_hat[:, 0] / np.linalg.norm(W_hat[:, 0]), pca.components_[0])))
''',
35: r'''The estimates of $\boldsymbol\mu$ and $\sigma^2$ should be close to the generating values $(1,2,3)^\top$ and $0.0625$, with sampling error because only 1000 observations are available. Compare $\sqrt{\hat\sigma^2}$ with `sigma_true`, which is a standard deviation.

The loading matrix is not identifiable: replacing $\mathbf W$ by $\mathbf W\mathbf R$ for an orthogonal $\mathbf R$ leaves $\mathbf W\mathbf W^\top$, and therefore the entire observed distribution, unchanged. For $m=1$, this means that $\mathbf W$ and $-\mathbf W$ are equally valid; an opposite sign does not indicate a bad fit. Finite-sample fluctuations can additionally change its length and direction slightly.

For $m=1$,
$$\hat{\mathbf W}=\mathbf u_1\sqrt{\lambda_1-\hat\sigma^2}.$$
Thus the loading vector is parallel to classical PCA's first principal direction, up to sign, but its length measures signal variation after subtracting the noise variance. `scikit-learn` uses $1/(l-1)$ for its reported eigenvalues, whereas our maximum-likelihood covariance uses $1/l$; this does not change the eigenvectors.''',
38: '''# Form the latent-space matrix using the estimated parameters.
M_hat = W_hat.T @ W_hat + sigma2_hat * np.eye(m)
# Solve for all posterior means and transpose back to one sample per row.
Z_hat = np.linalg.solve(M_hat, W_hat.T @ (X - mu_hat).T).T
# Decode the posterior means without adding fresh observation noise.
X_reconstructed = Z_hat @ W_hat.T + mu_hat
# Create a three-dimensional comparison plot.
fig = plt.figure(figsize=(9, 8))
# Add a three-dimensional axis.
ax = fig.add_subplot(111, projection='3d')
# Show the noisy observations with low opacity.
ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.15, label='Original observations')
# Show the reconstructions on the estimated affine principal line.
ax.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], X_reconstructed[:, 2], s=12, label='Posterior-mean reconstructions')
# Label the first coordinate.
ax.set_xlabel('$x_1$')
# Label the second coordinate.
ax.set_ylabel('$x_2$')
# Label the third coordinate.
ax.set_zlabel('$x_3$')
# Explain that posterior means shrink noisy observations toward the mean.
ax.set_title('pPCA: noisy observations and denoised reconstructions')
# Identify the two sets of points.
ax.legend()
# Render the figure.
plt.show()
''',
62: '''# Disable gradient tracking because encoding does not update the trained parameters.
with torch.no_grad():
    # Form the 2 x 2 posterior matrix on the same device and with the same dtype as W.
    M = W.T @ W + logsigma2.exp() * torch.eye(W.shape[1], dtype=W.dtype, device=W.device)
    # Solve M Z = W^T (X - mu) to obtain one encoding per column.
    image_encodings = torch.linalg.solve(M, W.T @ (X - mu_t))
''',
67: '''Consecutive frames should map to nearby positions because breathing changes the images gradually, while similar breathing phases should return to similar regions of the latent space. The first 25 color-coded points can therefore trace a curved or approximately cyclic trajectory corresponding to inhalation and exhalation; a perfectly closed circle is not required because breathing varies and pPCA uses a linear two-dimensional representation.''',
70: '''# Generate samples without retaining an autograd graph.
@torch.no_grad()
# Accept the trained parameters and the requested number of images.
def sample_ppca(W, mu, logsigma2, n_samples=16):
    """Return samples of shape (n_samples, d) from the learned pPCA distribution."""
    # Draw independent standard-normal latent vectors with matching dtype and device.
    z = torch.randn(n_samples, W.shape[1], dtype=W.dtype, device=W.device)
    # Convert the log variance to the noise standard deviation and draw independent noise.
    eps = torch.exp(0.5 * logsigma2) * torch.randn(n_samples, W.shape[0], dtype=W.dtype, device=W.device)
    # Decode each latent vector, add the learned mean, and add observation noise.
    return z @ W.T + mu + eps
''',
73: '''The model can reproduce the broad anatomical layout and dominant image variations, but a two-dimensional linear representation cannot capture all fine details or nonlinear motion. Independent isotropic Gaussian observation noise adds grain, and latent samples away from the observed encodings can produce implausible anatomy or intensity patterns. Quality could improve by increasing the latent dimension (chosen using held-out data), using a richer noise model, or learning a nonlinear generative model; displaying the noiseless decoder mean removes grain but is no longer a full sample from the fitted observation distribution.''',
}
for i, s in solutions.items():
    n['cells'][i]['source'] = s.splitlines(keepends=True)
    if n['cells'][i]['cell_type'] == 'code':
        n['cells'][i]['outputs'] = []
        n['cells'][i]['execution_count'] = None
s = ''.join(n['cells'][54]['source'])
s = s.replace('    ########## YOUR SOLUTION HERE ##########\n    # j = ...\n    # return j', '    # Combine the log-determinant and the sample-averaged quadratic term.\n    j = (d - m) * logsigma2 + Mlogdet + summed / (l * sigma2)\n    # Return the differentiable objective for backpropagation.\n    return j')
n['cells'][54]['source'] = s.splitlines(keepends=True)
p.write_text(json.dumps(n, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
# Provide the derivations as plain Markdown as well as editable notebook cells.
out = p.with_name('A3_analytical_solutions.md')
out.write_text('\n\n'.join('## Exercise ' + title + '\n\n' + solutions[i] for i, title in [(6,'1a'),(9,'1b'),(13,'2a'),(18,'2b'),(21,'2c')])+'\n', encoding='utf-8')
print('Filled all solution cells; exported analytical Markdown.')
