# Exercise 1: Gradient of a Quadratic Form

**Assignment 1 · Optimum Signal Processing · WS 2026**

## Given and goal

Let $\mathbf{x}\in\mathbb{R}^N$ and let $\mathbf{A}\in\mathbb{R}^{N\times N}$ be a constant, symmetric matrix, so $A_{ij}=A_{ji}$. For

$$f(\mathbf{x})=\mathbf{x}^{T}\mathbf{A}\mathbf{x},$$

show that $\nabla_{\mathbf{x}}f=2\mathbf{A}\mathbf{x}$. We use a column vector for the gradient.

## (a) Scalar representation

Expanding both matrix products gives

$$\boxed{f(\mathbf{x})=\sum_{i=1}^{N}\sum_{j=1}^{N}x_i A_{ij}x_j.}$$

## (b) Differentiate component by component

Fix any $k\in\{1,\ldots,N\}$. Since $\mathbf{A}$ is constant, apply the product rule to the two factors that depend on $\mathbf{x}$:

$$
\begin{aligned}
\frac{\partial f}{\partial x_k}
&=\sum_{i=1}^{N}\sum_{j=1}^{N} A_{ij}
\frac{\partial(x_i x_j)}{\partial x_k}\\
&=\sum_{i=1}^{N}\sum_{j=1}^{N} A_{ij}
\left(\frac{\partial x_i}{\partial x_k}x_j+x_i\frac{\partial x_j}{\partial x_k}\right)\\
&=\sum_{i=1}^{N}\sum_{j=1}^{N} A_{ij}
\left(\delta_{ik}x_j+x_i\delta_{jk}\right)\\
&=\sum_{j=1}^{N}A_{kj}x_j+\sum_{i=1}^{N}A_{ik}x_i.
\end{aligned}
$$

Here $\delta_{ik}$ is the Kronecker delta: it is 1 if $i=k$ and 0 otherwise. In particular, the product rule counts a diagonal term $A_{kk}x_k^2$ twice, as required.

By symmetry, $A_{ik}=A_{ki}$. Renaming the summation index in the second sum therefore yields

$$
\frac{\partial f}{\partial x_k}
=\sum_{j=1}^{N}A_{kj}x_j+\sum_{j=1}^{N}A_{kj}x_j
=2\sum_{j=1}^{N}A_{kj}x_j
=2(\mathbf{A}\mathbf{x})_k.
$$

Stacking these derivatives for every $k$ proves

$$\boxed{\nabla_{\mathbf{x}}f(\mathbf{x})=2\mathbf{A}\mathbf{x}.}$$

**Check:** Without symmetry, the same calculation gives $(\mathbf{A}+\mathbf{A}^{T})\mathbf{x}$, which reduces to the stated result when $\mathbf{A}=\mathbf{A}^{T}$.
