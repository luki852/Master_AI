## Exercise 1a

Let $\mathbf t\in\mathbb R^d$. Since $\mathbf W$ and $\boldsymbol\mu$ are deterministic, the definition gives
$$
\begin{aligned}
M_{\mathbf W\boldsymbol\xi+\boldsymbol\mu}(\mathbf t)
&=\mathbb E[\exp(\mathbf t^\top(\mathbf W\boldsymbol\xi+\boldsymbol\mu))]\\
&=\mathbb E[\exp(\mathbf t^\top\boldsymbol\mu)\exp(\mathbf t^\top\mathbf W\boldsymbol\xi)]\\
&=\exp(\mathbf t^\top\boldsymbol\mu)\mathbb E[\exp((\mathbf W^\top\mathbf t)^\top\boldsymbol\xi)]\\
&=\boxed{\exp(\mathbf t^\top\boldsymbol\mu)M_{\boldsymbol\xi}(\mathbf W^\top\mathbf t)}.
\end{aligned}
$$
Here $\mathbf W^\top\mathbf t\in\mathbb R^m$, as required for the argument of $M_{\boldsymbol\xi}$. The equality holds wherever the MGF is finite (and also as an equality of nonnegative extended expectations).

## Exercise 1b

Choose independent $\mathbf z\sim\mathcal N(\mathbf0,\mathbf I_m)$ and $\boldsymbol\varepsilon\sim\mathcal N(\mathbf0,\sigma^2\mathbf I_d)$ and set $\mathbf x=\mathbf W\mathbf z+\boldsymbol\mu+\boldsymbol\varepsilon$. Conditional on $\mathbf z$, only the noise is random, so this construction has exactly the prescribed conditional distribution.

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
$$\boxed{p(\mathbf x\mid\mathbf W,\boldsymbol\mu,\sigma^2)=\mathcal N(\mathbf x\mid\boldsymbol\mu,\mathbf W\mathbf W^\top+\sigma^2\mathbf I_d).}$$

## Exercise 2a

Write $\ell(\boldsymbol\mu)$ for the log-likelihood with $\mathbf W,\sigma^2$ fixed. Its first two terms are constant in $\boldsymbol\mu$. Since $\mathbf C^{-1}$ is symmetric, the chain rule gives
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
For every nonzero $\mathbf v$, $\mathbf v^\top(-l\mathbf C^{-1})\mathbf v<0$, since $\mathbf C$ is positive definite. Thus the log-likelihood is strictly concave in $\boldsymbol\mu$, and this stationary point is its unique global maximum.

## Exercise 2b

Using orthonormality, $\mathbf I_d=\sum_{i=1}^d\mathbf u_i\mathbf u_i^\top$. Hence
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
$$

## Exercise 2c

Set $s=\sigma^2>0$, $q=d-m>0$, and $A=\sum_{j=m+1}^d\lambda_j$. Terms independent of $s$ can be collected into a constant $c$:
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

**Degenerate case:** if $A=0$, the likelihood increases without bound as $s\downarrow0$; there is no finite maximizer under $\sigma^2>0$. The displayed estimate is then a boundary value, not an admissible positive variance.
