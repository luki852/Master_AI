# Exercise 2: The Variance Identity

**Assignment 1 · Optimum Signal Processing · WS 2026**

## Goal and assumption

For a real random variable $X$ with finite second moment $E[X^2]<\infty$, prove

$$\operatorname{Var}(X)=E[X^2]-(E[X])^2.$$

## Derivation

Write $\mu=E[X]$. By definition, variance is the expected squared deviation from the mean:

$$\operatorname{Var}(X)=E[(X-\mu)^2].$$

Expand the square and use linearity of expectation:

$$
\begin{aligned}
\operatorname{Var}(X)
&=E[X^2-2\mu X+\mu^2]\\
&=E[X^2]-2\mu E[X]+E[\mu^2]\\
&=E[X^2]-2\mu^2+\mu^2\\
&=E[X^2]-\mu^2.
\end{aligned}
$$

The third line uses $E[X]=\mu$ and the fact that $\mu^2$ is a constant, so $E[\mu^2]=\mu^2$. Thus

$$\boxed{\operatorname{Var}(X)=E[X^2]-(E[X])^2.}$$

**Interpretation:** Variance equals the second raw moment minus the square of the mean. The expressions $E[X^2]$ and $(E[X])^2$ are generally different.
