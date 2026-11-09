# Exercise 4: Covariance and Correlation for an Affine Relationship

**Assignment 1 · Optimum Signal Processing · WS 2026**

## Given and assumptions

Let $Y=aX+b$, where $a,b\in\mathbb{R}$ are constants and $E[X]=0$. Assume $E[X^2]<\infty$, and write $\sigma_X^2=\operatorname{Var}(X)$.

## 1. Means and centered variables

By linearity of expectation,

$$\mu_X=0,\qquad \mu_Y=E[aX+b]=aE[X]+b=b.$$

Hence

$$X-\mu_X=X,\qquad Y-\mu_Y=aX.$$

## 2. Covariance

Use the definition of covariance:

$$
\begin{aligned}
\operatorname{Cov}(X,Y)
&=E[(X-\mu_X)(Y-\mu_Y)]\\
&=E[X(aX)]\\
&=aE[X^2]\\
&=a\sigma_X^2.
\end{aligned}
$$

The last equality follows from $E[X]=0$. Therefore,

$$\boxed{\operatorname{Cov}(X,Y)=a\operatorname{Var}(X).}$$

## 3. Variance and standard deviation of $Y$

$$
\begin{aligned}
\sigma_Y^2
&=E[(Y-\mu_Y)^2]\\
&=E[(aX)^2]\\
&=a^2 E[X^2]
=a^2\sigma_X^2.
\end{aligned}
$$

Taking the nonnegative square root gives

$$\sigma_Y=|a|\sigma_X.$$

The absolute value matters, particularly when $a<0$.

## 4. Correlation coefficient

For $\sigma_X^2>0$ and $a\ne0$, both standard deviations are positive, so

$$
\rho_{X,Y}
=\frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}
=\frac{a\sigma_X^2}{\sigma_X|a|\sigma_X}
=\frac{a}{|a|}.
$$

Thus

$$\boxed{
\rho_{X,Y}=\begin{cases}
+1,&a>0,\\
-1,&a<0.
\end{cases}}
$$

| Condition | Covariance | Correlation | Interpretation |
|:---|:---|:---|:---|
| $a>0$, $\sigma_X^2>0$ | $a\sigma_X^2>0$ | $+1$ | Perfect positive linear correlation |
| $a<0$, $\sigma_X^2>0$ | $a\sigma_X^2<0$ | $-1$ | Perfect negative linear correlation |
| $a=0$ | $0$ | Undefined | $Y=b$ is constant |
| $\sigma_X^2=0$ | $0$ | Undefined | Both variables are constant almost surely |

**Role of the offset:** The constant $b$ changes the mean of $Y$, but disappears when centering. It therefore has no effect on covariance or correlation. In fact, the covariance and correlation formulas remain valid for any finite mean of $X$, since $Y-E[Y]=a(X-E[X])$.
