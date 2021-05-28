---
layout: post
title: "The Power of Stein's method in high-dimensions"
keywords: Stein's method, machine learning, generative modeling, bayesian inference
---


Computing the distance between two distributions is an important part of
machine learning and statisitcs. For example, if we have a target density
$\mathbb{P}$ with a density, $p(\mathbf{x}) \propto \exp(V(\mathbf{x}))$, known
upto a normalization constant, then we can use _MCMC_ to approximately sample
from $p$.

After running the _MCMC_ chain for a number of steps, we get a set of samples
$(\mathbf{x}^{i})_ {i=1}^N$. If the _MCMC_ chain is unbiased then we know the
sampling is asymptotically exact, that is

$$
Q_{n} \equiv \frac{1}{N}\sum _{i=1}^{N} \mathbf{x}^{i} \Rightarrow \mathbb{P} .
$$

However, if the sampler is not asymptotically exact or we would like to see if
the chain is mixing, then we can measure the distance between $\mathbb{P}$ and
$Q_{n}$. There are several ways to measure the distance between $\mathbb{P},
Q_n$, we mention integral probability metrcs.

An integral probability metric (_IPM_), $D(\mathbb{P}, \mathbb{Q})$, is a
distance between two probability distributions $\mathbb{P}, \mathbb{Q}$. Given
a function class $\mathcal{F}$, we define an _IPM_ as follows

$$
D(\mathbb{P}, \mathbb{Q}) = \sup_{f \in \mathcal{F}}
\left| \E_{\mathbb{P}}f - \E_{\mathbb{Q}}f \right| .
$$

If the function class consists of all functions $f: \mathbb{R}^d \rightarrow
\mathbb{R}$ with Lipschitz constant bounded by $1$, then $D$ is the
Wasserstein-1 distance. If $f$ is bounded by $1$, then $D$ is the total
variation metric. However, in _MCMC_ we do not have exact samples from the
target density $\mathbb{P}$, so computing both expectations is not feasible.
Stein's method provides a solution.

## Stein's Method
[Stein's method](https://arxiv.org/pdf/1109.1880.pdf) is a method introduced by
Charles Stein based on a cute integration trick. Let $p$ be the density of a
standard normal, then for any differentiable function $f$ we have the
following:

$$
\int p(\mathbf{x}) f'(\mathbf{x}) d\mathbf{x} = - \int p'(\mathbf{x}) f(\mathbf{x}) d\mathbf{x} \\
 = \int p(\mathbf{x}) \mathbf{x} f(\mathbf{x}) d\mathbf{x} .
$$

Now, define the Stein operator $A_{p}f (\mathbf{x}) = f'(\mathbf{x}) -
\mathbf{x} f(\mathbf{x})$, then for any differentiable function we have the
following identity:

$$
\E_{\mathbf{P}}[A_{p}f(\mathbf{x})] = 0
$$

This relation is known as Stein's identity, and surprisingly
$\E_{Q}[A_{p}f(\mathbf{x})] = 0$ if and only if $Q$ is the standard normal.
Therefore, if we were to define a function class, the Stein set $\mathcal{G}$,
using the Stein operator, such that if $g \in \mathcal{G}$ then there exists
$f$ such that $g=A_{p}f$. Now, we can define the Stein discrepancy as follows

$$
D_{Stein}(\mathbb{P}, \mathbb{Q}) = \sup_{g \in \mathcal{G}} \left| \E_{\mathbb{P}}g - \E_{\mathbb{Q}}g \right| \\
 = \sup_{g \in \mathcal{G}} \left| \E_{\mathbb{Q}}g \right| ,
$$

here, we only needed samples from one distribution since
$\E_{\mathbb{P}}[A_{p}f(\mathbf{x})] = 0$. Stein's method requires two key
ingredients:
  * A Stein operator $A_p$ for a given density $p$.
  * A Stein set $\mathcal{G}$.

Note, the Stein discrepancy is not an integral probability metric since it is
not symmetric.

## Stein Discrepancy.

Given a density $p$, known upto a normalization constant and with a
differntiable density, and samples from another distribution $Q_n$, we first
define the Stein operator $A_{p}$ as follows

$$
A_{p}f (\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})^{T} f(\mathbf{x}) + f'(x)
$$

where $f: \mathbb{R}^d \rightarrow \mathbb{R}$ is a differentiable function.
Computing the score function $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ does not
require knowing the normalization constant. Now, we can define a Stein set
$\mathcal{G}$.

## Kernel Stein Discrepancy.

## Stein discrepancies in Machine Leaning.

<!-- however they require either samples from both, likelihoods from both or -->
<!-- some expensive optimzation procedure. We detail -->
