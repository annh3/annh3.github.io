---
title: ELBO for diffusion models
tags:
  - diffusion
---

I was always told that I needed to know what an ELBO was. Most of the time, I did not see any ELBOs in any of the models that I actually trained. But when I got into diffusion models, that was an area in which the variational lower bound is tied to how the objective is actually derived. So, in this review, we will:

* Derive the ELBO from first principles
* Connect the ELBO to diffusion models

##### Introduction

Modern bayesian statistics is about approximating the posteriors of models which are not easy to compute. **Variational inference*** is a method that approximates difficult to compute probability densities through optimization, rather than through sampling. Let's formalize the general problem of 'modern bayesian statistics'. Consider a joint density of latent variables $z = z_{1:m}$ and observations $x = x_{1:m}$,  $p(z,x) = p(z) p(x | z)$.

I.e., draw the latent variable from a prior density $p(z)$ and relate the latents to the observation through the likelihood $p(x|z)$. Then, inference in a Bayesian model is conditioning on the data and computing the posterior $p(z|x)$. For contrast with variational methods, let's get into the details of a sampling based method for approximating the posterior.

###### MCMC (sampling based) methods for approximate inference

First, construct an ergodic Markov chain on z (i.e. all states of z are visited in the limit of time--irreducibility--and is aperiodic, does not get trapped in repeating cycles), whose stationary distribution is the posterior $p(z|x)$. The Markov chain has transition matrix $P$ so a stationary distribution $\pi$ has property $\pi = \pi P$. In this case, we cannot sample from $p(z|x)$ directly, but we can 'evaluate' the unnormalized posterior $p(z|x) \propto p(x|z)p(z)$. The pointwise evaluator is turned into a sampler. The histogram of the points sampled from this converges to $p(z|x)$. (So in some sense after sampling you can construct a categorical distribution if you want.)

```
Algorithm MCMC

1. from current state z, propose z' from some proposal distribution q(z'|z), e.g. a Gaussian centered at z
2. Compute the acceptance ratio acceptance = min(1, p(x|z')p(z')q(z|z') / 
   p(x|z)p(z)q(z'|z) )
3. Accept z' with probabliblity acceptance, o.w. stay at z
   
# the numerator is the reversal and the denominator is the forward
```

So, What's the issue with MCMC? We need to sample $p(z|x)$ faster than this. Let's see how we can optimize for the posterior. Assume a family $\mathcal{F}$ of approximate densities over the latent variables. Then, find the member of that family $\mathcal{F}$ that minimizes the KL divergence to the exact posterior, i.e., $q^{*}(z) = argmin_{q(z) \in \mathcal{F}} KL(q(z) || p(z|x))$.

###### Here are the quantities we'll be working with.

* $p(z|x) = \dfrac{p(z,x)}{p(x)}$
* $p(x)$, the marginal density of the observations, is called the evidence
* and $p(x) = \int p(z,x) dz$ is intractable

###### Let's get familiar with the evidence lower bound

The variational inference objective is $q^{*}(z) = argmin_{q(z) \in \mathcal{F}} KL(q(z) || p(z|x))$.

Recall the formula for KL divergence :

$D_{KL}(P || Q) = \sum_{x \in X} P(x) \log \frac{P(x)}{Q(x)}$

Now, let's expand the KL divergence.

$$
\begin{align*}
KL(q(z) || p(z|x)) &= \mathbb{E}[\log q(z)] - \mathbb{E}[\log p(z|x)] \\
&= \mathbb{E}[\log q(z)] - \mathbb{E}[\log p(z,x)] + \log p(x) \\
\end{align*}
$$
(We can remove the expectation around $\log p(x)$ as it does not depend on $q(z)$)

Now, let's name this expression the ELBO

$ELBO(q) = \mathbb{E}[\log p(z,x)] - \mathbb{E}[\log q(z)]$

We can see that it lower bounds $\log p(x)$, the evidence.

Here's an alternative way to write the ELBO.

$$
\begin{align*}
ELBO(q) &= \mathbb{E}[\log p(z,x)] - \mathbb{E}[\log q(z)] \\
&= \mathbb{E}[\log p(x|z)] + \mathbb{E}[\log p(z)] - \mathbb{E}[\log q(z)] \\
&= \mathbb{E}[\log p(x|z)] - KL(q(z) || p(z))\\
\end{align*}
$$

### Connection of ELBO to Diffusion Models

Let's map some terms from variational inference to diffusion model terminology.

* $x_0$ the clean latent, is the observation
* $z = x_{1:T}$ are the latents
* $p_{\theta}(x_0 | x_1)$, the step that produces the clean latent is the 'likelihood'
* $p_{\theta}(x_{t-1} | x_t)$ are latent to latent transitions. This can be thought of as an analogy to the prior $p(z)$
* $q(z)$ is the noising process
	* $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$

What is learned in a diffusion model is the entire model. 

* $p_{\theta} = p(X_T) \prod_{t} P(X_{t-1} | X_t)$

Notice that we're optimizing $p$ while $q$ is the fixed Gaussian Markov process. 
#### How is each step of the diffusion process aligned?

Let's start with this form of the ELBO.

$\log p(x) \ge ELBO + KL(q(z) || p(z|x))$

$\log p(x) \ge \mathbb{E}[\log p(x,z)] - \mathbb{E}[\log q(z)] + KL(q(z) || p(z|x))$

$\log p(x_0) \ge \mathbb{E}[\log \dfrac{p(x_{0:T})}{q(x_{1:T} | x_0)}] \coloneqq \mathcal{L}$, is the diffusion version

Per-step generative: $p(x_{0:T} = p(x_T) \prod_{t=1}^T p(x_{t-1}|x_t))$
Per-step forward: $q(x_{t:T}|x_0) = \prod_{t=1}^T q(x_t | x_{t-1})$

When we write $\mathcal{L}$ notice that we can't yet align the $p$ and $q$ transition kernels since they are going in opposite directions

$\mathcal{L} = \mathbb{E}[\log p(x_T) + \sum_{t=1}^T \log p(x_{t-1}| x_t)] - \sum_{t=1}^T \log q(x_t | x_{t-1}) ]$

We can use Baye's rule to invert the direction of $q$. Additionally, the Markov property means that we can condition on $x_0$. We'll subtly factor out $q(x_1 | x_0)$ which will cancel when we do the telescoping sums.

$$
\begin{align*}
\prod_{t=1}^T q(x_t | x_{t-1}) &= q(x_1 | x_0)\prod_{t=2}^T q(x_t | x_{t-1}) \\
&= q(x_1 | x_0) \prod_{t=2}^T (q(x_{t-1} | x_t, x_0) \frac{q(x_t | x_0)}{q(x_{t-1}| x_0)}) \\
&= q(x_1 | x_0) \frac{q(x_t | x_0)}{q(x_1 | x_0)} \prod_{t=2}^T q(x_{t-1} | x_t , x_0)\\
&= q(x_T | x_0) \prod_{t=2}^T q(x_{t-1} | x_t , x_0)\\
\end{align*}
$$


$$
\begin{align*}
\mathcal{L} &= \mathbb{E}[\log p(x_T) + \log p(x_0 | x_1) + \sum_{t=2}^T \log p(x_{t-1}| x_t) - \sum_{t=2}^T \log q(x_t | x_{t-1}) - \log q(x_T | x_0)] \\
&= \mathbb{E}[\log p(x_0 | x_1) + \sum_{t=2}^T D_{KL}(p(x_{t-1}|x) || q(x_t | x_{t-1})) + D_{KL}(p(x_T) || q(x_T | x_0))] \\
\end{align*}
$$
Where the first term is the reconstruction error, the middle term is time-step aligned noise prediction, and the last term is aligning the noise prior.


### Sources

1. https://arxiv.org/abs/1601.00670