---
layout: post
title: Diffusion and Flow Intro
published: false
usemathjax: true
tags: diffusion models, flow models, SDEs, ODEs
---


All diffusion and flow processes are about a process of destroying an image or data point through noise then learning to recover the image by modeling the destruction process, whether by modeling the gradient of the probability distribution as it evolves over time or by directly modeling the noise that is added at each step of the destruction process. There are a few mathematical structures that are helpful in understanding diffusion/flow--those being ODEs, SDEs, ELBOS, and Markov Processes, and we will go over all of these, but I found I can mostly ignore the concept of ELBOs and Markov Processes for diffusion/flow models when I have a good understanding of the theory of ODEs and SDEs. 

#### Definition of ODE

An ODE is deterministic and its solution is a *trajectory* or *path*,  in other words a function of the form:

$$X: [0,1] \rightarrow \mathbb{R}^d, t \rightarrow X_t$$

Every ODE is defined by a *vector field*, i.e.:

$$u: \mathbb{R}^d \times [0,1] \rightarrow \mathbb{R}^d, (x,t) \rightarrow u_{t}(x)$$

Such that 

$$\frac{d}{dt} X_t = u_t(X_t)$$

and 

$$X_0 = x_0$$

It helps to think of ODEs in terms of physical concepts. The vector field is basically the velocity of a data point (mass). If we want to be overly formal, the *flow* is the actual solution to the ODE and is defined as 

* $\psi: \mathbb{R}^d \times [0,1] \rightarrow \mathbb{R}^d$, $(x_0,t) \rightarrow \psi_t (x_0)$
* $\frac{d}{dt} \psi_{t}(x_0) = u_t(\psi_{t}(x_0))$ *(flow ODE)*
* $\psi_{0}(x_0) = x_0$

#### Definition of SDE

An SDE is stochastic. We can extend the ODE conceptual framework to SDEs by adding stochastic dynamics driven by Brownian motion. But first, what is Brownian motion?

##### Interlude into Brownian Motion

For those of you who are familiar with a random walk (imagine a graph where the where the edge transitions are probabilities and a particle flows along this graph), a Brownian motion can be thought of as a continuous random walk. Formally, a Brownian motion $W = (W_t)_{0 \le t \le 1}$ is a stochastic process such that $W_0 = 0$, the trajectories $t \rightarrow W_t$ are continuous and two conditions hold:

1. **Normal Increments**, meaning $W_t - W_s \sim \mathcal{N}(0, (t-s)I_d)$. So that the Brownian Motion path is a Gaussian with variance increasing linearly in time. 
2. **Independent Increments**. For $0 \le t_0 < t_1 < ... < t_n = 1$, $W_{t_1 - t_0}, ..., W_{t_{n} - t_{n-1}}$ are independent random variables.

![brownian motion](https://i.pinimg.com/736x/fc/da/69/fcda69f901d68216be9d83a01d39a413.jpg)

We basically amend the ODE by adding a Brownian motion increment. Since everything is now stochastic we cannot simply take the derivative of the vector field and must rewrite $\frac{d}{dt} X_t = u_{t}(X_t)$ as $h(X_{t+h} - X_t) = u_t(X_t) + R_t(h)$ where the error term $R_t(h)$ accounts for the approximation. The expression for the next position of the data point becomes:

$$X_{t+h} = X_t + h \cdot u_t(X_t) + h \cdot R_t(h)$$

To which we add the Brownian increment

$$X_{t+h} = X_t + h \cdot u_t(X_t) + \sigma_t (W_{t+h} - W_t) + h \cdot R_t(h)$$

When the time step goes to zero, the error term becomes negligible, so we can rewrite the SDE as

$$dX_t = u_t(X_t)dt + \sigma_t dW_t$$

$$X_0 = x_0$$

(Note: I'm actually confused about the derivation here since somehow making the error term explicit and then sending the timestep to 0 recovers the derivative TODO: actually step into the analysis of why the error term goes to zero, since the technique is to make this explicit so then everything can be re-written in derivative form.)

You can imagine an ODE as transporting a datapoint while an SDE transports or evolves data distributions. So any stochasticity in an ODE comes from the initial sampling of a clean image $X_0$ or noise $X_t$. Once the data point is sampled, the path it moves along to become noise is determined. For an SDE, on the other hand, once a data point $X_0$ is sampled, you can consider its path to be a path of evolving probability distributions.

#### Simulating ODES and SDEs

To get a better intuition for ODEs and SDEs, let's simulate them. For SDEs, the path is defined deterministically by the vector field $u(X,t)$ and stochastically by $\sigma(t)$. To simulate an SDE we sample $\epsilon_{t} \sim \mathcal{N}(0, I_d)$ and compute

$$X_{t+h} = X_t + h u_t(X_t) + \sqrt{h} \sigma(t) \epsilon_t, \epsilon_t \sim \mathcal{N}(0,I_d)$$

Here's an example of an SDE with a linear vector field, $u_(x,t) = -x-4$ and a linear brownian drift coefficient $\sigma(t) = 0.8t + 0.1$. 

<video width="100%" controls>
  <source src="[https://annh3.github.io](https://github.com/annh3/annh3.github.io/raw/refs/heads/gh-pages/assets/simple_sde.mp4)" type="video/mp4">
  Your browser does not support the video tag.
</video>


https://github.com/user-attachments/assets/3745af23-1356-41ca-8b0f-ec95462e9a0e


And here's an example of the ODE version, i.e. the same vector field with no brownian motion.

<video width="100%" controls>
  <source src="[https://annh3.github.io](https://github.com/annh3/annh3.github.io/raw/refs/heads/gh-pages/assets/simple_ode.mp4)" type="video/mp4">
  Your browser does not support the video tag.
</video>


https://github.com/user-attachments/assets/769518c4-2146-435d-b74d-51e1236c6928


Let's go through two examples on two different toy datasets of different diffusion approaches, learning to estimate noise via DDPM and learning to estimate score via Denoising Score Matching, then take a step back to work through the theory of deriving the training targets from the expressions for the path probabilities.

TODO: train a version with flow modeling 

Then, we'll talk about how to modulate image generation through a text prompt, and see what this looks like in practice in a multimodal architecture.

### Example 1 - Learning a Mixture of Gaussians with Denoising Score Matching

First, let's consider the mixture of gaussians defined by $x \sim \frac{1}{4} \mathcal{N}((-5,-5),I) + \frac{3}{4} \mathcal{N}((5,5),I)$. You can see from the plot that the data mass at $(-5,-5)$ is lighter than the mass at $(5,5)$.

![mixture of gaussians](https://i.pinimg.com/736x/c6/cc/98/c6cc984065e48fe2f133075380509d66.jpg)

For a mixture of gaussians, the score is nonlinear, but for one gaussian, parameterizing the score network as $W \cdot x + b$ is sufficient.

Here's the proof. (TODO: write proof here.)

We can estimate the score of the mixture of gaussians with a simple MLP, $s_{\theta}(x) = W_2 \cdot \sigma(W_1 \cdot x + b_1) + b_2$ where $\sigma$ is the softplus operator. 

### Denoising Objective

Define $q_{\sigma}(\tilde{x} | x)$ as the operator which applies gaussian noise with standard deviation $\sigma$ to the data point such that $q_{\sigma}(\tilde{x} | x) = \mathcal{N}(x, \sigma^2)$. 

When the noise is small, $q_{\sigma}(x) \approx p_{data}(x)$ so that $s_{\theta^{*}} = \nabla_{x} \log p_{data}(x) \approx \nabla_{x} \log q_{\sigma}(x)$.

So the denoising objective is 

$$\frac{1}{2} \mathbb{E}_{q_{\sigma(\tilde{x}|x)p_{data}(x)}} [\| s_{\theta}(\tilde{x}) - \nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x} | x)\|_2^{2}]$$

And $\tilde{x} = x + \sigma z$ where $z \sim \mathcal{N}(x, \sigma^2)$ is the input to the score network. 

We'll need an algebraic expression for the score, though another objective, the sliced score objective, torch's autograd function can be deployed in the objective function. 

Since $q(\tilde{x}) = (2 \pi)^{D/2} det(\Sigma)^{-1/2} exp(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu))$,

$$
\begin{align*}
\log q(\tilde{x}) &= - \frac{D}{2} \log(2 \pi) - \frac{1}{2} \log \det (\Sigma) + [-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)] \\
\nabla_{\tilde{x}} \log q(\tilde(x)) &= \nabla_{\tilde{x}} [-\frac{1}{2}(\tilde{x} - x)^T \frac{1}{\sigma^2} I (\tilde{x} - x)] \\
&= \nabla_{\tilde{x}} [-\frac{1}{2 \sigma^2}(\tilde{x} - x)^T(\tilde{x} - x)] \\
&= - \frac{(\tilde{x} - x)}{\sigma^2} \\
&= - \frac{z}{\sigma^2} \\
\end{align*}
$$

But before looking at the experiment runs for this, let's understand the motivation behind adding noise. Theorem 2 from Estimation of Non-Normalized Statistical Models by Score Matching says that 
