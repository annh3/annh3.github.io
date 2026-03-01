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

