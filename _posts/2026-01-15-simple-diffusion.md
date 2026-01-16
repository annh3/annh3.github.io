---
layout: post
title: Simple Example of Stable Diffusion
published: false
usemathjax: true
tags: diffusion, gaussian process, simple example
---

Let's get an intuition for diffusion models through a very simple example. I like to contrast diffusion modeling with autoregressive modeling for sequences. In autoregressive modeling, prediction for the next token $x_t$ scaffolds on previous predictions $x_{t-1}, x_{t-2}, ..., x_{0}$. In diffusion modeling, the stochastic process that is modeled is sequential noise corruption of a pure data source and the reverse process of it. By learning this process, a diffusion model can generate new data points drawn from the same distribution as the training data.

The corrupting noise can be drawn from any distribution but to simplify, let's consider a Gaussian Markov chain, characterized as

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_{t}} x_{t-1}, \beta_{t}I)$$
$$x_t = \sqrt{1 - \beta_{t}} x_{t-1} + \beta_{t}\epsilon$$

Before diving into an example implementation, let's go through the theory. The diffusion process happens for a fixed number of steps $T$. In my mind, there are two parts to this, learning the diffusion model, and generating from the diffusion model. 

Learning the diffusion model is about learning a predictor for the noise added to corrupt a datapoint. I.e., given $x_t$ and $t$, learn a predictor of the noise added to create $x_t$ from $x_0$. So the u_net learns $\epsilon(x_t,t)$, the noise corruption from $x_0$. Since $q$ the forward noise process is modelled analytically, the training loop lookes like this--sample $x_0$ (raw, pure data points), sample $t$ the timestep uniformly at random, compute the corrupted datapoint $x_t$ analytically, pass $(x_t, t)$ through the forward u_net to predict the error, backpropogate on the prediction error.

So, the u_net forward pass learns to predict, given a noise corrupted data point and the time step it is, the exact noise added to the pure image. What can we do with such a predictor?

Before we get into that, let's slow down to zoom in on the noise process. 

For this process $(\beta_t)_{t \in T}$, which is the isotropic variance coefficient, is a sequence which controls the speed of data destruction. For this example, we'll do a linear schedule from $1e-4$ to $1e-2$ which is typical. Then the sequence $(\alpha_t)_{t \in T}$ is formed by $\alpha_i = 1 - \beta_i$ and then the cumulative product of the sequence is taken to create $\bar{\alpha}$ so in particular the noise coefficient $\bar{\alpha_t}$ which multiplies the signal component, i.e. the mean, becomes exponentially smaller as the process goes on, so the datapoint destruction accelerates.

Now that we've reviewed the noise schedule for the diffusion process, let's understand the generation process. 
