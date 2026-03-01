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
* $\frac{d}{dt} \psi_{t}(x_0) = u_t(\psi_{t}(x_0))$ (flow ODE)
* $\psi_{0}(x_0) = x_0$

  
