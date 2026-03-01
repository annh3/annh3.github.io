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
