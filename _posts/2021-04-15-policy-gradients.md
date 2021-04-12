---
layout: post
title: Understanding and Implementing Policy Gradients
published: false
usemathjax: true
---

> Topics touched on: 

Policy gradients are a pretty cool class of reinforcement learning algorithms. They focus on directly optimizing the metric we care most about in reinforcement learning (the expected reward from acting in an environment), and because of this enjoy an elegant formulation that looks very similar to supervised maching learning, and has stability benefits over approaches like Q-learning, which indirectly learn a policy, and can suffer from too much [approximation](https://arxiv.org/abs/1812.02648).

While this blog post is in no sense comprehensive, I hope to show a good mixture of theory, what these algorithms look like in python code, emprical results, and high-level questions. There's a lot a great blog posts out there on policy gradients, and many go into depth in the mathematical fundamentals or empirical observations. I hope to (a) provide technical detail and (b) roughly show the line of thought that a scientist might have that would lead the development from simple Vanilla Policy Gradient, to the more advanced Trust Region Policy Optimization, and then to Proximal Policy Optimization, which is conceptually similar to TRPO but much easier to implement, and thus a popular sota baseline. 
