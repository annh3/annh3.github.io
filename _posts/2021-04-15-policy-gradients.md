---
layout: post
title: Understanding and Implementing Policy Gradients
published: false
usemathjax: true
---

> Topics touched on: 

Policy gradients are a pretty cool class of reinforcement learning algorithms. They focus on directly optimizing the metric we care most about in reinforcement learning (the expected reward from acting in an environment), and because of this enjoy an elegant formulation that looks very similar to supervised maching learning, and has stability benefits over approaches like Q-learning, which indirectly learn a policy, and can suffer from too much [approximation](https://arxiv.org/abs/1812.02648).

While this blog post is in no sense comprehensive, I hope to show a good mixture of theory and practice. There's a lot of [great](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html) [blog](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/) [posts](http://amid.fish/reproducing-deep-rl?fbclid=IwAR1VPZm3FSTrV8BZ4UdFc2ExZy0olusmaewmloTPhpA4QOnHKRI2LLOz3mM) out there on policy gradients and/or deep reinforcement learning, and many go into depth in the mathematical fundamentals or empirical observations. I hope to provide a pedagogically useful blogpost that introduces the concepts, derivations, python implementation examples, empirical observations that help us go from simple Vanilla Policy Gradient, to the more advanced Trust Region Policy Optimization, and then to Proximal Policy Optimization, which is conceptually similar to TRPO but much easier to implement (and thus a popular sota baseline.) 

Credits go to Daniel Takeshi's blog, John Schulman's PhD thesis, OpenAI's Spinning Up, and the Stanford CS234 lectures, from which the explanations in this post are synthesized. I also used the CS234 starter code for VPG and built my implementation of PPO right on top of it.

### Vanilla Policy Gradient

Let's first implement vanilla policy gradient (REINFORCE), the backbone of these three methods. (My acknowledgements go to Daniel Takeshi, since this section is highly based off of his blog post.)

The goal of reinforcement learning is for an agent to learn to act in a dynamic environment so as to maximize its expected cumulative reward over the course of a time-horizon. Policy gradient methods solve the problem of control by directly learning the policy $$\pi: \mathcal{S} \rightarrow \mathcal{A}$$ from observations of rewards obtained by interacting with the environment. 
