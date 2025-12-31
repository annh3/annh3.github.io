---
layout: post
title: Lifting the Conceptual Bludgeon from On and Off Policy Reinforcement Learning
published: false
usemathjax: true
tags: reinforcement learning, on policy, off policy, inference, real world application
---

*I started adding deadlifting back to my gym routine too!*

### A Stab at the Definition of On and Off Policy Reinforcement Learning

I wrote this from my heart:

*In the RL paradigm, there is an agent that is trained on the data it produces. Some people call this
learning from trial and error. Sometimes, the data generating distribution (which be produced by the
reference policy, the actor so to speak) is distributionally different than the policy being trained 
(sometimes called the target policy)*

### Why is this bad?

There are many reasons why one would train with the on-policy rl paradigm. One of them is **stable optimization**, meaning
that the policy converges to the intended or optimal distribution, something which importance weighting tries to do, by
re-weighting empirical data so that the estimate of the expected return, part of the optimization target, is computed more
accurately, or is an unbiased estimator. 

The goodness of on-policyness can even be linked to capacity efficiency, i.e. why approximating with low rank adapaters and
reinforcement learning on the same dataset achieves the same TestNLL, or some measure of generalization error, as full capacity
SFT.

### But isn't this kind of ambiguous?

*Let's say that you have an inference and training setup in which the inference server does $B$ episodes/trajectories
then collects the data and does a weight update, then broadcasts the new weights to the inference servers. Then during
the weight update, if the optimization batch size is $b << B$, then essentially off policy RL is happening. The batch
of $B$ episodes/trajectories is basically a replay buffer*

### Btw, here's how to do importance weighting



