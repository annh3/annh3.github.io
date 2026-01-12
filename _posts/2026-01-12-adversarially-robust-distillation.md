---
layout: post
title: Towards Efficient Intensely Robust Deep-ish Learning
published: true
usemathjax: true
tags: adversarial robustness, distillation, saddle points, convext optimization
---

*"It is in general that the unexplored attracts us..." - Lady Murasaki*

Let's explain the main equation from the Madry paper and then unfold that into how adversarial training is actually done in practice.

We will lift a concept from convex optimization to help geometrically interpret what is going on with adversarial deep learning.

First, here is the main equation from Madry et al, let's call it the saddle point equation.

#### Saddle Point Equation

$$min_{\theta} \mathbb{E}_{(x,y) \sim \mathcal{D}} [max_{\delta \in \mathcal{S}} L(\theta, x + \delta, y)]$$

Notice that this is a min max problem. Most effective attacks are whtie box, meaning the "attacker" has access to whatever is necessary to differentiate $$L$$ with repsect to the neural network parameters.

But before getting into the details of *inner maximizers*, let's dive into the convex optimization.

#### Strong Duality

Recall that strong duality occurs when the primal function ($$min f_{\theta}(x)$$) is convex and the constraints are convex. Sometimes, there are other routes to proving that a primal function is convex, including Slater's conditions. Sometimes, there is a connection to the primal function's polynomial time computability. But for now, let's just see how strong duality connects to the saddle point equation. 


#### More

1. (Adversarialy Robust Distillation)[https://arxiv.org/abs/1905.09747]
2. (Towards Deep Learning Models Resistant to Adversarial Attacks)[https://arxiv.org/pdf/1706.06083]
3. (Intriguing Properties of Neural Networks)[https://arxiv.org/abs/1312.6199]
4. (Madry and Kolter Adversarial Robustness Tutorial)[https://adversarial-ml-tutorial.org/adversarial_examples/]
5. (Boyd and Vandenberghe Convex Optimization)[https://stanford.edu/~boyd/cvxbook/]
6. (Duality Gap, Computational Complexity and NP Completeness: A Survey)[https://arxiv.org/abs/1012.5568]



