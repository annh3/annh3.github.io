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

Formally define a trajectory $$\tau$$ as a tuple $$(s_0, a_0, r_0, s_1, a_1, ..., r_T)$$ denoting a sequence of state-action-rewards observed over the course of some episode of interaction with the agent's environment, and let $$R(\tau)$$ denote the finite-horizon return, aka cumulative sum of rewards over finite timesteps. Then our goal is the maximize the _expected_ finite-horizon return where the expectation is over trajectories sampled from the stochastic policy $$\pi_{\theta}$$ (here we let $$\theta$$ denote the parameters of the policy $$\pi$$)--i.e.

$$ max_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}} [R(\tau)] $$

In order to optimize $$J$$ using gradient ascent, we need an efficiently computable form of its gradient. First, let's compute an analytical form of the gradient, then see how we can approximate it with sampling.

We use the log-derivative trick to push a form of the gradient of the policy into the the gradient of $$J$$.

I liked Daniel Takeshi's explanation of it--"the log derivative trick tells us how to insert a log into an expectation when starting from $$\nabla_{\theta} \mathbb{E}[f(x)]$$"

$$\begin{align*}
\nabla_{\theta} \mathbb{E}[f(x)] &= \nabla_{\theta} \int p_{\theta}(x) f(x) dx \\
&= \int \dfrac{p_{\theta}(x)}{p_{\theta}(x)} \nabla_{\theta} p_{\theta}(x) f(x) dx \\
&= \int p_{\theta}(x) \nabla_{\theta} \log p_{\theta}(x) f(x) dx \\
&= \mathbb{E}[\nabla_{\theta} \log p_{\theta}(x) f(x)]
\end{align*}$$

In the gradient of $$J$$ we are concerned with the gradient of the log probability of trajectories, so we derive its form now.

$$\begin{align*}
\nabla_{\theta} \log p_{\theta}(\tau) &= \nabla_{\theta} \log \left( \mu(s_{0}) \prod_{t=0}^{T-1} \pi_{\theta}(a_t|s_t) P(s_{t+1|s_t,a_t}) \right) \\
&= \nabla_{\theta} \left( \log \mu_{s_0} + \sum_{t=0}^{T-1} (\log \pi_{\theta}(a_t | s_t) + \log P(s_{t+1} | s_t, a_t))\right) \\
&= \nabla_{\theta} \sum_{t=0}^{T-1} \log \pi_{\theta}(a_t | s_t) \\
\end{align*}$$

Note that the dynamics of the environment $$P$$ disappears, as it does not depend on $$\theta$$, which shows that policy gradients can be used in a model-free manner.

Putting the pieces together, the gradient of the expected return is

$$\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}} [ R(\tau) \sum_{t=0}^T \nabla_{\theta} \log \pi_{\theta}(a_t|s_t)]$$

The log probability of each action is weighted by the rewards associated with it. The gradient has an intuitive interpretation--it encourages us to increase the probability of actions which lead to high expected return.

Conveniently, $$\nabla_{\theta} J$$ turns out to have the form of an expectation. Because of this, as long as we can take the gradient of the log of our policy, we can estimate it with [Monte Carlo simulation](https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/monte-carlo-methods-mathematical-foundations/expected-value) using samples from our environment--i.e.

$$\hat{g} = \dfrac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^T \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) R(\tau)$$

Where $$\mathcal{D}$$ is a dataset of trajectories.

### Policy Network

The policy is represented as a multi-layer perceptron, since we are interested in acting in high dimensional state spaces with continuous actions. (On an aside, a less utilitarian and very beautiful thing about deep RL is that it connects neuroscience with artificial intelligence. [Here's](https://deepmind.com/blog/article/Dopamine-and-temporal-difference-learning-A-fruitful-relationship-between-neuroscience-and-AI) recent work on connections between TD-learning and dopamine in animal learning.)

```
def build_mlp(input_size, output_size, n_layers, size):    
    modules = OrderedDict()
    modules['Linear_Input'] = nn.Linear(input_size, size)
    modules['ReLU_Input'] = nn.ReLU()
    for i in range(n_layers):
        modules['Linear_'+str(i)] = nn.Linear(size, size)
        modules['ReLU_'+str(i)] = nn.ReLU()
    modules['Linear_Output'] = nn.Linear(size,output_size)
    sequential = nn.Sequential(modules)
    return sequential

policy_network = build_mlp(observation_dim, 1, n_layers, layer_size)
```

Below is the implementation of the policy gradient update. Note here that advantage denotes $\sum_{t' = t}^T r_{t'}- V^{\pi}(s_t)$

```
def update_policy(self, observations, actions, advantages):

        self.optimizer.zero_grad()
        res = self.policy.action_distribution(observations).log_prob(actions) 

        loss = -(res * advantages).mean()
        loss.backward()
        self.optimizer.step()
```
### Variance
