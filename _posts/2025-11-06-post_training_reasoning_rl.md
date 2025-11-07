---
layout: post
title: stochastic parrots of post raining 🌧️ ☔️ post training reinforcement learning reasoning 
published: true
usemathjax: true
tags: rl, llms, foundation models, rollouts, sft
---

What sparked my curiousity? The long reflection tokens from the [m1](https://arxiv.org/pdf/2506.13585) paper which sent me reeling in a chat with my friend. Then I read another paper, the [Qwen3](https://arxiv.org/abs/2505.09388), where they talked about thinking and non thinking mode fusion, and I was like, is that really that difficult? It seems you can mix and match any sequence of domain-specific foundation model rollouts that by appending transition tokens which functionally are similar to conditional logic *and, or, where, etc ...*, fine-tune on those rollouts, rinse and repeat. 

The long reflection tokens from the [m1](https://arxiv.org/pdf/2506.13585), *'However, Recheck, Wait, Aha'* were apparently very important tokens for the reasoning paths for [stabilizing entropy](https://arxiv.org/abs/2505.22617) of the learned policy, which is apparently important for downstream reasoning rl performance. To preserve these tokens, which, due to their low $$\pi_{ref}$$ weight in the denominator of the $$IS$$, which creates a high $$\dfrac{\pi_{cur}}{\pi_{ref}}$$ for the advantage ($$\sum_{i=1}^t r_i - V$$), were clipped out of the PPO (and GSPO and GRPO) updates, the CIPSO objective from the m1 paper simply adds a stop gradient operation, so the high IS = $$\dfrac{\pi_{cur}}{\pi_{ref}}$$ term does not explode the gradient update in the chain rule of backpropagation (I believe the stop_gradient autograd function is implemented in from torch.nn.functional.autograd import F as g(x) = constant, i.e. treat the stop_gradient(g(x)) as g(x) = constant), and tokens (*'However, Recheck, Wait, Aha'*) are assigned credit in the update.

#### PPO Objective Function Minus the KL Term
$$J(\theta) = \mathbb{E}[\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} min(r_{i,t}A_{i,t}, clip(r_{i,t}, 1-\epsilon, 1+\epsilon)A_{i,t}) ]$$

#### GRPO Objective function

$$J(\theta) = \mathbb{E}[\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} min(r_{i,t}A_{i,t}, clip(r_{i,t}, 1-\epsilon, 1+\epsilon)A_{i,t}) ] $$


#### CISPO Objective function
$$J(\theta) = \mathbb{E}\left[\frac{1}{\sum_{i=1}^{G} |o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} sg(r_{i,t})A_{i,t} \log \pi_{\theta}(o_{i,t} \mid q, o_{i,\text{prev}}) M_{i,t} \right]$$

where $$M_{i,t} = 0$$ if $$A_{i,t} > 0$$ and $$r > 1 + \epsilon_{high}$$, $$M_{i,t} = 0$$ if $$A_{i,t} < 0$$ and $$r < 1 + \epsilon_{low}$$ and $$1$$ otherwise. 

The main insight from GSPO and GRPO, the "group part", is that one may approximate the baseline term (V) in an advantage computation $$A = R - V$$ with the average summed return of the group rollouts, the many completions for a fixed prompt, the (Prover, Verifier pairs). 

The main insight from GRPO to CISPO is that transforming the clipping of $$r_{i,t}A_{i,t}$$ into a mask allows one to inspect $$r_{i,t}$$ and $$A_{i,t}$$ and mask them out if both are over or under flow. I assume the 'Aha', 'Wait!' tokens have consistently high $$A_{i,t}$$ but decreasing $$r_{i,t}$$ as the policy learning weights them higher over time.

I shouted to S.Z., why was the entire section about the long reflection tokens in the m1 paper necessary??? I wonder if one studied the long reflection tokens from the m1 papers in Euclidean Space post post-training they would belong in the same subspace. Why am I always becoming an interpretability girlie, against my better judgment?! She said, I wonder about the false positives for the reflection tokens? I said, I can imagine some trigger words. (...) Who would have thought that one could simply inspect conditional probabilities?

The KL term, these days, is left out of most reasoning model objective functions as [policies deviate wildly from the reference policy](https://arxiv.org/pdf/2506.10910) anyway. I shouted to A.L., the rollout completion length penalty in section 2.2.3 of the [magistral](https://arxiv.org/pdf/2506.10910) paper makes no sense! Why not just append a STOP_THINKING token after correct reasoning traces and move this upstream to the Long CoT cold start behavior imitation stage of the post raining post training reasoning rl pipeline? I finally figured out how to do las vegas algorithms with foundation models, I had been thinking forever. Computability theory, my OG.

Anyway the point of this above sections was to point out the similarity between the fork in the road reflection tokens in the reasoning rollouts and the thinking fusion mode from Qwen3, at different layers of abstraction.

#### Entropy in the RL Reasoning Phase

Papers such as [this one](https://arxiv.org/abs/2505.22617) talk about the importance of the entropy of the optimized policy for downstream performance, fitting the equation $R = -a * e^{H} + b$ (H is always between 0 and 1 so $$e^{H}$$ is an convex curve inverted by the $-a$. The downstream evaluations are datasets such as OMNI-BENCH, AIME 2024, et cetera. (Recall that foundation models have a `log_proba` or `proba` function which allows you to compute $H = - p \sum \log p$). 

There are several ways to optimize for entropy of a reasoning RL policy

1. Entropy Bonus in the objective function

I imagine this is adding $- \beta H(x)$ in the objective function.
  
3. $$\epsilon_{high}$$ in the clip function of a GRPO objective

This is what [magistral](https://arxiv.org/pdf/2506.10910) does, they say they that entropy bonus of method 1 causes instability. The basic modification to the GRPO function is to adjust the clipping threshold.

$$J(\theta) = \mathbb{E}[\frac{1}{|o_i|}] \sum_{t=1}^{|o_i|} min(r_{i,t}A_{i,t}, clip(r_{i,t}, 1-\epsilon_{low}, 1+\epsilon_{high})A_{i,t}) $$

$$\epsilon_{high}$$ allows for $$\pi_{cur}$$ to deviate from $$\pi_{ref}$$, which adjusting the $\beta$ term on a KL penalty could do.

   
5. Converting the clip into a stop_gradient and masking in the CISPO objective

The stop_gradient is implement by returning None in the backward pass of an autodifferentiation graph, treating the variable like a constant. The difference with clipping is that the high IS term $$r_{i,t}$$ is part of the loss computation and weights the fork in the road tokens which contribute to high entropy (exploration) in the loss function accordingly. I'm not sure what the m1 paper means by 'clipped_out' as a token, my guess is that the token's `log_proba` needs to be high enough relative to the non exploration tokens. I'm curious about this.

I wonder, how much entropy is too much entropy? I wish the [entropy mechanism paper]((https://arxiv.org/abs/2505.22617)) had just reported on an exact value of $$H$$, if it exists, for the downstream tasks. I suppose the predictive equation is more flexibile as tasks become ever more computationally difficult.


#### KL Distillation

They took out the KL term in the RL phase of the pipeline (recall that the m1 and the magistral and probably a few other papers do not have a KL term in the RL objective)! But KL distillation, ah the good teacher and the professor forcing (so to speak, those are the titles of the papers!) can be used to train lightweight smaller models from rollouts from the heavyweight models (trained through the (pretraining -> Long CoT cold start -> reasoning RL -> thinking mode fusion -> general RL) pipeline). It seems inefficient that one must have `log_proba` access to the heavyweight model logits to compute this update, i.e. to do [logit distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).


*Thank you to SZ for amplifying my inspiration with random words like rollouts and transformers, for the blog in which I learned the term logit distillation and for RV for the open weight papers.*








