---
layout: post
title: stochastic parrots of post raining 🌧️ ☔️ post training reinforcement learning reasoning
published: false
usemathjax: true
tags: rl, llms, foundation models, rollouts, sft
---

What sparked my curiousity? The long reflection tokens from the [m1](https://arxiv.org/pdf/2506.13585) paper which sent me reeling on a chat with my friend. Then I read another paper, the [Qwen3](https://arxiv.org/abs/2505.09388), where they talked about thinking and non thinking mode fusion, and I was like, is that really that difficult? It seems you can mix and match any sequence of domain-specific foundation model rollouts that by appending transition tokens which functionally are similar to conditional logic *and, or, where, etc ...*, fine-tune on those rollouts, rinse and repeat. 

The long reflection tokens from the [m1](https://arxiv.org/pdf/2506.13585), *'However, Recheck, Wait, Aha'* were apparently very important tokens for the reasoning paths for [stabilizing entropy](https://arxiv.org/abs/2505.22617) of the learned policy. I'll assume, also, that avoiding entropy collapse is a good prior for the correctness of reasoning paths. To preserve these tokens, which, due to their low $$\pi_{ref}$$ weight in the denominator of the $$IS$$, which creates a high $$\dfrac{\pi_{cur}}{\pi_{ref}}$$ for the advantage ($$\sum_{i=1}^t r_i - V$$), were clipped out of the PPO (and GSPO and GRPO) updates, the CIPSO objective from the m1 paper simply adds a stop gradient operation, so the high IS = $$\dfrac{\pi_{cur}}{\pi_{ref}}$$ term does not explode the gradient update in the chain rule of backpropagation (I believe the stop_gradient autograd function is implemented in from torch.nn.functional.autograd import F as g(x) = constant, i.e. treat the stop_gradient(g(x)) as g(x) = constant), and tokens (*'However, Recheck, Wait, Aha'*) are assigned credit in the update.

#### PPO Objective Function Minus the KL Term
$$J(\theta) = \mathbb{E}[\frac{1}{|o_i|}] \sum_{t=1}^{|o_i|} min(r_{i,t}A_{i,t}, clip(r_{i,t}, 1-\epsilon, 1+\epsilon)A_{i,t}) $$

#### CISPO Objective function
$$J(\theta) = \mathbb{E}\left[\frac{1}{\sum_{i=1}^{G} |o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} sg(r_{i,t})A_{i,t} \log \pi_{\theta}(o_{i,t} \mid q, o_{i,\text{prev}})\right]$$

The main insight from GSPO and GRPO, the "group part", is that one may approximate the baseline term (V) in an advantage computation $$A = R - V$$ with the average summed return of the group rollouts, the many completions for a fixed prompt, the (Prover, Verifier pairs). 

I shouted to S.Z., why was the entire section about the long reflection tokens in the m1 paper necessary??? I wonder if one studied the long reflection tokens from the m1 papers in Euclidean Space post post-training they would belong in the same subspace. Why am I always becoming an interpretability girlie, against my better judgment?! She said, I wonder about the false positives for the reflection tokens? I said, I can imagine some trigger words. Who would have thought that one could simply inspect conditional probabilities?

The KL term, these days, is left out of most reasoning model objective functions as [policies deviate wildly from the reference policy](https://arxiv.org/pdf/2506.10910) anyway. I shouted to A.L., the rollout completion length penalty in section 2.2.3 of the [magistral](https://arxiv.org/pdf/2506.10910) paper makes no sense! Why not just append a STOP_THINKING token after correct reasoning traces and move this upstream to the Long CoT cold start behavior imitation stage of the post raining post training reasoning rl pipeline? I finally figured out how to do las vegas algorithms with foundation models, I had been thinking forever. Computability theory, my OG.

Anyway the point of this above sections was to point out the similarity between the fork in the road reflection tokens in the reasoning rollouts and the thinking fusion mode from qwen3, at different layers of abstraction.

#### KL Distillation

We took out the KL term in the RL phase of the pipeline (recall that the m1 and the magistral and probably a few other papers do not have a KL term in the RL objective) so now we can put the KL back in during the distillation phase!


*Thank you to Sharon Zhou for amplifying my inspiration with random words like rollouts and transformers, RV for sending me a bunch of papers, and D from R for being a middle man of the information transfer with RV.*
