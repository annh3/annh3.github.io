---
layout: post
title: stochastic parrots of post raining 🌧️ ☔️ post training reinforcement learning reasoning
published: false
usemathjax: true
tags: rl, llms, foundation models, rollouts, sft
---

I am indeed, becoming an interpretability girlie, against my better judgement. 

What sparked my curiousity? The long reflection tokens from the m1 paper which sent me reeling on a chat with my friend. Then I read another paper, the Qwen3, where they talked about thinking and non thinking mode fusion, and I was like, is that really that difficult? It seems you can mix and match any sequence of domain-specific foundation model rollouts that by appending transition tokens which functionally are similar to conditional logic *and, or, where, etc ...*, fine-tune on those rollouts, rinse and repeat. 

The long reflection tokens from the [m1](https://arxiv.org/pdf/2506.13585), *'However, Recheck, Wait, Aha'* were apparently very important tokens for the reasoning paths for [stabilizing entropy](https://arxiv.org/abs/2505.22617) of the learned policy. I'll assume, also, that avoiding entropy collapse is a good prior for the correctness of reasoning paths. To preserve these tokens, which, due to their low $$\pi_{ref}$$ weight in the denominator of the $$IS$$, which creates a high $$\dfrac{\pi_{cur}}{\pi_{ref}}$$ for the advantage ($$\sum_{i=1}^t r_i - V$$), were clipped out of the PPO (and GSPO and GRPO) updates, the CIPSO objective from the m1 paper simply adds a stop gradient operation, so the high IS = $$\dfrac{\pi_{cur}}{\pi_{ref}}$$ term does not explode the gradient update in the chain rule of backpropagation, and tokens (*'However, Recheck, Wait, Aha'*) are assigned credit in the update.

#### PPO Objective Function Minus the KL Term
$$J(\theta) = \mathbb{E}[\frac{1}{|o_i|}] \sum_{t=1}^{|o_i|} min(r_{i,t}A_{i,t}, clip(r_{i,t}, 1-\epsilon, 1+\epsilon)A_{i,t}) $$

#### CISPO Objective function
$$ J(\theta) = \mathbb{E}[\frac{1}{sum_{i=1}^G |o_i| } \sum_{i=1}^G \sum_{t=1}^{|o_i|} sg(r_{i,t})A_{i,t} \log \pi_{\theta}(o_{i,t} | q, o_{i,<t}))] $$




*Thank you to Sharon Zhou for amplifying my inspiration with random words like rollouts and transformers, Risto Vuorio for sending me a bunch of papers, and Dean from R for being a middle man of the information transfer with RV.*
