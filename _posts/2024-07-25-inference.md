---
layout: post
title: On Decoding, or, Inference in Large Language Models
published: false
usemathjax: true
tags: inference, large language models
---

"On Decoding", or, how to learn the model $$ \log(p(y | x )) = \sum_{j=0}^{m-1} \log(p(y_{j+1} | y_{\le_{j}}, x))$$ and estimate $$y^{*} = \arg\max_{y} p(y|x)$$, using a transformer architecture.

This estimation is exponential. Why? 

<!--excerpt-->

## Greedy Decoding

## Beam Decoding 

## Speculative Decoding

## MCTS Decoding
