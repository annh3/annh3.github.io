---
title: RoPE Embeddings (Multidimensional)
tags:
  - transformers
---

We'll first go over RoPE for the classic case of language modeling with an input tensor of shape `(batch, seq_len, embedding_dim)` and then extend to the $3D$ case for multimodal transformers. 

Fix a position $n$, this is the token position in the sequence. 

Then each pair $q_{2i}, q_{2i}$ (along the embedding dimension) has a rotation matrix

$$\begin{bmatrix} cos(n \cdot \theta_i) & -sin(n \cdot \theta_i) \\ sin(n \cdot \theta_i) & cos(n \cdot \theta_i) \end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}$$

Where $\theta_i = \frac{1}{10000^{(2i/d)}}$

Since adjacent indices are paired together the rotation matrix is a block diagonal.

$$\begin{pmatrix} R(n\theta_0) & & & \\ & R(n\theta_1) & & \\ & & \ddots & \\ & & & R(n\theta_{n/2}) \end{pmatrix} \begin{pmatrix} \mathbf{q}_1 \\ \mathbf{q}_2 \\ \vdots \\ \mathbf{q}_{n/2} \end{pmatrix}$$

So for each position in the sequence, we'll create a version of this matrix.

Ok, now we're ready to move onto $3D$ rotary positional embeddings. Consider an input with a height, width, and a time dimension. Then for each token at a coordinate (t,h,w) there are $d_t$ positions for time, $d_h$ for height, and $d_w$ for width. So the rotation matrix ends up looking (conceptually) like a gigantic block matrix. In practice, we can loop over each pair of coordinates and rotate them--this is much more computationally efficient.

$$\begin{pmatrix} \begin{pmatrix} R(n\theta_0) & & & \\ & R(n\theta_1) & & \\ & & \ddots & \\ & & & R(n\theta_{n/2}) \end{pmatrix} & & \\ & \begin{pmatrix} R(n\theta_0) & & & \\ & R(n\theta_1) & & \\ & & \ddots & \\ & & & R(n\theta_{n/2}) \end{pmatrix} & \\ & & \begin{pmatrix} R(n\theta_0) & & & \\ & R(n\theta_1) & & \\ & & \ddots & \\ & & & R(n\theta_{n/2}) \end{pmatrix} \end{pmatrix}$$