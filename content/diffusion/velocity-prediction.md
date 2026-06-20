---
title: Reparameterizing an ε-predictor as a v-predictor
tags:
  - diffusion
  - velocity-prediction
draft: false
---

A diffusion model trained to predict noise ($\epsilon$) can be turned into one
that predicts **velocity** ($v$) without retraining — the two are related by a
fixed, schedule-dependent change of variables. This note works through that
reparameterization and verifies it empirically: I load SD 1.5 (a noise
predictor) and sample from it two ways — once in $\epsilon$-space, once by
converting its output to velocity on the fly — and confirm the images match.

The velocity parameterization is the one made explicit in [Progressive
Distillation for Fast Sampling](https://arxiv.org/abs/2202.00512), and it's the
bridge from the diffusion view into the flow-matching framing, where the model
is expected to output a velocity field. Everything here is in **latent space**
([latent diffusion](https://arxiv.org/abs/2112.10752)): SD 1.5 diffuses the
VAE latent, not pixels.

## The setup

The forward process noises a clean latent $x_0$ with Gaussian noise $\epsilon$
on a schedule $(\alpha_t, \sigma_t)$:

$$
x_t = \alpha_t\, x_0 + \sigma_t\, \epsilon.
$$

Treating $x_t$ as a point mass moving in time, its **velocity** is just the
time derivative:

$$
v_t = \dot{\alpha}_t\, x_0 + \dot{\sigma}_t\, \epsilon.
$$

A noise predictor gives us $\hat\epsilon$. To express $v_t$ in terms of the
quantities we actually have at inference ($x_t$ and $\hat\epsilon$), solve the
forward equation for $x_0 = (x_t - \sigma_t \epsilon)/\alpha_t$ and substitute:

$$
v_t = \frac{\dot{\alpha}_t}{\alpha_t}\, x_t
      + \left( \dot{\sigma}_t - \frac{\dot{\alpha}_t\, \sigma_t}{\alpha_t} \right) \epsilon.
$$

That identity is the whole trick: **given an $\epsilon$-prediction and the
schedule, you get the velocity for free.** No retraining, no new weights.

> [!note]- The reverse direction: recovering $x_0$ from velocity
> For a velocity-based sampler you need to denoise — recover $\hat x_0$ — from
> $v_t$. Take the two defining equations:
> $$x_t = \alpha_t x_0 + \sigma_t \epsilon, \qquad v_t = \dot\alpha_t x_0 + \dot\sigma_t \epsilon.$$
> Eliminate $\epsilon$ (multiply the first by $\dot\sigma_t$, the second by
> $\sigma_t$, subtract) to get
> $$
> x_0 = \frac{\sigma_t v_t - \dot\sigma_t\, x_t}{\sigma_t \dot\alpha_t - \dot\sigma_t \alpha_t}.
> $$
> This is the denoiser form used in [2509.25170](https://arxiv.org/abs/2509.25170).

## A scheduler that exposes the derivatives

The standard `diffusers` DDIM scheduler doesn't hand you $\dot\alpha_t$ and
$\dot\sigma_t$, which the velocity formula needs. So I build the schedule
explicitly — linear-$\beta$, with $\alpha$ and $\sigma$ from the cumulative
product and their derivatives by finite difference. Having the time-segment
indexing explicit also makes the per-step bookkeeping easier to follow.

```python
class LinearBetaScheduler:
    def __init__(self, T=1000, beta_min=0.00085, beta_max=0.012):
        self.T = T
        self.betas = torch.linspace(beta_min, beta_max, T)
        self.alphas_cumprod = torch.cumprod(1 - self.betas, dim=0)

        self.alpha = self.alphas_cumprod.sqrt().to("mps")
        self.sigma = (1 - self.alphas_cumprod).sqrt().to("mps")

        # time derivatives via forward difference
        self.dot_alpha = torch.zeros(T).to("mps")
        self.dot_sigma = torch.zeros(T).to("mps")
        self.dot_alpha[1:] = self.alpha[1:] - self.alpha[:-1]
        self.dot_sigma[1:] = self.sigma[1:] - self.sigma[:-1]
        self.dot_alpha[0] = self.dot_alpha[1]   # boundary
        self.dot_sigma[0] = self.dot_sigma[1]
```

## Baseline: sampling in ε-space

First the ordinary path — predict noise, solve for $\hat z_0$, step. SD 1.5 runs
with [DDIM](https://arxiv.org/abs/2010.02502) at 20–50 steps; the deterministic
reverse process lets us skip timesteps because the marginals
$q(x_t \mid x_0)$ stay valid. Classifier-free guidance mixes the conditional and
unconditional noise predictions:

$$
\hat\epsilon_{\text{guided}} = \hat\epsilon_{\text{uncond}}
  + w\,(\hat\epsilon_{\text{cond}} - \hat\epsilon_{\text{uncond}}).
$$

```python
def ddpm_epsilon_sampler(prompt, scheduler, guidance_scale=7.5):
    timesteps = torch.linspace(999, 0, 50).long()
    z_t = sample_ddpm_latent(time=999)
    text_emb, uncond_emb = encode_text(prompt), encode_text("")

    for i, t in enumerate(timesteps):
        t_tensor = torch.tensor([t], device="mps")
        eps_uncond = unet(z_t, t_tensor, encoder_hidden_states=uncond_emb).sample
        eps_text   = unet(z_t, t_tensor, encoder_hidden_states=text_emb).sample
        eps = eps_uncond + guidance_scale * (eps_text - eps_uncond)

        z0 = (z_t - scheduler.sigma[t] * eps) / scheduler.alpha[t]
        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            z_t = scheduler.alpha[t_next] * z0 + scheduler.sigma[t_next] * eps
        else:
            z_t = z0
    return z_t
```

Prompt: *"A woman on vacation in Bali."*

![[assets/velocity_epsilon_conversion_16_0.png]]

## The conversion, applied

Now the velocity path. Wrap the (CFG-combined) noise predictor, convert its
output to velocity via the identity above, then denoise with the
$x_0$-from-$v$ formula.

A nice structural fact worth noting: the **velocity operator** (ε → v) and the
**CFG operator** (linear mix of cond/uncond) commute — so it doesn't matter
whether you apply guidance before or after converting to velocity.

```python
def noise_to_velocity(prompt, z_t, t, unet, scheduler, guidance_scale=7.5):
    eps = noise_predictor(prompt, z_t, t, unet, guidance_scale)
    v = (scheduler.dot_alpha[t] / scheduler.alpha[t]) * z_t \
        + (scheduler.dot_sigma[t]
           - scheduler.dot_alpha[t] * scheduler.sigma[t] / scheduler.alpha[t]) * eps
    return v, eps

def velocity_ddim_sampler(prompt, unet, scheduler, guidance_scale=7.5):
    timesteps = torch.linspace(999, 0, 50).long()
    z_t = sample_ddpm_latent(time=999)

    for i, t in enumerate(timesteps):
        v, eps = noise_to_velocity(prompt, z_t, t, unet, scheduler, guidance_scale)
        denom = scheduler.sigma[t] * scheduler.dot_alpha[t] \
                - scheduler.dot_sigma[t] * scheduler.alpha[t]
        z0 = (scheduler.sigma[t] * v - scheduler.dot_sigma[t] * z_t) / denom
        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            z_t = scheduler.alpha[t_next] * z0 + scheduler.sigma[t_next] * eps
        else:
            z_t = z0
    return z_t
```

Same prompt, sampling entirely through the velocity reparameterization:

![[assets/velocity_epsilon_conversion_28_0.png]]

The two images agree — the velocity reparameterization reproduces the
ε-predictor's samples, as it must, since the conversion is exact. The payoff:
SD 1.5 can now be dropped into a velocity-expecting pipeline (flow-matching
samplers) without any retraining.
