---
layout: post
title: Cultural Contrastive Learning
published: false
usemathjax: true
tags: numinous
---

*After a long hiatus from this blog, I'm writing about a three-week experiment
I did to express a proof of concept methodology for finetuning multimodal
embeddings to imitate a human's engram when one has access to social media
platforms, building on ideas from contrastive learning, multimodal foundation
models, philosophy, and media theory. The full text can be found [here](numinous_contrastive.pdf)*

Modality gap in representation learning is a well-studied problem. While
many have definitions of it in measurable benchmarks, most conceptualiza-
tions and solutions of the problem focus on style and lower-order concrete
semantics. We elucidate a new perspective and class of problems in the
space of multimodal representation learning, especially as it pertains to
personalization, provide a proof of concept of finetuning a representation
space for this problem, and discuss its applications in various generative AI
pipelines.

Qualia refers to the subjective, qualitative, and felt experiences of an
individual’s conscious experience. Examples include the feeling of
pain, the taste of coffee, or the color red as a you, the individual, per-
ceive it. Qualia is conditional on an individual’s neural achitecture,
so to speak, and the experiences they collect through their life, the
particular environments they are embedded in (in the sense of other
agents being part of an environment, in the sense of the Sapir Whorf
Hypothesis [Whorf 1956], in the sense of Wittgenstein [Wittgen-
stein 1953]), and so on. (The Sapir-Whorf hypothesis suggests that
language influences thought. This connects to Wittgenstein’s con-
cept of language games, where meaning emerges from use within
specific contexts.)

In culture (so qualia of a collective group of individuals), such as
literature and art, qualia can be described formally as synaesthesia
(sensor crossover between modalities), aesthetic affinity (a form of
emotional kinship), or semiotics (shared symbolic languages). Other
informal words for this might be resonance, evocation, zeitgeist
convergence (shared cultural moment expression).

The concept of a modality gap was first introduced in Mind the Gap
(Liang et al, 2022) [Liang et al . 2022] which posits that geometric
inductive bias introduced in multimodal embeddings in which uni-
modal domains are tokenized and embedded separately creates a
modality gap on image and image caption distributions.

A searchable continuous latent space which solves the modality
gap lends itself to a multimodal embedding as well as a latent space
for user-conditional multimodal generation. We believe that this is
an approximation to understanding the phenomenal binding prob-
lem [Pearce 2012], which is about how objects, background objects,
as well as abstract and affective features are integrated into a unified
experience for an individual.

A motivating application of customizing a CLIP-like latent space
is its use in custom text-conditional diffusion pipelines [Ramesh et al .
2022]. A custom latent space approach could be complementary to
fine-tuning diffusion weights [Ruiz et al. 2022], which focuses more
on direct style-transfer like results rather than semantic understand-
ing.

Thus far, the representational learning research community has
focused on multimodal distributions of (text, image) pairs which
are relatively straightforward in their translation. For example, The
Mind the Gap paper evaluates models for their geometric gap on the
COCO dataset, [Lin et al . 2014] which contains photos of generic ob-
jects (in the same sense that [Ramesh et al. 2021] pre-trained DALL-
E1 on image, text pairs for which the text appeared in Wikipedia
$$> 100$$ times), Voyage, [Voyage AI 2024], evaluates mixed modality
search on the distribution (text, image of text), i.e. that the string
"Hello world" retrieves an image of "Hello world" rather than string
such as "Cat."

While these evaluations form a baseline of multimodal repre-
sentation gap, they still represent relatively simple cross domain
transformations. For example, the transformation from image to
image caption focuses on the object level, which most if not almost
all observers of the image would agree on. And the transformation
from text to image of text is as simple as save a pdf with "text" in it.
In some sense, it means that these joint distributions have higher
mutual information and are easier to learn than an individual’s
sensory space.

An individual’s sensory space, on the other hand, is shaped by
their histories, experiences, unique biology. Digitally, it is traceable,
for example, through a user’s hypertextual [Liu and Almeda 2025]
space, intentional navigation through the web, manual linking, et
cetera. When we learn a custom user adapter downstream of a pre-
trained baseline multimodal embeddings model, we are in, some
sense, learning this transformation.

The dataset we explore with is a second-degree scrape of a test
user’s text-based and image-based semantic space. This is chosen as
a joint distribution which is representative of the test user’s qualia
(perhaps, in a cultural subspace).

We choose to join Substack and Pinterest space as they represent
intentional content discovery platforms in which users explore an
inner space such as aesthetics for home design, visualizing futures,
or introspective writing with emotional qualities. In particular, these are two domains in which we expect a high frequency of cross-
artefact association driven by the user’s intuitive style, or rhizomatic
[Wikipedia contributors 2025] thinking.

In particular, this user-qualia-representative dataset is one in
which various forms of a modality gap appear with embedding
spaces such as voyage-multimodal-3.





