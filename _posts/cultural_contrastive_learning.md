---
layout: post
title: Cultural Contrastive Learning
published: false
usemathjax: true
tags: numinous
---

Modality gap in representation learning is a well-studied problem. While
many have definitions of it in measurable benchmarks, most conceptualiza-
tions and solutions of the problem focus on style and lower-order concrete
semantics. We elucidate a new perspective and class of problems in the
space of multimodal representation learning, especially as it pertains to
personalization, provide a proof of concept of finetuning a representation
space for this problem, and discuss its applications in various generative AI
pipelines.

### Introduction

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

### Background

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

### Dataset

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

### Initial Evaluations on Raw Text and Image Features

The first evaluation above shows a 0.79 recall rate on Pinterest
image and associated ground truth captions, which shows a good
image-to-image-caption generalization ability of voyage-multimodal-
3. The second evaluation above shows a clear modality gap on
Substack posts, which are out of distribution of the (image, image-
caption) distribution. In the raw feature space, text queries retrieve
images of text regardless of the semantic content, rather than any
other type of image artifact, thus failing the text to pdf of text
evaluation in [Voyage AI 2024].

### Methodology
#### Soft Labeling

Given the initial analysis of embedding raw text features, we choose
to work, in this paper, with AI annotations of raw text and image
features, so that the cultural knowledge of foundation models serves
as a form of supervision signal. In future work, a finetuned model
based on an individual could provide more attuned personalization.

We provide, in the Appendix A.1, the prompts used.
The annotation allows us to project the raw image and text fea-
tures to roughly the same space, when embedded with voyage-multimodal-3
or nomic-text-v1, meaning the new feature space has good seman-
tic overlap.

From this annotation semantic space, we run UMAP [McInnes
et al . 2018] on the text and image domains individually, then HDB-
SCAN [Malzer and Baum 2019] to produce clusters. We then com-
pute cluster assignments across domains via centroid similarity and
filter to a 1-1 mapping via the Hungarian Algorithm [CP-Algorithms
2023]. Figure 3. Shows a visualization of the clustered space.

In Appendix A.2 we provide samples of semantic cross-modal
clusters produced by this method.

#### Finetuning

LoRA [Hu et al . 2021] introduced the idea of sparse fine-tuning
the through the addition of low-rank matrices. Inspired, we learn
a linear and a two-layer adapter, which is applied to the output
of voyage-multimodal-3 to associate user qualia across text and
image space.

The soft cluster labels provide positive-negative labels for fine-
tuning. The initial loss function we use is the InfoNCE loss [van den
Oord et al. 2018], written here for concreteness.

$$ InfoNCE(q,X) = - \log (\sum_{pos} exp(\dfrac{sim(X_q,X_{pos}}{\tau})) + \log (\sum_{neg} exp(\dfrac{sim(X_q,X_{neg}}{\tau})) $$

Where 𝑞 is an anchor point. For a given batch, we cycle through all
points so each point is the anchor once in the computation of a per-
batch loss. In the notation 𝑋𝑖 represents a row-vector of the dataset.
The loss function penalizes high similarity between the anchor and
negative points while rewarding high similarity between the anchor
and positive points.

#### Results

We measure few of initial metrics: cross-modal retrieval and cluster
coherence (the ratio of retrieved artefacts which are of the same clus-
ter), uniformity [Wang and Isola 2020], and create a query-histogram
visualization for the effect of temperature on the fine-tuned distribu-
tion. The metrics are computed held-out clusters after the Hungarian
Algorithm is applied to filter the cluster matching. We chose to hold
out novel clusters rather than just data points as it is a more difficult
generalization measurement. We note that the coherence jump from
1-layer to 2-layer is significant, as the Universal Approximation The-
orem [Hornik et al . 1989] would suggest, and demonstrate on these
two architectures as a proof of concept. The uniformity score is
measured as it is associated with downstream retrieval performance
and the author’s choice to measure it was due to initial observations
on the uniformity of the query histograms.

## References
<a id="1">[1]</a> 
CP-Algorithms. 2023. Hungarian Algorithm. Retrieved September 4, 2025, from https://cp-algorithms.com/graph/hungarian-algorithm.html.

<a id="2">[2]</a> 
Kurt Hornik, Maxwell Stinchcombe, and Halbert White. 1989. Multilayer feedforward
networks are universal approximators. *Neural Networks* 2, 5 (1989), 359–366.

<a id="3">[3]</a> 
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, and Weizhu Chen. 2021. LoRA: Low-Rank Adaptation of Large Language
Models. *arXiv preprint arXiv:2106.09685* (2021).

<a id="4">[4]</a> 
Wei Liang, Yujia Zhang, Yewon Kwon, Serena Yeung, and James Zou. 2022. Mind the
Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation
Learning. In *Advances in Neural Information Processing Systems (NeurIPS)*.

<a id="5">[5]</a>
Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James
Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. 2014.
Microsoft COCO: Common Objects in Context. *arXiv preprint arXiv:1405.0312* (2014).

<a id="6">[6]</a>
Shirley Liu and Santiago G. Almeda. 2025. Agency Among Agents: Designing with
Hypertextual Friction in the Algorithmic Web. *arXiv preprint arXiv:2507.23585* (2025).

<a id="7">[7]</a>
Christoph Malzer and Michael Baum. 2019. A Hybrid Approach To Hierarchical
Density-based Cluster Selection. *arXiv preprint arXiv:1911.02282* (2019).

<a id="8">[8]</a>
Leland McInnes, John Healy, and James Melville. 2018. UMAP: Uniform Mani-
fold Approximation and Projection for Dimension Reduction. *arXiv preprint
arXiv:1802.03426* (2018).

<a id="9">[9]</a>
David Pearce. 2012. Non-materialist physicalism: an experimentally testable conjecture.
Available at https://www.physicalism.com/.

<a id="10">[10]</a>
Tongzhou Wang and Phillip Isola. 2020. Understanding Contrastive Representation
Learning through Alignment and Uniformity on the Hypersphere. *arXiv preprint
arXiv:2005.10242* (2020).

<a id="11">[11]</a>
Benjamin Lee Whorf. 1956. *Language, Thought, and Reality: Selected Writings of Ben-
jamin Lee Whorf.* MIT Press.

<a id="12">[12]</a>
Wikipedia contributors. 2025. Rhizomatic learning. Retrieved September 4, 2025, from
https://en.wikipedia.org/wiki/Rhizomatic_learning.

<a id="13">[13]</a>
Ludwig Wittgenstein. 1953. *Philosophical Investigations.* Blackwell. Original work published posthumously.



