Official implementation of ATDHRH: Adaptive Topology with Dynamic Hierarchical Relational Hypergraph for Human-like Visual Perception (Still in the ideation and implementation stage, will be updated accordingly)
---

## Table of Contents

1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Objective](#objective)
4. [Basic Concepts](#basic-concepts)
5. [Related Works](#related-works)
6. [Literature Gap](#literature-gap)
7. [High-Level Intuition](#high-level-intuition)
8. [Methodology](#methodology)
9. [Week-by-Week Plan](#week-by-week-plan)
10. [Repository Layout](#repository-layout)
11. [Getting Started](#getting-started)
12. [References](#references)
13. [Citation](#citation)

---

## Introduction

For decades artificial vision systems have pursued a a simple goal of teaching machines to see the way humans do. CNNs approached this by treating images as spatial grids, sliding fixed kernels across uniform pixel neighborhoods. Vision Transformers abandoned the grid in favor of patch sequences connected by global attention.

But Human vision is not a convolution over a pixel array, nor a sequence of attended tokens. It is a **dynamic, hierarchical, relational** process where meaning emerges from how parts group into wholes, how wholes nest within larger structures, and how all of this reorganizes itself in real time as a scene is understood. No grid, no sequence, and no pairwise attention mechanism can natively represent this. They can approximate it statistically, given enough data, but the architecture itself works against the task.

This paper proposes that the correct representational foundation for human-like vision is the a Hypergraph: a structure that encodes group membership (the dynamic, hierarchical and relational part), adaptive topology as first-class properties rather than statistical emergents.

---

## Motivation

- **Practical and theoretical benefits.** Structured representation (the Hypergraph Data Strcuture) gives adaptive vision systems better fixation efficiency, higher accuracy at matched compute, and zero-shot cross-domain deployment in practice; theoretically it yields the first completeness result for visual representation, proving what structural properties are necessary rather than merely empirically helpful.

- **The three structural deficiencies of current approaches.** Existing representations like grids, sequences, pairwise graphs, flat hypergraphs lack at least one of: *adaptive topology* (structure conditioned on input), *group-relational hyperedges* (joint identity of a set, not pairwise approximation), and *learned hierarchy* (composition from parts to scenes). No existing structure provides all three simultaneously.

- **Why this problem matters and how algorithmic design solves it.** Current systems compensate for this structural mismatch through depth, data, and attention which helps partially but cannot close the gap, because the underlying data structure itself is wrong. The field has now converged on the necessary infrastructure (graph transformers, differentiable hypergraph theory, large-scale benchmarks) to make a principled, algorithmically-designed solution so designing a Hypergraph that encodes the dynamic, hierarchical and relational relationships is a step towards solving the problem.

### Why this problem matters and how algorithmic design solves it (Continued)

- **Medical imaging:** Lesion boundaries are group properties of surrounding tissue and pairwise attention cannot represent this cleanly, which is why radiologist accuracy still exceeds AI on boundary-sensitive diagnoses

- **Autonomous driving:** Occluded objects exist as visual fragments and without group-relational representation, a system cannot bind fragments into a coherent entity fast enough to act safely

- **Robotic manipulation:** Part-whole relationships like handle belongs to cup, cup sits on tray — are encoded as statistical correlations in current architectures, producing brittle generalization

- **Theoretical:** There is strong evidence that group membership and adaptive topology **cannot be learned efficiently** without being encoded architecturally — no amount of data or depth compensates for a structurally wrong representation

### Why It Has Not Been Solved

- **Standard GNNs:** Move beyond grids but edges are pairwise only — seven regions belonging to one object require six edges that carry no information about collective identity; group membership is lost by definition

- **Static Hypergraph Networks** *(Feng et al., AAAI 2019):* Introduced hyperedges but topology was fixed before training — grouping was decided before the model saw what it was grouping, making it content-blind

- **Dynamic Graph Networks:** Recovered content-sensitivity by evolving graph structure during inference but operated on simple pairwise graphs only — solving the wrong half of the problem

- **Windowed attention** *(Swin Transformer):* Introduced locality but windows are fixed spatial rectangles imposed regardless of where semantic boundaries fall — structurally equivalent to the CNN grid assumption

- **Combining existing approaches is insufficient:** Components were designed under incompatible assumptions; stitching them together is computationally expensive, unstable to train, and still lacks semantic guidance for meaningful group construction

- **The precise gap:** No single architecture has simultaneously achieved group-relational structure, content-adaptive topology, explicit hierarchy, and semantic guidance during graph construction — at benchmark scale

---

## Objective

To design, formally justify and empirically validate that **Dynamic Hierarchical Relational Hypergraph** a data structure that simultaneously provides **adaptive topology**, **group-relational hyperedges**, and **learned hierarchy** and to show that replacing the flat internal state of adaptive vision systems with a Hypergraph improves fixation efficiency, recognition accuracy, and cross-domain generalization.

---

## Basic Concepts

- Graph Taxonomy
- Why Graphs Over Grids for Vision
- What is Vision as a Computational Problem?
- Visual Processing and Visual Perception
- Understanding Passive Vision and Active Vision

### Graphs: The Intuition

**What is a Graph?**

- A formal way of saying **"these things relate to these other things."**
- **Nodes ($V$)** represent entities (people, objects, image regions, words).
- **Edges ($E$)** represent relationships between entities.
- Represented by an adjacency matrix $A$, where $A_{ij}=1$ if node $i$ connects to node $j$.

$$G=(V,E)$$

### What is a Hypergraph?

- In an ordinary **graph**, an edge joins *exactly two* vertices.
- A **hypergraph** generalizes this: a **hyperedge** can join *any number* of vertices.
- Formally, $H = (X, E)$ where $X$ is a set of vertices and $E \subseteq \mathcal{P}(X)\setminus\{\emptyset\}$ is a set of hyperedges.
- An ordinary graph is the special case where every hyperedge has size 2.
- **Directed hypergraphs**: each edge is a pair $(T, H_e)$ of a tail set and head set of vertices.

$$e_1=\{a,b,c\}, \quad e_2=\{c,d,e\}$$

### Representations

- **Incidence matrix** $I$ ($n\times m$): $I_{ij}=1$ if vertex $x_i \in$ edge $e_j$, else $0$.
- **Levi (bipartite incidence) graph**: parts $X$ and $E$; connect $x$–$e$ iff $x \in e$.
- **2-section / clique graph**: same vertices as $H$; join $u,v$ whenever they co-occur in some hyperedge (replaces each hyperedge with a clique).
- **Dual hypergraph** $H^*$: transpose the incidence matrix — vertices and edges swap roles.

### Special Classes

- **$k$-uniform**: every hyperedge has exactly $k$ vertices ($2$-uniform = ordinary graph).
- **$d$-regular**: every vertex belongs to exactly $d$ hyperedges.
- **$k$-partite**: vertices split into $k$ parts, each hyperedge takes one vertex per part.
- **Rank**: maximum hyperedge size in $H$.
- **Simple**: no loops, no repeated hyperedges. **Reduced**: no hyperedge is a subset of another.
- **Downward-closed** hypergraph $=$ an **abstract simplicial complex**.
- **Laminar**: any two hyperedges are disjoint or nested.
- **Subhypergraph**, **partial hypergraph**, **section hypergraph**: restrict vertices, edges, or both.

### Coloring, Cycles & Acyclicity

- **Coloring**: color vertices so no hyperedge of size $\ge 2$ is monochromatic; $\chi(H)$ is the fewest colors needed.
- **2-colorable** hypergraphs $=$ bipartite hypergraphs (Property B).
- **Berge cycle**: alternating vertex/edge sequence, like a cycle in the Levi graph; $H$ is Berge-acyclic iff its Levi graph is acyclic.
- **Tight cycle** (in $k$-uniform $H$): consecutive $k$-tuples of a vertex sequence all form hyperedges.
- **$\alpha,\beta,\gamma$-acyclicity**: increasingly strong notions, important in database theory:

$$\text{Berge-acyclic} \Rightarrow \gamma\text{-acyclic} \Rightarrow \beta\text{-acyclic} \Rightarrow \alpha\text{-acyclic}$$

### Isomorphism & Applications

- **Homomorphism**: vertex map sending every edge to an edge.
- **Isomorphism**: bijection on vertices + permutation of edges preserving incidence; **automorphisms** of $H$ form a group.

| Undirected uses | Directed & other uses |
|---|---|
| SAT problems | Network / transport modeling |
| Relational databases | Money-laundering detection |
| Machine learning & recommenders | VLSI partitioning |
| Bioinformatics | Sparse matrix computation |

### How Information Flows: Message Passing

- Every node holds a **feature vector** — what it currently knows about itself
- At each layer, three steps happen:
  1. **Send** — Each node sends its feature vector to all its neighbors
  2. **Aggregate** — Each node collects and combines what it receives
  3. **Update** — Each node updates its own representation

**Graph Convolution:** Same idea as image convolution — but sliding over a **graph neighborhood** instead of a pixel grid

### Why Graphs Over Grids for Vision

**Pixel Grid (CNN)**
- Every pixel equally related to its 8 neighbors
- No notion of whether neighbors belong to the **same object**
- Structure **imposed** on the image regardless of content

$\rightarrow$ **Region Graph (GNN)**
- Semantically similar regions are connected
- Connections reflect **meaning**, not just proximity
- Structure **discovered** from the content itself

> **But graphs are still pairwise** — Standard graphs connect exactly **two** nodes at a time and they cannot express that a set of regions *collectively* belongs to one object. This is why we need **hypergraphs**.

### What is Vision as a Computational Problem

- A camera produces a **grid of numbers** and each number is a light intensity value at one pixel location
- The computational problem of vision is: **given these numbers, extract meaning**
- Meaning includes what objects are present, where they are, how they relate to each other, what is happening in the scene
- This is trivial for humans and **extraordinarily difficult** for machines and a gap that has not been fully closed despite decades of research
- The difficulty is not computational power, it is that **meaning does not live in pixel values**; it lives in the relationships between them

### Visual Processing vs Visual Perception

**Visual Processing**
- **Mechanical transformation** of pixel values through mathematical operations
- Detects edges, textures, gradients, intensity patterns
- Produces **feature maps** which are the compressed representations of local statistics
- No understanding of what anything is and purely structural
- **CNNs do this extremely well**

$\neq$ **Visual Perception**
- **Interpretive assignment** of meaning to visual elements
- Groups elements into objects, separates figure from ground
- Infers **relationships** about what belongs together, what is in front, what is doing what
- Requires understanding of structure, not just statistics
- **Current AI falls short here**

> **The core gap** — Current architectures are excellent at processing and poor at perception because perception requires **relational, group-level representation** that grids and sequences cannot express

### Passive Vision vs Active Vision

**Passive Vision**
- Image arrives as a **fixed input** and system has no control over what it sees
- Task is to classify, detect, or segment from a **single static observation**
- Examples: ImageNet classification, object detection, segmentation
- **CNNs and ViTs are passive**

$\leftrightarrow$ **Active Vision**
- System **controls what it looks at** and decides where to direct attention next
- Perception is a **closed loop** and each observation informs the next action
- Representation built **iteratively and selectively**
- Examples: robotic gaze control, visual question answering, embodied navigation
- **Humans are active**

**The Biological Reality: Vision is Active** — Passive vision systems assume the full image is equally informative everywhere. Biological vision assumes almost nothing is informative until actively queried. This is a **fundamental architectural disagreement**.

### Where DHH Sits: Bridging Passive and Active

- **Current DHH architectures are passive**, they receive a fixed image and process it in a single forward pass
- However, **dynamic topology construction** at each layer is an internal form of active querying, the model decides which regions matter and reorganizes its structure accordingly
- This makes DHH a **step toward active vision** without requiring external action — selectivity is internal to the representation.
- **Full active vision** would extend this externally — the model deciding which part of a scene to observe next, requesting higher resolution in salient regions, iterating until confident
- This positions DHH as a **natural foundation** for active vision systems.

---

## Related Works

### I. Vision Backbones

- **ResNet** [3]: Deep Residual Learning for Image Recognition, *CVPR 2016*. Introduces residual connections; establishes the convolutional **grid** as the dominant vision backbone. [[GitHub]](https://github.com/kaiminghe/deep-residual-networks)
- **ViT** [4]: An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale, *ICLR 2021*. Treats an image as a fixed-length **sequence** of patches connected by global pairwise attention. [[GitHub]](https://github.com/google-research/vision_transformer)
- **Swin Transformer** [5]: Hierarchical Vision Transformer using Shifted Windows, *ICCV 2021*. Adds hierarchy and locality via fixed spatial windows; windows are geometric, not semantic. [[GitHub]](https://github.com/microsoft/Swin-Transformer)
- **ConvNeXt**: A ConvNet for the 2020s, *CVPR 2022*. Modernizes the CNN grid to match Transformer performance; topology remains fixed at design time. [[GitHub]](https://github.com/AlassaneSakande/A-ConvNet-of-2020s)

> All four share one trait: connectivity (grid neighborhood, patch sequence, or window) is imposed *before* the model sees the content.

### II. Graph-Based Vision Backbones

- **ViG** [6]: Vision GNN: An Image is Worth Graph of Nodes, *NeurIPS 2022*. First to treat image patches as graph nodes with $k$-NN pairwise edges instead of a grid. [[GitHub]](https://github.com/jichengyuan/Vision_GNN)
- **DGCNN** [7]: Dynamic Graph CNN for Learning on Point Clouds, *ACM TOG 2019*. Recomputes $k$-NN graph edges per layer in feature space ("EdgeConv"); pairwise only. [[GitHub]](https://github.com/WangYueFt/dgcnn)
- **DGMN2** [8]: Dynamic Graph Message Passing Networks, *TPAMI 2022*. Learns sparse, content-dependent pairwise edges for message passing on feature maps. [[GitHub]](https://github.com/fudan-zvg/DGMN2)
- **GreedyViG** [9]: Dynamic Axial Graph Construction for Efficient Vision GNNs, *CVPR 2024*. Greedy, axis-wise dynamic graph construction for efficiency; still a simple graph. [[GitHub]](https://github.com/SLDGroup/GreedyViG)
- **AdaptViG** [10]: Adaptive Vision GNN with Exponential Decay Gating, *CVPR 2026*. Gates pairwise edges by learned decay; adapts edge *weights*, not group structure. [[GitHub]](https://github.com/mmunir127/AdaptViG)

> Progression: fixed graph $\to$ dynamic graph $\to$ gated graph — but the edge is *always pairwise*.

### III. Adaptive Vision & Hypergraph Methods

**Adaptive (non-hypergraph) vision**

- **AdaptiveNN** [2]: Emulating human-like adaptive vision for efficient and flexible machine visual perception, *Nature Machine Intelligence 2025*. Sequentially selects where to look (active, coarse-to-fine); representation itself stays a flat feature map. [[GitHub]](https://github.com/LeapLabTHU/AdaptiveNN)

**Hypergraph vision methods**

- **ViHGNN** [11]: Vision HGNN: An Image is More than a Graph of Nodes, *ICCV 2023*. First vision hypergraph network; hyperedges are built once from static $k$-NN clusters. [[GitHub]](https://github.com/VITA-Group/ViHGNN)
- **DVHGNN** [12]: Multi-Scale Dilated Vision HGNN for Efficient Vision Recognition, *CVPR 2025*. Adds multi-scale, dilated hyperedge sampling; scales are fixed hyperparameters, not learned hierarchy. [[GitHub]](https://github.com/lcs0215/DVHGNN)
- **SoftHGNN** [13]: Soft Hypergraph Neural Networks for General Visual Recognition, *IJCV 2026*. Relaxes hard hyperedge membership to soft assignment weights; topology still precomputed per layer, not content-conditioned end-to-end. [[GitHub]](https://github.com/MengqiLei/SoftHGNN)
- **HGVT** [14]: Hypergraph Vision Transformers: Images are More than Nodes, More than Edges, *CVPR 2025*. Combines attention with hyperedges; hierarchy is not explicit across layers. [[Project Page]](https://rgendiff.github.io/hgvt2025/)
- **HGFormer** [15]: Topology-Aware Vision Transformer with HyperGraph Learning, *IEEE TMM 2025*. Learns hyperedge topology jointly with attention, but without an explicit part→object→scene hierarchy.
- **HGNN+** [16]: General Hypergraph Neural Networks, *TPAMI 2023*. General-purpose hypergraph convolution framework; hyperedges are given as input, not constructed from content.

### IV. Hierarchy & Dynamic Structure Learning

**Hierarchical graph learning (pooling)**

- **DiffPool** [17]: Hierarchical Graph Representation Learning with Differentiable Pooling, *NeurIPS 2018*. Learns soft cluster assignments to coarsen a graph hierarchically; designed for generic graphs, not vision, and clusters are pairwise-graph clusters, not group-relational hyperedges.
- **MinCutPool** [18]: Spectral Clustering with Graph Neural Networks for Graph Pooling, *ICML 2020*. Spectral relaxation of graph cuts for pooling; hierarchy exists, but topology per level is still a simple graph.
- **SAGPool** [19]: Self-Attention Graph Pooling, *ICML 2019*. Uses self-attention scores to drop nodes and coarsen the graph; selection is per-node, not group-forming.

**Dynamic topology / graph structure learning**

- **GraphGLOW** [20]: Universal and Generalizable Structure Learning for GNNs, *NeurIPS 2023*. Learns a shared structure-learning module transferable across graphs; operates on pairwise adjacency only.
- **LGI-LS** [21]: Learning Graph Structure, *NeurIPS 2022*. Infers a latent pairwise graph jointly with the downstream task; no notion of hyperedges or hierarchy. [[Project Page]](https://jianglin954.github.io/LGI-LS/)

---

## Literature Gap

- **Adaptive topology exists** (GreedyViG, AdaptViG, AdaptiveNN, GraphGLOW, LGI-LS) but only over **pairwise** graphs — structure adapts, relation type does not.
- **Group-relational hyperedges exist** (ViHGNN, DVHGNN, SoftHGNN, HGVT, HGFormer, HGNN+) but topology is largely **precomputed or per-layer static** — content adapts weights, not membership, across the full pipeline.
- **Learned hierarchy exists** (DiffPool, MinCutPool, SAGPool) but only for **generic pairwise graphs**, never combined with hyperedges or vision backbones at scale.
- **No method in the reviewed literature combines all three** adaptive topology, group-relational hyperedges, and learned hierarchy in a single, content-conditioned vision backbone.
- This is precisely the gap the proposed **Dynamic Hierarchical Relational Hypergraph (DHH)** targets.

---

## High-Level Intuition

- An image should be parsed as a **living structure** that reorganizes itself as understanding deepens — not as a grid or a sequence
- Start with raw patches $\rightarrow$ group semantically similar ones into hyperedges $\rightarrow$ reason over groups collectively $\rightarrow$ reform groups at higher abstraction $\rightarrow$ repeat across hierarchy
- Topology is **not fixed** as it is constructed from content, refined by the model's own intermediate representations, and different for every image

**Four mechanisms working together:**

- **Hyperedges**: many patches belong to one collective unit simultaneously, replacing pairwise edges
- **Dynamic construction**: grouping decided from image features at each layer, not precomputed
- **Hierarchy**: each stage groups outputs of the previous: edges $\rightarrow$ parts $\rightarrow$ objects $\rightarrow$ scene
- **Semantic guidance**: model uses its own class-level understanding to decide what belongs together

---

## Methodology

### Step 1: The Raw Image as Nodes

- The image is cut into a $14\times14$ grid $\Rightarrow$ **196 patches**.
- Each patch becomes a **node** with a feature vector $x_i \in \mathbb{R}^d$.
- At this stage the nodes are isolated — **no connections exist yet**.

### Step 2: Nodes Talk to Neighbors

$$A = \mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d}}\right)$$

- $Q, K$: query and key vectors derived from each node's features.
- $A[i,j]$ in plain words: **"how much node $i$ listens to node $j$."**
- Rows of $A$ sum to 1 — each node distributes its attention across all others.
- This happens **before** any grouping — it is local communication, not grouping yet.

### Step 3: Grouping into Hyperedges (The Core Step)

$$S = \mathrm{softmax}(\mathrm{MLP}(X)) \in \mathbb{R}^{n \times m}$$

- Each **column** of $S$ is one hyperedge.
- $S[i,k]$: how strongly patch $i$ belongs to hyperedge $k$.
- Example: patches $23, 24, 87 \to$ hyperedge 1 (a cat region).
- Patch $120 \to$ hyperedge 3 (background sky).

**This is the key step**: pairwise edges become *group* membership.

### Step 3 Continued: Why $S$ is Dynamic

Same formula $S = \mathrm{softmax}(\mathrm{MLP}(X))$ — **completely different output**.

- **Hospital image** — groups by organ / tissue type
- **Street scene** — groups by vehicle / building / sky

**The formula does not change — the grouping is entirely content-driven.**

### Step 4: Coarsening — Building the Next Level

$$X^{(l+1)} = S^{\top} X^{(l)}$$

- Each **hyperedge becomes one node** at the next level.
- $S^\top$ turns **columns** of $S$ (hyperedges) into **rows** of the new node set.
- $196$ patch nodes $\to$ $32$ hypernodes.
- Each hypernode's feature is a weighted average of the patches in its group.

### Step 5: Cross-Level Attention

- $A_{\text{cross}}$ lets a Level-0 node attend **directly** to a Level-2 hypernode — **skipping Level 1**.
- Plain language: a distinctive patch (a scar, a license plate) can signal the whole scene directly, without waiting to pass through every intermediate level.

### The Full Hierarchy

- Level 0 (196 patches) $\to$ Level 1 (32 semantic groups) $\to$ Level 2 (8 scene regions) $\to$ Level 3 (1 global node).
- Conceptually: **textures $\to$ parts $\to$ objects $\to$ scene.**

### The Four Data Structures

| Symbol | Name | Shape | What it holds |
|---|---|---|---|
| $X^{(l)}$ | Node features | $n_l \times d$ | feature vector per node at level $l$ |
| $H^{(l)}=S^{(l)}$ | Hyperedge assignment | $n_l \times m_l$ | how much each node belongs to each hyperedge |
| $A^{(l)}$ | Within-level attention | $n_l \times n_l$ | how nodes at level $l$ listen to each other |
| $A_{\text{cross}}^{(l,l+1)}$ | Cross-level attention | $n_l \times n_{l+1}$ | how a node can skip ahead to a higher level |

$n_l$ = number of nodes at level $l$ (e.g. $196, 32, 8, 1$); $m_l$ = number of hyperedges at level $l$.

### Training: Why Four Losses

$$\mathcal{L} = \mathcal{L}_{\text{mincut}} + \mathcal{L}_{\text{ortho}} + \mathcal{L}_{\text{cross}} + \mathcal{L}_{\text{task}}$$

- **$\mathcal{L}_{\text{mincut}}$**: without it, hyperedge groups form **randomly** instead of around related content.
- **$\mathcal{L}_{\text{ortho}}$**: without it, all patches **collapse into one giant group**.
- **$\mathcal{L}_{\text{cross}}$**: without it, groupings at different levels become **inconsistent** with each other.
- **$\mathcal{L}_{\text{task}}$**: the actual recognition goal — classification, detection, etc.

**Each loss patches one specific failure mode; none is optional.**

---

## Week-by-Week Plan

- **Week 1 — Foundation and Literature** — Done. [Literature Review](https://docs.google.com/spreadsheets/d/17-M-9dAxNQfgtTe4xh06hGaZQGL3WXeJvjQfyZjSRGU/edit?gid=0)
- **Week 2 — Problem Formalization** — Described in the Methodology Section of the slide
- **Week 3 — Baseline Implementation** — [HgVT (CVPR 2025)](https://rgendiff.github.io/hgvt2025/). Get standard HgVT running on CIFAR-10.
- **Week 4 — Core Implementation** — Implement dynamic hypergraph construction. Get full end-to-end forward pass running.
- **Week 5 — Hierarchy and Experiments** — Add hierarchical staging across layers. Run first real experiments on CIFAR-100 or ImageNet subset. Record four conditions: baseline CNN, ViT, static hypergraph, dynamic flat and our architecture.
- **Week 6 — Ablations and Analysis** — Remove each component one at a time: no hierarchy, no dynamic construction, pairwise edges only. Each ablation becomes a row in your results table. Run visualizations showing what hyperedges the model constructs on sample images.
- **Week 7 — Paper Writing** — Write in this order: methodology $\rightarrow$ experiments $\rightarrow$ introduction $\rightarrow$ related work $\rightarrow$ abstract.
- **Week 8 — Revision and Submission** — Tighten writing, finalize figures, results tables, architecture diagram, visualization examples. Submit.

---

## Repository Layout

```
ATHDRA/
├── README.md
├── LICENSE
├── requirements.txt
├── refs.bib                     # Full bibliography (reconstructed from the proposal)
├── .gitignore
├── configs/
│   └── dhh_cifar100.yaml        # Example training config
├── src/
│   └── athdra/
│       ├── __init__.py
│       ├── layers/
│       │   ├── __init__.py
│       │   ├── attention.py     # Within-level & cross-level attention (Steps 2, 5)
│       │   └── hyperedge.py     # S = softmax(MLP(X)) construction + coarsening (Steps 3, 4)
│       ├── losses/
│       │   ├── __init__.py
│       │   └── dhh_losses.py    # L_mincut + L_ortho + L_cross + L_task
│       ├── models/
│       │   ├── __init__.py
│       │   ├── dhh_block.py      # One DHH stage
│       │   └── dhh_net.py        # Full hierarchy: patches → parts → objects → scene
│       └── data/
│           ├── __init__.py
│           └── datasets.py       # CIFAR-10 / CIFAR-100 loaders
├── scripts/
│   └── train.py                  # End-to-end training entry point
├── experiments/
│   └── ablations.md              # Week 6 ablation grid
└── docs/
    └── methodology.md            # Full formal methodology notes
```

---

## Getting Started

```bash
# Clone
git clone https://github.com/Rushali-Sharif/ATHDRA.git
cd ATHDRA

# Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Sanity check: forward pass on random input
python -c "import torch; from athdra.models.dhh_net import DHHNet; \
m = DHHNet(num_classes=100); \
out = m(torch.randn(2, 196, 256)); \
print('logits:', out['logits'].shape)"

# Train (Week 3–5)
python scripts/train.py --config configs/dhh_cifar100.yaml
```

---

## References

1. M. Rodrigo, C. Cuevas, and N. García, "Comprehensive comparison between vision transformers and convolutional neural networks for face recognition tasks," *Scientific Reports*, vol. 14, no. 1, p. 21392, 2024.
2. Y. Wang, Y. Yue, Y. Yue, H. Wang, H. Jiang, Y. Han, Z. Ni, Y. Pu, M. Shi, R. Lu, et al., "Emulating human-like adaptive vision for efficient and flexible machine visual perception," *Nature Machine Intelligence*, pp. 1–19, 2025.
3. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," *arXiv preprint arXiv:1512.03385*, 2015.
4. A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16x16 words: Transformers for image recognition at scale," *ICLR*, 2021.
5. Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin transformer: Hierarchical vision transformer using shifted windows," in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021.
6. K. Han, Y. Wang, J. Guo, Y. Tang, and E. Wu, "Vision gnn: An image is worth graph of nodes," in *NeurIPS*, 2022.
7. Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, and J. M. Solomon, "Dynamic graph cnn for learning on point clouds," *ACM Transactions on Graphics (TOG)*, 2019.
8. L. Zhang, M. Chen, A. Arnab, X. Xue, and P. H. Torr, "Dynamic graph message passing networks for visual recognition," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2022.
9. M. Munir, W. Avery, M. M. Rahman, and R. Marculescu, "Greedyvig: Dynamic axial graph construction for efficient vision gnns," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 6118–6127, June 2024.
10. M. Munir, M. M. Rahman, and R. Marculescu, "Adaptvig: Adaptive vision gnn with exponential decay gating," in *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, pp. 440–450, 2026.
11. Y. Han, P. Wang, S. Kundu, Y. Ding, and Z. Wang, "Vision hgnn: An image is more than a graph of nodes," in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 19878–19888, October 2023.
12. C. Li, T. Li, X. Hu, D. Luo, and T. Jin, "Dvhgnn: Multi-scale dilated vision hgnn for efficient vision recognition," in *Proceedings of the Computer Vision and Pattern Recognition Conference*, pp. 20158–20168, 2025.
13. M. Lei, Y. Wu, S. Li, X. Zheng, J. Wang, S. Du, and Y. Gao, "Softhgnn: Soft hypergraph neural networks for general visual recognition: M. lei et al.," *International Journal of Computer Vision*, vol. 134, no. 5, p. 244, 2026.
14. J. Fixelle, "Hypergraph vision transformers: Images are more than nodes, more than edges," in *Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)*, pp. 9751–9761, June 2025.
15. H. Wang, S. Zhang, and B. Leng, "Hgformer: Topology-aware vision transformer with hypergraph learning," *IEEE Transactions on Multimedia*, 2025.
16. Y. Gao, Y. Feng, S. Ji, and R. Ji, "Hgnn+: General hypergraph neural networks," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 45, no. 3, pp. 3181–3199, 2022.
17. Z. Ying, J. You, C. Morris, X. Ren, W. Hamilton, and J. Leskovec, "Hierarchical graph representation learning with differentiable pooling," *Advances in Neural Information Processing Systems*, vol. 31, 2018.
18. F. M. Bianchi, D. Grattarola, and C. Alippi, "Spectral clustering with graph neural networks for graph pooling," in *International Conference on Machine Learning*, pp. 874–883, PMLR, 2020.
19. J. Lee, I. Lee, and J. Kang, "Self-attention graph pooling," in *International Conference on Machine Learning*, pp. 3734–3743, PMLR, 2019.
20. W. Zhao, Q. Wu, C. Yang, and J. Yan, "Graphglow: Universal and generalizable structure learning for graph neural networks," in *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pp. 3525–3536, 2023.
21. J. Lu, Y. Xu, H. Wang, Y. Bai, and Y. Fu, "Latent graph inference with limited supervision," 2023.

---
