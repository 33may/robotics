> Source: https://arxiv.org/abs/2411.16253 (v2). Converted from the arXiv HTML with pandoc on 2026-10-06. Math is the authors' LaTeX. Figures link to arxiv.org. PDF: `../pdf/wang-2025-octree-graph.pdf`. Tables may be imperfect, check the PDF before quoting a number.

# Open-Vocabulary Octree-Graph for 3D Scene Understanding

Zhigang Wang <sup>†</sup><sup>†</sup>thanks: Equal contribution. Affiliation: Northwestern Polytechnical University Affiliation: Shanghai AI Laboratory    Yifei Su Affiliation: University of Chinese Academy of Sciences Affiliation: CASIA    Chenhui Li    Dong Wang Affiliation: Shanghai AI Laboratory    Yan Huang Affiliation: University of Chinese Academy of Sciences Affiliation: CASIA    Xuelong Li Affiliation: TeleAI    Bin Zhao <sup>†</sup><sup>†</sup>thanks: Corresponding author. Affiliation: Northwestern Polytechnical University Affiliation: Shanghai AI Laboratory

###### Abstract

Open-vocabulary 3D scene understanding is indispensable for embodied agents. Recent works leverage pretrained vision-language models (VLMs) for object segmentation and project them to point clouds to build 3D maps. Despite progress, a point cloud is a set of unordered coordinates that requires substantial storage space and does not directly convey occupancy information or spatial relation, making existing methods inefficient for downstream tasks, e.g., path planning and text-based object retrieval. To address these issues, we propose Octree-Graph, a novel scene representation for open-vocabulary 3D scene understanding. Specifically, a Chronological Group-wise Segment Merging (CGSM) strategy and an Instance Feature Aggregation (IFA) algorithm are first designed to get 3D instances and corresponding semantic features. Subsequently, an adaptive-octree structure is developed that stores semantics and depicts the occupancy of an object adjustably according to its shape. Finally, the Octree-Graph is constructed where each adaptive-octree acts as a graph node, and edges describe the spatial relations among nodes. Extensive experiments on various tasks are conducted on several widely-used datasets, demonstrating the versatility and effectiveness of our method. Code is available [here](https://github.com/yifeisu/OV-Octree-Graph).

## 1 Introduction

3D scene understanding is receiving increasing attention due to its widespread usage in robots \[[55](#bib.bib41)\] and VR/AR applications \[[17](#bib.bib42)\]. Previous works \[[38](#bib.bib8), [34](#bib.bib22), [42](#bib.bib26), [21](#bib.bib27), [19](#bib.bib32)\] trained models on particular 3D scene datasets to complete this task. Although significant progress has been achieved, they are limited to a closed-set category. Recently, we have witnessed the impressive generalization ability of foundation models (*e.g*., SAM \[[18](#bib.bib16)\] and CLIP \[[33](#bib.bib14)\]) which can perceive various objects in unseen scenarios, inspiring a lot of open-vocabulary 3D scene understanding methods \[[25](#bib.bib10), [7](#bib.bib6), [44](#bib.bib7), [16](#bib.bib28), [4](#bib.bib29), [5](#bib.bib30), [49](#bib.bib31)\]. Given an RGB-D sequence with camera poses, mainstream methods leverage the off-the-shelf foundation models to generate 2D object masks and corresponding visual-language features, and then project them to point clouds to construct a semantic 3D map.

![Refer to caption](https://arxiv.org/html/2411.16253v2/2411.16253v2/intro_new.png)

Figure 1: (a) A 3D scene. (b) The corresponding semantic 3D map based on point clouds (6.8M). (c) Our Octree-Graph where each object is represented by the proposed adaptive-octree and each edge contains rich spatial relations among objects. All adaptive-octrees occupy 42KB of storage space in total.

Despite the favorable open-vocabulary understanding capability, they have two drawbacks. 1) Inefficient space representation of 3D scenes. Most mainstream methods \[[31](#bib.bib19), [48](#bib.bib13), [15](#bib.bib5)\] build the 3D map based on point clouds, as shown in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding") (b). Point clouds are unordered discrete coordinates that require considerable storage space, making existing methods inefficient to deploy on embodied agents with limited storage resources. Moreover, point clouds lack explicit representation of occupancy information and spatial connectivity which are critical for downstream tasks, *e.g*., path planning and text-based object retrieval. 2) Inaccurate semantic object segmentation for 3D map construction. Most methods overlook the inaccuracy of foundation models/vision-language models (VLMs) when conducting object segmentation and feature extraction, inevitably causing imprecise 3D object segments and degraded semantics.

To alleviate these problems, we propose Octree-Graph as shown in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding") (c), a novel open-vocabulary scene representation designed to characterize the occupancy and semantics of each object, as well as the relations among them. Specifically, the adaptive-octree is first proposed to depict each object’s occupancy, which inherits the advantages of the octree structure by hierarchically representing a 3D space with structured sub-regions. Compared to the point cloud without regional or hierarchical information, it can save significant storage space. Furthermore, our adaptive-octree initializes each object adaptively according to its shape, enabling a precise description of the occupancy within a limited octree depth. This is particularly suitable for objects with large aspect ratios, *e.g*., walls and floors. Based on this, the Octree-Graph is constructed where each adaptive-octree acts as a graph node, and each edge encompasses rich relations among objects, *e.g*., distances and relative orientations. The proposed Octree-Graph can be directly applied to downstream tasks such as object retrieval, occupancy queries, and path planning, thus providing significant convenience.

To obtain accurate semantic objects for Octree-Graph construction, we devise a training-free pipeline. First, given input images, 2D proposals are segmented via an off-the-shelf segmenter, and corresponding visual-language features are extracted by pretrained VLMs. Then, they are projected into 3D space as point cloud segments. Second, to correctly merge segments belonging to the same instance, a Chronological Group-wise Segment Merging (CGSM) strategy is proposed, where the segments are partitioned into several groups in time order. Each group is individually processed to leverage spatiotemporal details from the neighborhood while avoiding interference from global redundancy. Third, an Instance Feature Aggregation (IFA) method is proposed to obtain semantic representations for each object. Unlike existing works that directly average features as a result, we simultaneously consider the representativeness and distinctiveness of a feature during the fusion process. Our contributions are summarized as follows.

- •
  We propose the Octree-Graph for open-vocabulary 3D scene understanding, which efficiently depicts objects’ occupancies, semantics, and relations, benefiting several downstream tasks.
- •
  We propose a Chronological Group-wise Segment Merging (CGSM) strategy and an Instance Feature Aggregation (IFA) method to obtain accurate semantic objects.
- •
  We conduct extensive experiments, demonstrating the versatility, effectiveness, and efficiency of our method.

## 2 Related Work

Foundation Models. Recently, foundation models have exhibited impressive zero-shot perception ability. Here, we review several foundation models related to our work. CLIP \[[33](#bib.bib14)\] is a popular vision-language model that associates images and texts through contrastive learning, significantly promoting many vision-language tasks. SAM \[[18](#bib.bib16)\] is a class-agnostic 2D segmentation model trained with over 1 billion masks, demonstrating powerful zero-shot performance. OVSeg \[[22](#bib.bib17)\] finetunes CLIP to gain the ability of open-vocabulary semantic segmentation. CropFormer \[[32](#bib.bib18)\] fuses the full image and high-resolution image crops to improve segmentation performance. TAP \[[30](#bib.bib15)\] can simultaneously conduct recognition, segmentation, and caption generation. Additionally, many other methods \[[20](#bib.bib33), [6](#bib.bib34), [45](#bib.bib35), [8](#bib.bib36), [28](#bib.bib37), [24](#bib.bib38), [35](#bib.bib40), [10](#bib.bib43), [14](#bib.bib44), [43](#bib.bib45)\] are proposed for 2D open-vocabulary object detection and segmentation.

![Refer to caption](https://arxiv.org/html/2411.16253v2/2411.16253v2/overview1115.png)

Figure 2: Overview of our Octree-Graph. (a) Chronological Group-wise Segment Merging (CGSM). Given posed RGB-D inputs, 2D masks with semantic features are first extracted and then projected into the 3D space, where CGSM is conducted to merge segments. (b) Instance Feature Aggregation (IFA). Feature aggregation is performed for each merged object, which considers both intra- and inter-object similarity. (c) The Octree-Graph is constructed to efficiently and accurately represent the scene, facilitating various downstream tasks.

Open-Vocabulary 3D Scene Understanding. Based on the organization form of scene representation, we categorize these works into four types. 1) NeRF/Gaussian 3D mapping. These methods perform 3D scene understanding and scene/object reconstruction simultaneously, *e.g.*, OpenObj \[[3](#bib.bib52)\]. Although achieving good performance, they need extra effort to train the NeRF or 3D Gaussian models. 2) point/grid-wise 3D mapping. This branch involves directly projecting semantic features to each 3D point. OpenScene \[[31](#bib.bib19)\] and ConceptFusion \[[15](#bib.bib5)\] extract CLIP features from the images and densely project them to the point cloud. VLMaps \[[12](#bib.bib20)\] adopts a similar pipeline to project visual-language features to a grid-based BEV map. 3) instance-wise 3D mapping. These works explicitly obtain each 3D instance and fuse its visual-language features for 3D mapping. OpenIns3D \[[13](#bib.bib21)\] is a 3D-input-only framework that gets objects by open-vocabulary 3D detection. OVIR-3D \[[25](#bib.bib10)\], SAM3D \[[51](#bib.bib11)\], and MaskClustering \[[48](#bib.bib13)\] follow a 2D-to-3D pipeline where 2D masks are projected into 3D space for instance merging based on semantic similarity and 3D overlap. SAI3D \[[53](#bib.bib12)\] uses both 2D proposals and 3D super points for instance segmentation. OpenMask3D \[[40](#bib.bib9)\], Open3DIS \[[29](#bib.bib25)\], and SA3DIP \[[50](#bib.bib50)\] leverage extra 3D instance detectors to get more accurate object proposals. However, the used 3D models cannot be considered purely zero-shot methods. 4) 3D scene graph/octree. A few works use a graph or octree to organize the scene. ConceptGraph \[[7](#bib.bib6)\] and Clio \[[27](#bib.bib55)\] cluster object segments and construct a scene graph to enhance spatial reasoning. HOV-SG \[[44](#bib.bib7)\] proposes a hierarchical 3D scene graph to enable scene representation of different granularities. OctreeOcc \[[26](#bib.bib53)\] and PlenOctrees \[[54](#bib.bib54)\] use the octree structure to store semantic class and rendering information, respectively. In contrast, our Octree-Graph represents each object using an adaptive-octree and models their relations using a graph, supporting efficient occupancy queries and spatial reasoning.

## 3 Method

### 3.1 Framework Overview

As shown in Fig [2](#S2.F2 "Figure 2 ‣ 2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), given a sequence of RGB images $\mathcal{I}_{c}=\{\mathbf{I}_{t}^{c}\}_{t=1}^{T}$ and depth images $\mathcal{I}_{d}=\{\mathbf{I}_{t}^{d}\}_{t=1}^{T}$ scanned in a scene, we first leverage VLMs to extract segment proposals (§ [3.2](#S3.SS2 "3.2 Segment Proposal and Comprehension ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding")). Next, we chronologically merge these segments into an instance map $\mathcal{M}$ via a group-wise merging strategy (§ [3.3](#S3.SS3 "3.3 Chronological Group-wise Segment Merging ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding")). Then we dynamically aggregate the redundant semantics of each instance into a distinctive feature (§ [3.4](#S3.SS4 "3.4 Instance Feature Aggregation ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding")). Finally, we build an Octree-Graph $G$ to represent spatial relations among instances, with the adaptive-octree to detail instance occupancy. Based on this, we implemented LLM-based object retrieval and path planning algorithms (§ [3.5](#S3.SS5 "3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding")).

### 3.2 Segment Proposal and Comprehension

For each frame $\mathbf{I}^{c}_{t}$ at time $t$, we first adopt an off-the-shelf proposal generator, *e.g*., CropFormer \[[32](#bib.bib18)\], to extract a set of 2D masks $\mathcal{P}_{t}^{2d}=\{\mathbf{m}_{i}\}_{i=1}^{n_{t}}$, where $n_{t}$ is the mask number. We then filter out tiny and marginal masks to ensure the proposal quality. Next, each $\mathbf{m}_{i}$ is fed into the visual encoder and caption generator to obtain the visual feature $\mathbf{f}_{i}^{v}$ and caption feature $\mathbf{f}_{i}^{c}$. Finally, we project each mask $\mathbf{m}_{i}$ into the 3D space as a point cloud segment and perform DBSCAN \[[37](#bib.bib39)\] denoise, obtaining segments $\mathcal{P}_{t}^{3d}=\{\mathcal{S}_{i}\}_{i=1}^{n_{t}}$.

### 3.3 Chronological Group-wise Segment Merging

Existing segment merging strategies are typically categorized into two types: 1) frame-wise, which sequentially or hierarchically merges the adjacent frames \[[7](#bib.bib6), [25](#bib.bib10)\], integrating similar segments efficiently. 2) graph-wise, which merges segments across all frames \[[44](#bib.bib7), [48](#bib.bib13)\]. These methods have achieved great success, while the former solely relying on a single frame can be easily affected by proposal noises, *e.g*., associating unrelated instances once an under-segment is merged. The latter, which processes all segments together, may introduce redundant computations and be affected by irrelevant segments. To this end, we propose a Chronological Group-wise Segment Merging (CGSM) strategy with semantic-guided under-segment filtering and a dynamic threshold decay strategy.

![Refer to caption](https://arxiv.org/html/2411.16253v2/2411.16253v2/CSGM.png)

Figure 3: Illustration of group split and CGSM merging.

Chronological Group-Wise Split. Given the prior that an instance often appears in multiple consecutive frames, as shown in Fig. [3](#S3.F3 "Figure 3 ‣ 3.3 Chronological Group-wise Segment Merging ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), CGSM first partitions all frames into several groups in time order with interval $I$, obtaining the set of segments $\mathcal{G}_{i}$ for each group. In this way, a group can retain the adjacent segments while avoiding interference caused by global segments. Based on these groups, we perform iterations of merging to integrate separate segments into an instance map $\mathcal{M}$. Concretely, we start by merging $\mathcal{G}_{0}$ into an intermediate instance map $\mathcal{M}_{0}$. Subsequently, we iteratively take the union $\{\mathcal{M}_{k-1},\mathcal{G}_{k}\}$ as input for the $k^{\mathrm{th}}$ merging, until the final instance map $\mathcal{M}$ is constructed. Next, we elaborate on the details of a single merging step.

Segment Group Merging. For two segments $\{\mathcal{S}_{m},\mathcal{S}_{n}\}$, we define $\phi_{\mathrm{sem}}^{{v}}(m,n)$ as the cosine similarity between their visual features, and $\phi_{\mathrm{sem}}^{{c}}(m,n)$ the cosine similarity between caption features. Regarding geometric similarity, we compute $\phi_{\mathrm{geo}}^{\mathrm{iou}}(m,n)$ as the intersection over union of two segments. Additionally, we calculate the ratio of $\mathcal{S}_{n}$ contained within $\mathcal{S}_{m}$ as $\phi_{\mathrm{geo}}^{\mathrm{ior}}(m,n)=\left|\mathcal{S}_{m}\cap\mathcal{S}_{n}\right|/\left|\mathcal{S}_{n}\right|$. $\left|\cdot\right|$ denotes the amount of points in a 3D segment. Intuitively, assuming $\mathcal{S}_{m}$ is an under-segment containing a correct segment $\mathcal{S}_{n}$, $\phi_{\mathrm{geo}}^{\mathrm{ior}}(m,n)$ will be relatively large. Based on this, we can collect all segments contained in $\mathcal{S}_{m}$ as $\{\mathcal{S}_{j}~|~\phi_{\mathrm{geo}}^{\mathrm{ior}}(m,j)\geq 0.8\}$. If the semantic feature variance of these contained segments exceeds a threshold $\tau_{u}$, it indicates that $\mathcal{S}_{m}$ is probably an under-segment containing different objects, and $\mathcal{S}_{m}$ will be filtered out. We term this process as semantic-guided under-segment filtering. To merge the left segments, we compute an overall similarity $\phi=\phi_{\mathrm{geo}}^{\mathrm{iou}}+\phi_{\mathrm{geo}}^{\mathrm{ior}}+\phi_{\mathrm{sem}}^{{v}}+\phi_{\mathrm{sem}}^{{c}}$, and iteratively merge segments within group $\mathcal{G}_{i}$. At each iteration, we merge highly similar segments satisfying $\phi(m,n)\geq\theta_{i}$. However, simply doing so struggles to merge partially observed segments or over-segments sharing low spatial similarity. To this end, we linearly decay $\theta_{i}$ at each step inspired by \[[44](#bib.bib7), [53](#bib.bib12)\].

### 3.4 Instance Feature Aggregation

After obtaining the instance map $\mathcal{M}$, each 3D instance $\mathcal{O}_{i}$ in $\mathcal{M}$ is associated with multiple segment features $\mathcal{F}_{i}=\{\mathbf{f}_{i,j}^{v}~|~\mathbf{f}_{i,j}^{v}\in\mathcal{O}_{i}\}$ based on 2D-3D relations (for simplicity, we omit caption features here). To aggregate these features, previous methods either perform averaging \[[25](#bib.bib10), [7](#bib.bib6)\] or select the dominant feature via clustering \[[44](#bib.bib7), [40](#bib.bib9)\]. However, they overlook the distinction between different instance features. Hence, we propose a weighted average method to fuse an instance’s features for an optimal feature both representative and distinctive, as shown in Fig. [2](#S2.F2 "Figure 2 ‣ 2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding") (b). Specifically, taking the visual modality for illustration, we average $\mathcal{F}_{i}$ to a central feature $\bar{\mathbf{f}}_{i}^{v}$ for each instance, and the neighboring instances of $\mathcal{O}_{i}$ are then defined by $\mathcal{N}_{i}=\{\mathcal{O}_{k}~|~\mathrm{cos}\left(\bar{\mathbf{f}}_{i}^{v},\bar{\mathbf{f}}_{k}^{v}\right)\geq\tau_{d}\}$. Based on this, we aggregate the features $\mathcal{F}_{i}$ into an optimal ${\mathbf{f}_{i}^{v}}^{*}$ via assigning a dynamic fusion weight $a_{i,j}^{v}$ to each $\mathbf{f}_{i,j}^{v}$ in $\mathcal{F}_{i}$:

$$
a_{i,j}^{v}=\mathrm{cos}\left(\mathbf{f}_{i,j}^{v},\bar{\mathbf{f}}_{i}^{v}\right)-\sum_{O_{k}\in\mathcal{N}_{i}}{\mathrm{cos}\left(\mathbf{f}_{i,j}^{v},\bar{\mathbf{f}}_{k}^{v}\right)},
$$

(1)

where $\mathrm{cos}(\cdot)$ denotes cosine similarity, and $a_{i,j}^{v}$ is normalized via softmax. Intuitively, a feature gets a larger weight if it is closer to its own cluster center and farther from neighboring instances. The caption feature ${\mathbf{f}_{i}^{c}}^{*}$ can be formulated by replacing $\mathbf{f}_{i,j}^{v}$ with $\mathbf{f}_{i,j}^{c}$ in the above process. The final instance feature $\mathbf{f}_{i}^{*}$ is the average of ${\mathbf{f}_{i}^{v}}^{*}$ and ${\mathbf{f}_{i}^{c}}^{*}$.

![Refer to caption](https://arxiv.org/html/2411.16253v2/2411.16253v2/method_octree.png)

Figure 4: Illustration of the nodes and edges in Octree-Graph.

### 3.5 Octree-Graph Construction and Applications

To efficiently and accurately represent a scene, we design a hybrid structure, termed Octree-Graph. This structure utilizes a graph as the high-level architecture to organize objects and their spatial relations. Furthermore, we propose an adaptive-octree to depict the occupancy information of each object, which acts as a node of the Octree-Graph.

Graph Construction. An Octree-Graph can be defined as $G$ with nodes $\mathbf{N_{*}}$ and edges $\mathbf{E_{*}}$. The node $\mathbf{N}_{i}$ consists of correlated semantics $n_{i}^{s}$ (*e.g*., captions and features), center $n_{i}^{c}$, and adaptive-octree $n_{i}^{o}$. While the edge $\mathbf{E}_{i,j}$ comprises the semantic relation $e_{i,j}^{s}$, spatial distance $e_{i,j}^{d}$ and the 3D vector $\mathbf{e}_{i,j}^{v}$ between node $i$ and node $j$. Notably, the semantic relations between nodes are aligned with the world coordinate system of the corresponding point cloud. As shown in Fig. [4](#S3.F4 "Figure 4 ‣ 3.4 Instance Feature Aggregation ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), the semantic relation $e_{i,j}^{s}$ between node $\mathbf{N}_{i}$ and $\mathbf{N}_{j}$ is characterized as “right”.

Adaptive-Octree Construction. The classical octree \[[11](#bib.bib23)\] is a tree-based structure capable of efficiently representing a 3D space with much less storage requirements than point clouds. During octree construction, the root node is defined by the minimal bounding box containing the point cloud. This box, centered at $c\in\mathbb{R}^{3}$ with a side length of $d$, is divided into eight sub-regions of side length $d/2$ using axis-aligned planes. Each sub-region serves as a child node, and the process continues recursively for each node until the desired octree depth $L_{\mathrm{max}}$ is reached or no point clouds are present within the node. We recommend referring to \[[11](#bib.bib23)\] for more details about the octree.

However, the traditional octree is proposed to represent an entire 3D space, which uses cubic voxels as units to depict occupancy details. This leads to dilemmas of redundant representation when depicting an object, *e.g*., an object with a large aspect ratio requires a very deep octree to approximate its shape. To this end, we propose the adaptive-octree with varying voxels that adaptively adjust their sizes and shapes according to the object’s shape. As shown in Fig. [5](#S3.F5 "Figure 5 ‣ 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), an adaptive-octree is constructed from an instance-level point cloud $P$. The size of each node in this adaptive-octree can be computed as follows:

$$
\mathbf{d}_{l}=\left(\mathbf{b}_{\mathrm{max}}-\mathbf{b}_{\mathrm{min}}\right)/{2^{l}},
$$

(2)

where $\mathbf{b}_{\mathrm{max}}$ and $\mathbf{b}_{\mathrm{min}}$ are the coordinates of the lower left corner and the upper right corner of $P$’s bounding box. $l\in\left\{1,2,\cdots,L_{\mathrm{max}}\right\}$ denotes the depth of the adaptive-octree. As shown in Fig. [5](#S3.F5 "Figure 5 ‣ 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), the center $\mathbf{c}_{l}\in\mathbb{R}^{3}$ of the $l$-th layer node can be determined by the center $\mathbf{c}_{l-1}$ of the parent node and the edge length $\mathbf{d}_{l}$ of the current node. The adaptive-octree can be quickly constructed from the point cloud.

![Refer to caption](https://arxiv.org/html/2411.16253v2/2411.16253v2/octree_exp_final.png)

Figure 5: Illustration of the construction of the adaptive-octree. The above displays the process, and the below shows an example.

Octree-Graph Applications. Based on the Octree-Graph, we offer object retrieval and path planning functionalities which are critical for embodied agents. For object retrieval, two types of queries are supported, *i.e.,* $\mathrm{Query}\left(\mathit{target}\right)$ and $\mathrm{Query}\left(\mathit{reference},\mathit{relation},\mathit{target}\right)$. The former allows for directly locating an object by comparing the similarity between queries and stored semantics. The latter supports complex queries by sequentially locating the reference object, the edge that matches the described relation, and finally the target. For more complex queries, we leverage the reasoning capabilities of LLMs to decompose the task and flexibly call two types of functions to achieve the goal.

In path planning tasks, querying occupancy information is fundamental. The proposed Octree-Graph supports such queries, enabling us to easily implement path planning algorithms like classical $A^{*}$ \[[9](#bib.bib24)\] and the recent \[[52](#bib.bib51)\].

|                                   |               | Replica               |                         |                       | ScanNet               |                         |                       |
|-----------------------------------|---------------|-----------------------|-------------------------|-----------------------|-----------------------|-------------------------|-----------------------|
| Method                            | CLIP Backbone | mIoU$\uparrow$ | F-mIoU$\uparrow$ | mAcc$\uparrow$ | mIoU$\uparrow$ | F-mIoU$\uparrow$ | mAcc$\uparrow$ |
| ConceptFusion \[[15](#bib.bib5)\] | OVSeg         | 0.10                  | 0.21                    | 0.16                  | 0.08                  | 0.11                    | 0.15                  |
|                                   | Vit-H-14      | 0.10                  | 0.18                    | 0.17                  | 0.11                  | 0.12                    | 0.21                  |
| ConceptGraph \[[7](#bib.bib6)\]   | OVSeg         | 0.13                  | 0.27                    | 0.21                  | 0.15                  | 0.18                    | 0.23                  |
|                                   | Vit-H-14      | 0.18                  | 0.23                    | 0.30                  | 0.16                  | 0.20                    | 0.28                  |
| HOV-SG \[[44](#bib.bib7)\]        | OVSeg         | 0.144                 | 0.255                   | 0.212                 | 0.214                 | 0.258                   | 0.420                 |
|                                   | Vit-H-14      | 0.231                 | 0.386                   | 0.304                 | 0.222                 | 0.303                   | 0.431                 |
| Ours                              | OVSeg         | 0.320                 | 0.553                   | 0.414                 | 0.393                 | 0.508                   | 0.601                 |
|                                   | Vit-H-14      | 0.263                 | 0.479                   | 0.387                 | 0.356                 | 0.477                   | 0.574                 |

Table 1: Zero-shot 3D semantic segmentation results on Replica and ScanNet benchmark.

|                                         |                     |                       |                       |
|-----------------------------------------|---------------------|-----------------------|-----------------------|
| Method                                  | AP$\uparrow$ | AP50$\uparrow$ | AP25$\uparrow$ |
| sup. mask + sup. semantic               |                     |                       |                       |
| Mask3D \[[38](#bib.bib8)\]              | 26.9                | 36.2                  | 41.4                  |
| sup. mask + z.s. semantic               |                     |                       |                       |
| Open3DIS \[[29](#bib.bib25)\]           | 23.7                | 29.4                  | 32.8                  |
| Open3DIS \[[29](#bib.bib25)\] (3D only) | 18.6                | 23.1                  | 27.3                  |
| OpenMask3D \[[40](#bib.bib9)\] (Mask3D) | 15.4                | 19.9                  | 23.1                  |
| Ours (Mask3D)                           | 23.2                | 30.3                  | 33.3                  |
| z.s. mask + z.s. semantic               |                     |                       |                       |
| OVIR-3D \[[25](#bib.bib10)\]            | 9.3                 | 18.7                  | 25.0                  |
| SAM3D \[[51](#bib.bib11)\]              | 9.8                 | 15.2                  | 20.7                  |
| SAI3D \[[53](#bib.bib12)\]              | 12.7                | 18.8                  | 24.1                  |
| Mask-Clustering \[[48](#bib.bib13)\]    | 12.0                | 23.3                  | 30.1                  |
| Ours                                    | 14.3                | 25.8                  | 33.6                  |

Table 2: 3D instance segmentation results on ScanNet200. $sup.$ means supervised training, $z.s.$ denotes the zero-shot setting.

<table>
<colgroup>
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
<col style="width: 25%" />
</colgroup>
<thead class="ltx_thead">
<tr id="S3.T3.3.1" class="header ltx_tr">
<th id="S3.T3.3.1.1" class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt" style="padding-left: 4.3pt; padding-right: 4.3pt">Method</th>
<th id="S3.T3.3.1.2" class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt" style="padding-left: 4.3pt; padding-right: 4.3pt"><table>
<tbody>
<tr id="S3.T3.3.1.2.1.1" class="odd ltx_tr">
<td id="S3.T3.3.1.2.1.1.1" class="ltx_td ltx_nopad_r ltx_align_left" style="padding-left: 4.3pt; padding-right: 4.3pt">Retrieval</td>
</tr>
<tr id="S3.T3.3.1.2.1.2" class="even ltx_tr">
<td id="S3.T3.3.1.2.1.2.1" class="ltx_td ltx_nopad_r ltx_align_left" style="padding-left: 4.3pt; padding-right: 4.3pt">Algorithm</td>
</tr>
</tbody>
</table></th>
<th id="S3.T3.3.1.3" class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt" style="padding-left: 4.3pt; padding-right: 4.3pt"><table>
<tbody>
<tr id="S3.T3.3.1.3.1.1" class="odd ltx_tr">
<td id="S3.T3.3.1.3.1.1.1" class="ltx_td ltx_nopad_r ltx_align_left" style="padding-left: 4.3pt; padding-right: 4.3pt">Acc$\uparrow$</td>
</tr>
<tr id="S3.T3.3.1.3.1.2" class="even ltx_tr">
<td id="S3.T3.3.1.3.1.2.1" class="ltx_td ltx_nopad_r ltx_align_left" style="padding-left: 4.3pt; padding-right: 4.3pt">@0.1</td>
</tr>
</tbody>
</table></th>
<th id="S3.T3.3.1.4" class="ltx_td ltx_nopad_r ltx_align_left ltx_th ltx_th_column ltx_border_tt" style="padding-left: 4.3pt; padding-right: 4.3pt"><table>
<tbody>
<tr id="S3.T3.3.1.4.1.1" class="odd ltx_tr">
<td id="S3.T3.3.1.4.1.1.1" class="ltx_td ltx_nopad_r ltx_align_left" style="padding-left: 4.3pt; padding-right: 4.3pt">Acc$\uparrow$</td>
</tr>
<tr id="S3.T3.3.1.4.1.2" class="even ltx_tr">
<td id="S3.T3.3.1.4.1.2.1" class="ltx_td ltx_nopad_r ltx_align_left" style="padding-left: 4.3pt; padding-right: 4.3pt">@0.25</td>
</tr>
</tbody>
</table></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr id="S3.T3.3.2" class="odd ltx_tr">
<th id="S3.T3.3.2.1" class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t" style="padding-left: 4.3pt; padding-right: 4.3pt">ConceptGraphs [<a href="#bib.bib6">7</a>]</th>
<td id="S3.T3.3.2.2" class="ltx_td ltx_align_left ltx_border_t" style="padding-left: 4.3pt; padding-right: 4.3pt">Deductive [<a href="#bib.bib4">23</a>]</td>
<td id="S3.T3.3.2.3" class="ltx_td ltx_align_left ltx_border_t" style="padding-left: 4.3pt; padding-right: 4.3pt">0.15</td>
<td id="S3.T3.3.2.4" class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t" style="padding-left: 4.3pt; padding-right: 4.3pt">0.08</td>
</tr>
<tr id="S3.T3.3.3" class="even ltx_tr">
<th id="S3.T3.3.3.1" class="ltx_td ltx_align_left ltx_th ltx_th_row" style="padding-left: 4.3pt; padding-right: 4.3pt">Open-Fusion [<a href="#bib.bib48">47</a>]</th>
<td id="S3.T3.3.3.2" class="ltx_td ltx_align_left" style="padding-left: 4.3pt; padding-right: 4.3pt">CLIP [<a href="#bib.bib14">33</a>]</td>
<td id="S3.T3.3.3.3" class="ltx_td ltx_align_left" style="padding-left: 4.3pt; padding-right: 4.3pt">0.13</td>
<td id="S3.T3.3.3.4" class="ltx_td ltx_nopad_r ltx_align_left" style="padding-left: 4.3pt; padding-right: 4.3pt">0.02</td>
</tr>
<tr id="S3.T3.3.4" class="odd ltx_tr">
<th id="S3.T3.3.4.1" class="ltx_td ltx_align_left ltx_th ltx_th_row" style="padding-left: 4.3pt; padding-right: 4.3pt">BBQ [<a href="#bib.bib4">23</a>]</th>
<td id="S3.T3.3.4.2" class="ltx_td ltx_align_left" style="padding-left: 4.3pt; padding-right: 4.3pt">CLIP [<a href="#bib.bib14">33</a>]</td>
<td id="S3.T3.3.4.3" class="ltx_td ltx_align_left" style="padding-left: 4.3pt; padding-right: 4.3pt">0.10</td>
<td id="S3.T3.3.4.4" class="ltx_td ltx_nopad_r ltx_align_left" style="padding-left: 4.3pt; padding-right: 4.3pt">0.06</td>
</tr>
<tr id="S3.T3.3.5" class="even ltx_tr">
<th id="S3.T3.3.5.1" class="ltx_td ltx_align_left ltx_th ltx_th_row" style="padding-left: 4.3pt; padding-right: 4.3pt">BBQ [<a href="#bib.bib4">23</a>]</th>
<td id="S3.T3.3.5.2" class="ltx_td ltx_align_left" style="padding-left: 4.3pt; padding-right: 4.3pt">Deductive [<a href="#bib.bib4">23</a>]</td>
<td id="S3.T3.3.5.3" class="ltx_td ltx_align_left" style="padding-left: 4.3pt; padding-right: 4.3pt">0.23</td>
<td id="S3.T3.3.5.4" class="ltx_td ltx_nopad_r ltx_align_left" style="padding-left: 4.3pt; padding-right: 4.3pt">0.18</td>
</tr>
<tr id="S3.T3.3.6" class="odd ltx_tr">
<th id="S3.T3.3.6.1" class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb" style="padding-left: 4.3pt; padding-right: 4.3pt">Ours</th>
<td id="S3.T3.3.6.2" class="ltx_td ltx_align_left ltx_border_bb" style="padding-left: 4.3pt; padding-right: 4.3pt">Octree-Graph+LLM</td>
<td id="S3.T3.3.6.3" class="ltx_td ltx_align_left ltx_border_bb" style="padding-left: 4.3pt; padding-right: 4.3pt">0.26</td>
<td id="S3.T3.3.6.4" class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb" style="padding-left: 4.3pt; padding-right: 4.3pt">0.23</td>
</tr>
</tbody>
</table>

Table 3: Text-based object retrieval results on the Sr3D dataset.

| Method                     | SR($s$=1.0m) | SR($s$=0.5m) | SR($s$=0.25m) |
|----------------------------|----------------------------|----------------------------|-----------------------------|
| HOV-SG \[[44](#bib.bib7)\] | 55.25                      | 46.75                      | 32.16                       |
| Ours                       | 97.88                      | 96.88                      | 96.38                       |

Table 4: Path planning results on HM3DSem. SR denotes success rate (%). $s$ is the threshold within which the distance between the navigation endpoint and the destination is considered successful.

## 4 Experiment

To validate the versatility and effectiveness of our method, we carry out extensive experiments, including semantic segmentation, instance segmentation, text-based object retrieval, and path planning. We compare our method with different SOTA methods in these tasks, and conduct comprehensive ablation studies to investigate several key components, demonstrating the effectiveness of our designs.

### 4.1 Implementation Details

We use CropFormer \[[32](#bib.bib18)\] as the 2D proposal generator following \[[48](#bib.bib13)\]. To extract visual features, we test two commonly used VLMs, *i.e.,* CLIP ViT-H \[[33](#bib.bib14)\] and OVSeg ViT-L \[[22](#bib.bib17)\]. We adopt TAP \[[30](#bib.bib15)\] for generating the mask caption. Additionally, we filtered out masks with pixels less than $25$ and segments with points less than $50$. The similarity threshold $\tau_{d}$ is empirically set to $0.7$. The group split interval $I$, the under-segment filtering threshold $\tau_{u}$, and the decay parameter $\theta_{i}$ are set to 200, 0.02, and 0.8 through hyper-parameter experiments. Considering the dimensions of indoor objects, we set the maximum depth $L_{\mathrm{max}}$ of the adaptive-octree to 4.

### 4.2 Dataset and Evaluation Metrics

Dataset. For zero-shot 3D semantic segmentation, we evaluate our method on common scenes following \[[15](#bib.bib5), [7](#bib.bib6), [44](#bib.bib7)\], i.e., $8$ scenes from Replica \[[39](#bib.bib3)\] dataset and $5$ scenes from ScanNet \[[2](#bib.bib1)\]. For zero-shot 3D instance segmentation, we assess our method on the widely-used ScanNet200 \[[36](#bib.bib2)\] benchmark, including a validation set of $312$ scenes with $200$ categories. For text-based object retrieval, we test our method on Sr3D \[[1](#bib.bib46)\] dataset, and follow the experiment setting of BBQ \[[23](#bib.bib4)\] that subsampled 526 free-form queries from 8 scenes. For the path planning task, we employ the HM3DSem \[[46](#bib.bib47)\] dataset used in HOV-SG \[[44](#bib.bib7)\], where 8 scenes are selected for evaluation. Moreover, we also conduct real-world experiments to validate our effectiveness.

Evaluation Metrics. Following the mainstream evaluation metrics \[[44](#bib.bib7)\], we assess 3D semantic segmentation results via commonly used mean IoU (mIoU), frequency-weighted mean IoU (F-mIoU), and mean Accuracy (mAcc). For 3D instance segmentation, we report the standard Average Precision (AP) at IoU thresholds 25$\%$ and 50$\%$, along with the mean of AP from 50$\%$ to 95$\%$ at 5$\%$ interval. For text-based object retrieval, we follow BBQ \[[23](#bib.bib4)\], using Acc@0.1 and Acc@0.25 as evaluation metrics where retrieval is treated as a true positive if the IoU between the predicted object’s bounding box and the ground-truth bounding box surpasses 0.1 and 0.25, respectively. For the path planning task, we randomly select positions in the empty areas of a scene as the starting point and destination. When the endpoint of navigation is within a threshold $s$ (*i.e.*, 1m, 0.5m, and 0.25m) from the destination, the path planning is considered successful.

Besides, to quantify spatial representation accuracy, we introduce the Effective Occupancy Ratio (EOR) as a metric. The occupancy range $O_{\mathrm{pc}}$ of a point cloud is determined by expanding this point cloud with a dilation $\bigtriangleup r=0.005$, and the occupancy range of the octree is denoted as $O_{\mathrm{oct}}$. Then, the EOR is calculated as $\mathrm{EOR}=\frac{O_{\mathrm{oct}}\cap O_{\mathrm{pc}}}{O_{\mathrm{oct}}}$. We denote the mean EOR for all objects in a scene as mEOR.

### 4.3 Quantitative Comparison

3D Semantic Segmentation. Tab. [1](#S3.T1 "Table 1 ‣ 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding") reports the numerical results for zero-shot 3D semantic segmentation on Replica and ScanNet datasets. In this experiment, we compare the results generated by our CGSM and IFA with other works. It can be seen that our method significantly outperforms existing methods across all metrics on both datasets, demonstrating the effectiveness of the proposed CGSM and IFA. Compared to the existing SoTA 3D scene graph, HOV-SG \[[44](#bib.bib7)\], we achieve +8.9% mIoU and +11.0% mAcc on the Replica dataset. Similarly, we present a +17.1% mIoU and a +17.0% mAcc on ScanNet with the same settings.

3D Instance Segmentation. The quantitative results for 3D instance segmentation are shown in Tab. [2](#S3.T2 "Table 2 ‣ 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"). We follow \[[48](#bib.bib13)\] to categorize all methods into 3 groups based on whether the proposal generation and semantic prediction are trained. Under the fully zero-shot setting, our method surpasses the previous most advanced method with gains of 2.3%, 2.5%, and 3.5% in AP, AP25 and AP50, respectively. These results further demonstrate the effectiveness of the proposed CGSM and IFA. When using supervised 3D models for proposal generation, our method significantly outperforms OpenMask3D \[[40](#bib.bib9)\] and the Open3DIS \[[29](#bib.bib25)\] variant with only the 3D proposals, validating the superiority of our feature aggregation method IFA. Besides, our method achieves comparable results with the corresponding SOTA method Open3DIS \[[29](#bib.bib25)\], which specially designs a combination of 2D and 3D proposals.

Text-based Object Retrieval. Tab. [3](#S3.T3 "Table 3 ‣ 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding") presents the comparison results of text-based object retrieval on Sr3D \[[1](#bib.bib46)\] dataset. Our method outperforms the SOTA method BBQ \[[23](#bib.bib4)\] by 3.0% and 5.0% in terms of Acc@0.1 and Acc@0.25, respectively. This experiment veriffes the quality of our con-structed graph. We attribute the performance gain to our accurate semantic object segmentation and the rich relations stored in the Octree-Graph.

Path Planning. For each sense in the HM3DSem \[[46](#bib.bib47)\] dataset, we randomly select 100 pairs of starting points and destinations in navigable areas. HOV-SG \[[44](#bib.bib7)\] can be directly used for path planning, thus it is evaluated and compared with our method in this task. Tab. [4](#S3.T4 "Table 4 ‣ 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding") shows the results, from which we can see that our method significantly surpasses HOV-SG, especially when the threshold $s$ is small. This is because HOV-SG relies on Voronoi graph \[[41](#bib.bib49)\] for path planning, where the waypoints and paths are pre-calculated, making it improper for precise navigation. In contrast, our Octree-Graph supports navigation to any empty area, unless the destination is mistakenly occupied by the adaptive-octree.

![Refer to caption](https://arxiv.org/html/2411.16253v2/2411.16253v2/visualize-small.png)

Figure 6: Visual comparisons. (a) Semantic segmentation results on Replica. (b) Instance segmentation results on ScanNet200.

| Merging Strategy                  | mIoU$\uparrow$ | F-mIoU$\uparrow$ | mAcc$\uparrow$ |
|-----------------------------------|-----------------------|-------------------------|-----------------------|
| Frame-wise                        | 0.323                 | 0.439                   | 0.519                 |
| Global-wise                       | 0.286                 | 0.414                   | 0.476                 |
| Ours (CGSM) $I$=400 | 0.338                 | 0.463                   | 0.555                 |
| Ours (CGSM) $I$=200 | 0.356                 | 0.477                   | 0.574                 |
| Ours (CGSM) $I$=100 | 0.344                 | 0.462                   | 0.571                 |

Table 5: Ablation study on the segment merging strategy and different temporal intervals for group partitioning of our CGSM.

<table>
<colgroup>
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 20%" />
</colgroup>
<thead class="ltx_thead">
<tr id="S4.T6.3.1" class="header ltx_tr">
<th id="S4.T6.3.1.1" class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt" style="padding-left: 2.8pt; padding-right: 2.8pt"><table>
<tbody>
<tr id="S4.T6.3.1.1.1.1" class="odd ltx_tr">
<td id="S4.T6.3.1.1.1.1.1" class="ltx_td ltx_nopad_r ltx_align_center" style="padding-left: 2.8pt; padding-right: 2.8pt">under-segment</td>
</tr>
<tr id="S4.T6.3.1.1.1.2" class="even ltx_tr">
<td id="S4.T6.3.1.1.1.2.1" class="ltx_td ltx_nopad_r ltx_align_center" style="padding-left: 2.8pt; padding-right: 2.8pt">filtering</td>
</tr>
</tbody>
</table></th>
<th id="S4.T6.3.1.2" class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt" style="padding-left: 2.8pt; padding-right: 2.8pt"><table>
<tbody>
<tr id="S4.T6.3.1.2.1.1" class="odd ltx_tr">
<td id="S4.T6.3.1.2.1.1.1" class="ltx_td ltx_nopad_r ltx_align_center" style="padding-left: 2.8pt; padding-right: 2.8pt">threshold</td>
</tr>
<tr id="S4.T6.3.1.2.1.2" class="even ltx_tr">
<td id="S4.T6.3.1.2.1.2.1" class="ltx_td ltx_nopad_r ltx_align_center" style="padding-left: 2.8pt; padding-right: 2.8pt">decay</td>
</tr>
</tbody>
</table></th>
<th id="S4.T6.3.1.3" class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" style="padding-left: 2.8pt; padding-right: 2.8pt">mIoU$\uparrow$</th>
<th id="S4.T6.3.1.4" class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt" style="padding-left: 2.8pt; padding-right: 2.8pt">F-IoU$\uparrow$</th>
<th id="S4.T6.3.1.5" class="ltx_td ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_tt" style="padding-left: 2.8pt; padding-right: 2.8pt">mAcc$\uparrow$</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr id="S4.T6.3.2" class="odd ltx_tr">
<th id="S4.T6.3.2.1" class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t" style="padding-left: 2.8pt; padding-right: 2.8pt">✗</th>
<th id="S4.T6.3.2.2" class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t" style="padding-left: 2.8pt; padding-right: 2.8pt">✗</th>
<td id="S4.T6.3.2.3" class="ltx_td ltx_align_center ltx_border_t" style="padding-left: 2.8pt; padding-right: 2.8pt">0.337</td>
<td id="S4.T6.3.2.4" class="ltx_td ltx_align_center ltx_border_t" style="padding-left: 2.8pt; padding-right: 2.8pt">0.460</td>
<td id="S4.T6.3.2.5" class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t" style="padding-left: 2.8pt; padding-right: 2.8pt">0.547</td>
</tr>
<tr id="S4.T6.3.3" class="even ltx_tr">
<th id="S4.T6.3.3.1" class="ltx_td ltx_align_center ltx_th ltx_th_row" style="padding-left: 2.8pt; padding-right: 2.8pt">✓</th>
<th id="S4.T6.3.3.2" class="ltx_td ltx_align_center ltx_th ltx_th_row" style="padding-left: 2.8pt; padding-right: 2.8pt">✗</th>
<td id="S4.T6.3.3.3" class="ltx_td ltx_align_center" style="padding-left: 2.8pt; padding-right: 2.8pt">0.346</td>
<td id="S4.T6.3.3.4" class="ltx_td ltx_align_center" style="padding-left: 2.8pt; padding-right: 2.8pt">0.471</td>
<td id="S4.T6.3.3.5" class="ltx_td ltx_nopad_r ltx_align_center" style="padding-left: 2.8pt; padding-right: 2.8pt">0.557</td>
</tr>
<tr id="S4.T6.3.4" class="odd ltx_tr">
<th id="S4.T6.3.4.1" class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb" style="padding-left: 2.8pt; padding-right: 2.8pt">✓</th>
<th id="S4.T6.3.4.2" class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb" style="padding-left: 2.8pt; padding-right: 2.8pt">✓</th>
<td id="S4.T6.3.4.3" class="ltx_td ltx_align_center ltx_border_bb" style="padding-left: 2.8pt; padding-right: 2.8pt">0.356</td>
<td id="S4.T6.3.4.4" class="ltx_td ltx_align_center ltx_border_bb" style="padding-left: 2.8pt; padding-right: 2.8pt">0.477</td>
<td id="S4.T6.3.4.5" class="ltx_td ltx_nopad_r ltx_align_center ltx_border_bb" style="padding-left: 2.8pt; padding-right: 2.8pt">0.574</td>
</tr>
</tbody>
</table>

Table 6: Ablations study on the strategies for segment merging.

| Aggregation Strategy | mIoU$\uparrow$ | F-mIoU$\uparrow$ | mAcc$\uparrow$ |
|----------------------|-----------------------|-------------------------|-----------------------|
| Average              | 0.338                 | 0.453                   | 0.563                 |
| Top-5                | 0.342                 | 0.470                   | 0.570                 |
| DBSCAN               | 0.345                 | 0.459                   | 0.566                 |
| Ours (IFA)           | 0.356                 | 0.477                   | 0.574                 |

Table 7: Ablation study on various feature aggregation strategies.

|                 |                          |                       |                          |                       |
|-----------------|--------------------------|-----------------------|--------------------------|-----------------------|
|                 | Replica                  |                       | ScanNet                  |                       |
| Method          | Storage$\downarrow$ | mEOR$\uparrow$ | Storage$\downarrow$ | mEOR$\uparrow$ |
| Point cloud     | 18.5MB                   |                       | 6.4MB                    |                       |
| Octree          | 17.6KB                   | 0.0057                | 41.1KB                   | 0.0041                |
| Adaptive-Octree | 29.8KB                   | 0.0108                | 69.3KB                   | 0.0070                |

Table 8: Ablation study on the efficiency of the adaptive-octree.

| Method            | Structure    | Storage$\downarrow$ | time$\downarrow$ |
|-------------------|--------------|--------------------------|-----------------------|
| A\*               | Octree-Graph | 268.41Kb                 | 0.032s                |
| A\*               | Point Cloud  | 71.16Mb                  | 2.154s                |
| Jump Point Search | Octree-Graph | 268.41Kb                 | 0.081s                |
| Jump Point Search | Point Cloud  | 71.16Mb                  | 2.153s                |

Table 9: Ablation study on path planning efficiency.

### 4.4 Ablation Studies

We analyze the impact of our key designs via zero-shot semantic segmentation experiments on ScanNet.

Effect of Group-Wise Split. We compare the proposed Chronological Group-wise Segment Merging (CGSM) with the vanilla frame-wise and global-wise merging strategies. As shown in Tab. [5](#S4.T5 "Table 5 ‣ 4.3 Quantitative Comparison ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), we achieve notable gains compared to both methods, with +3.3% mIoU over frame-wise merging and +7.0% mIoU over global-wise merging. We also analyze the impact of hyper-parameter $I$, and the results in Rows 3-5 show that our method exhibits robustness to $I$ ranging from 100 to 400.

Analysis of Designs on Segment Merging. Tab. [6](#S4.T6 "Table 6 ‣ 4.3 Quantitative Comparison ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding") presents the results with two key components for merging a single group. Row $0$ serves as a fixed-threshold group-wise merging baseline with no extra design. Row $1$ is equipped with our semantic-guided under-segment filtering, and achieves +$\textbf{0.9}\%$ mIoU and +$\textbf{1.0}\%$ mAcc. Row $2$ further incorporates the threshold decay strategy, resulting in an additional $\textbf{1.0}\%$ mIoU and $\textbf{1.7}\%$ mAcc gains. These validate the effectiveness of the two strategies during each group-wise merging.

Effect of Instance Feature Aggregation. In Tab. [7](#S4.T7 "Table 7 ‣ 4.3 Quantitative Comparison ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), we compare the proposed Instance Feature Aggregation (IFA) method with several commonly used methods. For fairness, all methods use the same features as our IFA. Row $1$ simply averages features across all views, yielding unsatisfactory results. Row $2$ and Row $3$ leverage Top-5 criterion \[[38](#bib.bib8), [40](#bib.bib9)\] and DBSCAN algorithm \[[44](#bib.bib7)\] to select the predominant feature, achieving gains of +$\textbf{0.4}\%$ and +$\textbf{0.7}\%$ mIoU, respectively. By contrast, our IFA achieves an improvement of $\textbf{1.8}\%$ mIoU over Row $1$.

Adaptive-Octree Efficiency. Based on the results of instance generation in Replica and ScanNet, Tab. [8](#S4.T8 "Table 8 ‣ 4.3 Quantitative Comparison ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding") provides a comparison of different spatial representations with respect to storage space and the accuracy of occupancy. For the same scene, octree and our adaptive-octree consume two orders of magnitude less storage compared to point clouds. Our adaptive-octree requires a bit more storage than the traditional octree due to its additional record of bounding boxes for each object. However, at the same depth, the adaptive-octree exhibits a much higher mEOR compared to the octree. This means that the space described by the adaptive-octree is more closely aligned with the target regions. In summary, the adaptive-octree requires much less storage space than point clouds and provides more accurate occupancy information than a traditional octree.

Path Planning Efficiency based on Octree-Graph. To further verify the efficiency of our method, we conduct path planning experiments using A\* \[[9](#bib.bib24)\] and Jump Point Search \[[52](#bib.bib51)\] algorithms on the HM3DSem \[[46](#bib.bib47)\] dataset, where the point cloud representation and our Octree-Graph are compared. It can be seen from Tab. [9](#S4.T9 "Table 9 ‣ 4.3 Quantitative Comparison ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding") that Octree-Graph uses much less storage space and spends much less time. This is crucial for real-world deployment.

### 4.5 Qualitative Analysis

Visualization Results. Fig. [6](#S4.F6 "Figure 6 ‣ 4.3 Quantitative Comparison ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding") visualizes the results of semantic segmentation and instance segmentation, respectively. We can see that our method exhibits more accurate object semantics and fewer incorrect segments than comparison methods. Fig. [8](#S4.F8 "Figure 8 ‣ 4.5 Qualitative Analysis ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding") demonstrates the segment merging results of our CGSM and its baseline (*i.e.,* frame-wise sequential merging), where CGSM correctly resolves the over-segmented long table without introducing excessive merges between different objects.

Object Retrieval and Path Planning. We also conduct real-world experiments to further validate the effectiveness of our method. Fig. [7](#S4.F7 "Figure 7 ‣ 4.5 Qualitative Analysis ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding") presents the results where a real scene is set up and scanned by an Intel Realsense D435i camera. Then we reconstruct the colored point cloud and establish the Octree-Graph. Based on these, we deploy our method on a robotic dog and a drone with NVIDIA Orin NX as onboard computers. As shown in Fig. [7](#S4.F7 "Figure 7 ‣ 4.5 Qualitative Analysis ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), robots can accurately find the target and successfully navigate to it relying on our Octree-Graph. The dynamic process of this experiment can be found in the supplementary video demo.

![Refer to caption](https://arxiv.org/html/2411.16253v2/2411.16253v2/real_exp_final.png)

Figure 7: Visualization of the real-world experiment. The left column shows a real scene and the reconstructed colored point cloud with the retrieval target highlighted by our adaptive-octree. The right column presents path planning using a robotic dog and a drone based on our Octree-Graph.

![Refer to caption](https://arxiv.org/html/2411.16253v2/2411.16253v2/vis.png)

Figure 8: Segment merging comparison.

## 5 Conclusion

In this paper, we propose Octree-Graph, a novel scene representation for open-vocabulary 3D scene understanding. Specifically, an adaptive-octree structure is devised to characterize the occupancy of an object, which acts as the node of the Octree-Graph. The edges describe rich relations among objects for spatial reasoning. For Octree-Graph construction, we also develop a training-free pipeline to conduct semantic object segmentation, where a Chronological Group-wise Segment Merging (CGSM) strategy is designed to alleviate inaccurate segment proposals, and an Instance Feature Aggregation (IFA) method is devised to get a semantic feature both representative and distinctive. Extensive evaluations on several tasks validate the versatility and effectiveness of our Octree-Graph.

## Acknowledgments

This work is supported by the Shanghai AI Laboratory, the National Natural Science Foundation of China (62376222), and Young Elite Scientists Sponsorship Program by CAST (2023QNRC001).

## References

- \[1\] P. Achlioptas, A. Abdelreheem, F. Xia, M. Elhoseiny, and L. J. Guibas (2020) ReferIt3D: neural listeners for fine-grained 3d object identification in real-world scenes. In Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part I, Vol. 12346, pp. 422–440. Cited by: [§4.2](#S4.SS2.p1.1 "4.2 Dataset and Evaluation Metrics ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.3](#S4.SS3.p3.1 "4.3 Quantitative Comparison ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[2\] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner (2017) Scannet: richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5828–5839. Cited by: [§4.2](#S4.SS2.p1.1 "4.2 Dataset and Evaluation Metrics ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[3\] Y. Deng, J. Wang, J. Zhao, J. Dou, Y. Yang, and Y. Yue (2025) OpenObj: open-vocabulary object-level neural radiance fields with fine-grained understanding. IEEE Robotics Autom. Lett. 10 (1), pp. 652–659. Cited by: [§2](#S2.p2.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[4\] R. Ding, J. Yang, C. Xue, W. Zhang, S. Bai, and X. Qi (2023) Lowis3D: language-driven open-world instance-level 3d scene understanding. CoRR abs/2308.00353. Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[5\] R. Ding, J. Yang, C. Xue, W. Zhang, S. Bai, and X. Qi (2023) PLA: language-driven open-vocabulary 3d scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[6\] G. Ghiasi, X. Gu, Y. Cui, and T. Lin (2022) Scaling open-vocabulary image segmentation with image-level labels. In Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XXXVI, Vol. 13696, pp. 540–557. Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[7\] Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen, A. Agarwal, C. Rivera, W. Paul, K. Ellis, R. Chellappa, et al. (2023) Conceptgraphs: open-vocabulary 3d scene graphs for perception and planning. arXiv preprint arXiv:2309.16650. Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§2](#S2.p2.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§3.3](#S3.SS3.p1.1 "3.3 Chronological Group-wise Segment Merging ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§3.4](#S3.SS4.p1.1 "3.4 Instance Feature Aggregation ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [Table 1](#S3.T1.3.5.1.1 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [Table 3](#S3.T3.3.2.1 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.2](#S4.SS2.p1.1 "4.2 Dataset and Evaluation Metrics ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[8\] X. Gu, T. Lin, W. Kuo, and Y. Cui (2022) Open-vocabulary object detection via vision and language knowledge distillation. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[9\] P. E. Hart, N. J. Nilsson, and B. Raphael (1968) A formal basis for the heuristic determination of minimum cost paths. IEEE transactions on Systems Science and Cybernetics 4 (2), pp. 100–107. Cited by: [§3.5](#S3.SS5.p6.1 "3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.4](#S4.SS4.p6.1 "4.4 Ablation Studies ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[10\] S. He, H. Ding, and W. Jiang (2023) Semantic-promoted debiasing and background disambiguation for zero-shot instance segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 19498–19507. Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[11\] A. Hornung, K. M. Wurm, M. Bennewitz, C. Stachniss, and W. Burgard (2013) OctoMap: an efficient probabilistic 3d mapping framework based on octrees. Autonomous robots 34, pp. 189–206. Cited by: [§3.5](#S3.SS5.p3.1 "3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[12\] C. Huang, O. Mees, A. Zeng, and W. Burgard (2023) Visual language maps for robot navigation. In IEEE International Conference on Robotics and Automation, ICRA 2023, London, UK, May 29 - June 2, 2023, pp. 10608–10615. Cited by: [§2](#S2.p2.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[13\] Z. Huang, X. Wu, X. Chen, H. Zhao, L. Zhu, and J. Lasenby (2023) OpenIns3D: snap and lookup for 3d open-vocabulary instance segmentation. CoRR abs/2309.00616. Cited by: [§2](#S2.p2.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[14\] D. Huynh, J. Kuen, Z. Lin, J. Gu, and E. Elhamifar (2022) Open-vocabulary instance segmentation via robust cross-modal pseudo-labeling. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pp. 7010–7021. Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[15\] K. M. Jatavallabhula, A. Kuwajerwala, Q. Gu, M. Omama, T. Chen, S. Li, G. Iyer, S. Saryazdi, N. Keetha, A. Tewari, J. B. Tenenbaum, C. M. de Melo, M. Krishna, L. Paull, F. Shkurti, and A. Torralba (2023) ConceptFusion: open-set multimodal 3d mapping. Robotics: Science and Systems. Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§2](#S2.p2.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [Table 1](#S3.T1.3.3.1.1 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.2](#S4.SS2.p1.1 "4.2 Dataset and Evaluation Metrics ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[16\] L. Jiang, S. Shi, and B. Schiele (2024) Open-vocabulary 3d semantic segmentation with foundation models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 21284–21294. Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[17\] J. Kerr, C. M. Kim, K. Goldberg, A. Kanazawa, and M. Tancik (2023) LERF: language embedded radiance fields. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pp. 19672–19682. Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[18\] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W. Lo, P. Dollár, and R. Girshick (2023) Segment anything. arXiv:2304.02643. Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§2](#S2.p1.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[19\] X. Kong, S. Liu, M. Taher, and A. J. Davison (2023) VMAP: vectorised object mapping for neural field SLAM. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 952–961. Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[20\] B. Li, K. Q. Weinberger, S. J. Belongie, V. Koltun, and R. Ranftl (2022) Language-driven semantic segmentation. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[21\] K. Li, D. DeTone, S. Chen, M. Vo, I. Reid, H. Rezatofighi, C. Sweeney, J. Straub, and R. A. Newcombe (2021) ODAM: object detection, association, and mapping using posed RGB video. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pp. 5978–5988. Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[22\] F. Liang, B. Wu, X. Dai, K. Li, Y. Zhao, H. Zhang, P. Zhang, P. Vajda, and D. Marculescu (2023) Open-vocabulary semantic segmentation with mask-adapted CLIP. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 7061–7070. Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.1](#S4.SS1.p1.1 "4.1 Implementation Details ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[23\] S. Linok, T. Zemskova, S. Ladanova, R. Titkov, and D. Yudin (2024) Beyond bare queries: open-vocabulary object retrieval with 3d scene graph. arXiv preprint arXiv:2406.07113. Cited by: [Table 3](#S3.T3.3.2.2 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [Table 3](#S3.T3.3.4.1 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [Table 3](#S3.T3.3.5.1 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [Table 3](#S3.T3.3.5.2 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.2](#S4.SS2.p1.1 "4.2 Dataset and Evaluation Metrics ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.2](#S4.SS2.p2.1 "4.2 Dataset and Evaluation Metrics ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.3](#S4.SS3.p3.1 "4.3 Quantitative Comparison ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[24\] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, C. Li, J. Yang, H. Su, J. Zhu, et al. (2023) Grounding dino: marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499. Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[25\] S. Lu, H. Chang, E. P. Jing, A. Boularias, and K. Bekris (2023) OVIR-3d: open-vocabulary 3d instance retrieval without training on 3d data. In 7th Annual Conference on Robot Learning, Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§2](#S2.p2.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§3.3](#S3.SS3.p1.1 "3.3 Chronological Group-wise Segment Merging ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§3.4](#S3.SS4.p1.1 "3.4 Instance Feature Aggregation ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [Table 2](#S3.T2.3.10.1 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[26\] Y. Lu, X. Zhu, T. Wang, and Y. Ma (2024) OctreeOcc: efficient and multi-granularity occupancy prediction using octree queries. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, Cited by: [§2](#S2.p2.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[27\] D. Maggio, Y. Chang, N. Hughes, M. Trang, J. D. Griffith, C. Dougherty, E. Cristofalo, L. Schmid, and L. Carlone (2024) Clio: real-time task-driven open-set 3d scene graphs. IEEE Robotics Autom. Lett. 9 (10), pp. 8921–8928. Cited by: [§2](#S2.p2.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[28\] M. Minderer, A. A. Gritsenko, A. Stone, M. Neumann, D. Weissenborn, A. Dosovitskiy, A. Mahendran, A. Arnab, M. Dehghani, Z. Shen, X. Wang, X. Zhai, T. Kipf, and N. Houlsby (2022) Simple open-vocabulary object detection. In Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part X, Vol. 13670, pp. 728–755. Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[29\] P. D. A. Nguyen, T. D. Ngo, E. Kalogerakis, C. Gan, A. Tran, C. Pham, and K. Nguyen (2024) Open3DIS: open-vocabulary 3d instance segmentation with 2d mask guidance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: [§2](#S2.p2.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [Table 2](#S3.T2.3.5.1 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [Table 2](#S3.T2.3.6.1 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.3](#S4.SS3.p2.1 "4.3 Quantitative Comparison ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[30\] T. Pan, L. Tang, X. Wang, and S. Shan (2023) Tokenize anything via prompting. arXiv preprint arXiv:2312.09128. Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.1](#S4.SS1.p1.1 "4.1 Implementation Details ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[31\] S. Peng, K. Genova, C. M. Jiang, A. Tagliasacchi, M. Pollefeys, and T. A. Funkhouser (2023) OpenScene: 3d scene understanding with open vocabularies. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 815–824. Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§2](#S2.p2.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[32\] L. Qi, J. Kuen, T. Shen, J. Gu, W. Li, W. Guo, J. Jia, Z. Lin, and M. Yang (2023) High quality entity segmentation. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pp. 4024–4033. Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§3.2](#S3.SS2.p1.1 "3.2 Segment Proposal and Comprehension ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.1](#S4.SS1.p1.1 "4.1 Implementation Details ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[33\] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever (2021) Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML, Vol. 139, pp. 8748–8763. Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§2](#S2.p1.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [Table 3](#S3.T3.3.3.2 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [Table 3](#S3.T3.3.4.2 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.1](#S4.SS1.p1.1 "4.1 Implementation Details ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[34\] D. Robert, B. Vallet, and L. Landrieu Learning multi-view aggregation in the wild for large-scale 3d semantic segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pp. 5565–5574. Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[35\] D. Rozenberszki, O. Litany, and A. Dai (2022) Language-grounded indoor 3d semantic segmentation in the wild. In Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XXXIII, Vol. 13693, pp. 125–141. Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[36\] D. Rozenberszki, O. Litany, and A. Dai (2022) Language-grounded indoor 3d semantic segmentation in the wild. In European Conference on Computer Vision, pp. 125–141. Cited by: [§4.2](#S4.SS2.p1.1 "4.2 Dataset and Evaluation Metrics ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[37\] E. Schubert, J. Sander, M. Ester, H. Kriegel, and X. Xu (2017) DBSCAN revisited, revisited: why and how you should (still) use DBSCAN. ACM Trans. Database Syst. 42 (3), pp. 19:1–19:21. Cited by: [§3.2](#S3.SS2.p1.1 "3.2 Segment Proposal and Comprehension ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[38\] J. Schult, F. Engelmann, A. Hermans, O. Litany, S. Tang, and B. Leibe (2023) Mask3D: Mask Transformer for 3D Semantic Instance Segmentation. Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [Table 2](#S3.T2.3.3.1 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.4](#S4.SS4.p4.1 "4.4 Ablation Studies ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[39\] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma, et al. (2019) The replica dataset: a digital replica of indoor spaces. arXiv preprint arXiv:1906.05797. Cited by: [§4.2](#S4.SS2.p1.1 "4.2 Dataset and Evaluation Metrics ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[40\] A. Takmaz, E. Fedele, R. W. Sumner, M. Pollefeys, F. Tombari, and F. Engelmann (2023) OpenMask3D: Open-Vocabulary 3D Instance Segmentation. In Advances in Neural Information Processing Systems, Cited by: [§2](#S2.p2.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§3.4](#S3.SS4.p1.1 "3.4 Instance Feature Aggregation ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [Table 2](#S3.T2.3.7.1 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.3](#S4.SS3.p2.1 "4.3 Quantitative Comparison ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.4](#S4.SS4.p4.1 "4.4 Ablation Studies ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[41\] S. Thrun and A. Bücken (1996) Integrating grid-based and topological maps for mobile robot navigation. In Proceedings of the Thirteenth National Conference on Artificial Intelligence and Eighth Innovative Applications of Artificial Intelligence Conference, AAAI 96, IAAI 96, Portland, Oregon, USA, August 4-8, 1996, Volume 2, pp. 944–950. Cited by: [§4.3](#S4.SS3.p4.1 "4.3 Quantitative Comparison ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[42\] K. N. Tuan Duc Ngo (2023) ISBNet: a 3d point cloud instance segmentation network with instance-aware sampling and box-aware dynamic convolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[43\] V. VS, N. Yu, C. Xing, C. Qin, M. Gao, J. C. Niebles, V. M. Patel, and R. Xu (2023) Mask-free OVIS: open-vocabulary instance segmentation without manual mask annotations. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 23539–23549. Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[44\] A. Werby, C. Huang, M. Büchner, A. Valada, and W. Burgard (2024) Hierarchical open-vocabulary 3d scene graphs for language-grounded robot navigation. Robotics: Science and Systems. Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§2](#S2.p2.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§3.3](#S3.SS3.p1.1 "3.3 Chronological Group-wise Segment Merging ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§3.3](#S3.SS3.p3.1 "3.3 Chronological Group-wise Segment Merging ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§3.4](#S3.SS4.p1.1 "3.4 Instance Feature Aggregation ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [Table 1](#S3.T1.3.7.1.1 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [Table 4](#S3.T4.3.2.1 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.2](#S4.SS2.p1.1 "4.2 Dataset and Evaluation Metrics ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.2](#S4.SS2.p2.1 "4.2 Dataset and Evaluation Metrics ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.3](#S4.SS3.p1.1 "4.3 Quantitative Comparison ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.3](#S4.SS3.p4.1 "4.3 Quantitative Comparison ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.4](#S4.SS4.p4.1 "4.4 Ablation Studies ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[45\] J. Xu, S. D. Mello, S. Liu, W. Byeon, T. M. Breuel, J. Kautz, and X. Wang (2022) GroupViT: semantic segmentation emerges from text supervision. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pp. 18113–18123. Cited by: [§2](#S2.p1.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[46\] K. Yadav, R. Ramrakhya, S. K. Ramakrishnan, T. Gervet, J. M. Turner, A. Gokaslan, N. Maestre, A. X. Chang, D. Batra, M. Savva, A. W. Clegg, and D. S. Chaplot (2023) Habitat-matterport 3d semantics dataset. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 4927–4936. Cited by: [§4.2](#S4.SS2.p1.1 "4.2 Dataset and Evaluation Metrics ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.3](#S4.SS3.p4.1 "4.3 Quantitative Comparison ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.4](#S4.SS4.p6.1 "4.4 Ablation Studies ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[47\] K. Yamazaki, T. Hanyu, K. Vo, T. Pham, M. Tran, G. Doretto, A. Nguyen, and N. Le (2024) Open-fusion: real-time open-vocabulary 3d mapping and queryable scene representation. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 9411–9417. Cited by: [Table 3](#S3.T3.3.3.1 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[48\] M. Yan, J. Zhang, Y. Zhu, and H. Wang (2024) Maskclustering: view consensus based mask graph clustering for open-vocabulary 3d instance segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 28274–28284. Cited by: [§1](#S1.p2.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§2](#S2.p2.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§3.3](#S3.SS3.p1.1 "3.3 Chronological Group-wise Segment Merging ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [Table 2](#S3.T2.3.13.1 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.1](#S4.SS1.p1.1 "4.1 Implementation Details ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.3](#S4.SS3.p2.1 "4.3 Quantitative Comparison ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[49\] J. Yang, R. Ding, W. Deng, Z. Wang, and X. Qi (2024) RegionPLC: regional point-language contrastive learning for open-world 3d scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[50\] X. Yang, X. Gu, X. Yin, and X. Gao (2024) SA3DIP: segment any 3d instance with potential 3d priors. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, Cited by: [§2](#S2.p2.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[51\] Y. Yang, X. Wu, T. He, H. Zhao, and X. Liu (2023) Sam3d: segment anything in 3d scenes. arXiv preprint arXiv:2306.03908. Cited by: [§2](#S2.p2.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [Table 2](#S3.T2.3.11.1 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[52\] Z. Yang, J. Li, L. Yang, and H. Chen (2022) A smooth jump point search algorithm for mobile robots path planning based on a two-dimensional grid model.  J. Robotics 2022, pp. 7682201:1–7682201:15. Cited by: [§3.5](#S3.SS5.p6.1 "3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§4.4](#S4.SS4.p6.1 "4.4 Ablation Studies ‣ 4 Experiment ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[53\] Y. Yin, Y. Liu, Y. Xiao, D. Cohen-Or, J. Huang, and B. Chen (2024) Sai3d: segment any instance in 3d scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3292–3302. Cited by: [§2](#S2.p2.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [§3.3](#S3.SS3.p3.1 "3.3 Chronological Group-wise Segment Merging ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding"), [Table 2](#S3.T2.3.12.1 "In 3.5 Octree-Graph Construction and Applications ‣ 3 Method ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[54\] A. Yu, R. Li, M. Tancik, H. Li, R. Ng, and A. Kanazawa (2021) PlenOctrees for real-time rendering of neural radiance fields. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pp. 5732–5741. Cited by: [§2](#S2.p2.1 "2 Related Work ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
- \[55\] J. Zhang, L. Dai, F. Meng, Q. Fan, X. Chen, K. Xu, and H. Wang (2023) 3D-aware object goal navigation via simultaneous exploration and identification. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 6672–6682. Cited by: [§1](#S1.p1.1 "1 Introduction ‣ Open-Vocabulary Octree-Graph for 3D Scene Understanding").
