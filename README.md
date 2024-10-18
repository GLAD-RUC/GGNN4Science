# **A Survey of Geometric Graph Neural Networks: Data Structures, Models and Applications**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/RUC-GLAD/GGNN4Science/blob/main/LICENSE)  **[[arXiv]](https://arxiv.org/abs/2403.00485)**

<p align="center" width="100%">
    <img src="assets/ruc.svg" height="100"> &emsp;&emsp;&emsp;
    <img src="assets/thu.png" height="100"> &emsp;&emsp;&emsp;
    <img src="assets/tencent.png" width="160" height="100"/>
    <img src="assets/stanford.jpg" width="200" height="100"/>
</p>

## **Abstract**
Geometric graph is a special kind of graph with geometric features, which is vital to model many scientific problems. Unlike generic graphs, geometric graphs often exhibit physical symmetries of translations, rotations, and reflections, making them ineffectively processed by current Graph Neural Networks (GNNs). To tackle this issue, researchers proposed a variety of Geometric Graph Neural Networks equipped with invariant/equivariant properties to better characterize the geometry and topology of geometric graphs. Given the current progress in this field, it is imperative to conduct a comprehensive survey of data structures, models, and applications related to geometric GNNs. In this paper, based on the necessary but concise mathematical preliminaries, we provide a unified view of existing models from the geometric message passing perspective. Additionally, we summarize the applications as well as the related datasets to facilitate later research for methodology development and experimental evaluation. We also discuss the challenges and future potential directions of Geometric GNNs at the end of this survey. 

## **Table of Contents**
- [**A Survey of Geometric Graph Neural Networks: Data Structures, Models and Applications**](#a-survey-of-geometric-graph-neural-networks-data-structures-models-and-applications)
  - [**Abstract**](#abstract)
  - [**Architectures and Models**](#architectures-and-models)
    - [**Invariant Graph Neural Networks**](#invariant-graph-neural-networks)
    - [**Equivariant Graph Neural Networks**](#equivariant-graph-neural-networks)
      - [**Scalarization-Based Models**](#scalarization-based-models)
      - [**High-Degree Steerable Models**](#high-degree-steerable-models)
    - [**Geometric Graph Transformers**](#geometric-graph-transformers)
  - [**Geometric GNNs for Physics**](#geometric-gnns-for-physics)
    - [**Particle**](#particle)
      - [**1. N-Body Simulation**](#1-n-body-simulation)
      - [**2. Scene Simulation**](#2-scene-simulation)
  - [**Geometric GNNs for Biochemistry**](#geometric-gnns-for-biochemistry)
    - [**Small Molecule**](#small-molecule)
      - [**1. Molecule Property Prediction**](#1-molecule-property-prediction)
      - [**2. Molecular Dynamics**](#2-molecular-dynamics)
      - [**3. Molecule Generation**](#3-molecule-generation)
      - [**4. Molecule Pretraining**](#4-molecule-pretraining)
    - [**Protein**](#protein)
      - [**1. Protein Property Prediction**](#1-protein-property-prediction)
      - [**2. Protein Generation**](#2-protein-generation)
        - [**2.1 Protein Inverse Folding**](#21-protein-inverse-folding)
        - [**2.2 Protein Folding**](#22-protein-folding)
        - [**2.3 Protein Structure and Sequence Co-Design**](#23-protein-structure-and-sequence-co-design)
      - [**3. Pretraining**](#3-pretraining)
    - [**Mol + Mol**](#mol--mol)
      - [**1. Linker Design**](#1-linker-design)
      - [**2. Chemical Reaction**](#2-chemical-reaction)
    - [**Mol + Protein**](#mol--protein)
      - [**1. Ligand Binding Affinity**](#1-ligand-binding-affinity)
      - [**2. Protein-Ligand Docking Pose Prediction**](#2-protein-ligand-docking-pose-prediction)
      - [**3. Pocket-Based Mol Sampling**](#3-pocket-based-mol-sampling)
    - [**Protein + Protein**](#protein--protein)
      - [**1. Protein Interface Prediction**](#1-protein-interface-prediction)
      - [**2. Binding Affinity Prediction**](#2-binding-affinity-prediction)
      - [**3. Protein-Protein Docking Pose Prediction**](#3-protein-protein-docking-pose-prediction)
      - [**4. Antibody Design**](#4-antibody-design)
      - [**5. Peptide Design**](#5-peptide-design)
  - [**Other Domains**](#other-domains)
    - [**Crystal Property Prediction**](#crystal-property-prediction)
    - [**Crystal Generation**](#crystal-generation)
    - [**RNA Structure Ranking**](#rna-structure-ranking)
  - [**Related Surveys and Tutorials**](#related-surveys-and-tutorials)



## **Architectures and Models**
### **Invariant Graph Neural Networks**
> + [NIPS'17] SchNet: [A continuous-filter convolutional neural network for modeling quantum interactions](https://papers.nips.cc/paper/2017/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf)
> + [ICLR'20] DimeNet: [Directional Message Passing for Molecular Graphs](https://openreview.net/pdf?id=B1eWbxStPH)
>   + [arXiv:2011.14115] DimeNet++: [Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules](https://arxiv.org/pdf/2011.14115.pdf)
>   + [ICPP'23] FastDimeNet++: [Training DimeNet++ in 22 minutes](https://dl.acm.org/doi/pdf/10.1145/3605573.3605577)
> + [ICML'20] LieConv: [Generalizing Convolutional Neural Networks for Equivariance to Lie Groups on Arbitrary Continuous Data](https://proceedings.mlr.press/v119/finzi20a/finzi20a.pdf)
> + [NeurIPS'21] GemNet: [Universal Directional Graph Neural Networks for Molecules](https://proceedings.neurips.cc/paper/2021/file/35cf8659cfcb13224cbd47863a34fc58-Paper.pdf)
> + [ICLR'22] SphereNet: [Spherical Message Passing for 3D Molecular Graphs](https://openreview.net/pdf?id=givsRXsOt9r)
> + [NeurIPS'22] ComENet: [Towards Complete and Efficient Message Passing for 3D Molecular Graphs](https://papers.nips.cc/paper_files/paper/2022/file/0418973e545b932939302cb605d06f43-Paper-Conference.pdf)
> + [AAAI'24] QMP: [A Plug-and-Play Quaternion Message-Passing Module for Molecular Conformation Representation](https://ojs.aaai.org/index.php/AAAI/article/download/29602/31016)


### **Equivariant Graph Neural Networks**
#### **Scalarization-Based Models**
> + [arXiv:1910.00753] Radial Field: [Equivariant Flows: sampling configurations for multi-body systems with symmetric energies](https://arxiv.org/abs/1910.00753)
> + [ICLR'21] GVP-GNN: [Learning from Protein Structure with Geometric Vector Perceptrons](https://openreview.net/pdf?id=1YLJDvSx6J4)
> + [ICML'21] EGNN: [E(n) Equivariant Graph Neural Networks](https://proceedings.mlr.press/v139/satorras21a/satorras21a.pdf)
> + [ICML'21] PaiNN: [Equivariant message passing for the prediction of tensorial properties and molecular spectra](https://proceedings.mlr.press/v139/schutt21a/schutt21a.pdf)
> + [NeurIPS'21] LoCS: [Roto-translated Local Coordinate Frames For Interacting Dynamical Systems](https://proceedings.neurips.cc/paper/2021/file/32b991e5d77ad140559ffb95522992d0-Paper.pdf)
>   + [NeurIPS'23] G-LoCS: [Latent Field Discovery In Interacting Dynamical Systems With Neural Fields](https://proceedings.neurips.cc/paper_files/paper/2023/file/6521bd47ebaa28228cd6c74cb85afb65-Paper-Conference.pdf)
> + [ICLR'22] GMN: [Equivariant Graph Mechanics Networks with Constraints](https://openreview.net/pdf?id=SHbhHHfePhP)
> + [ICLR'22] Frame-Averaging: [Frame Averaging for Invariant and Equivariant Network Design](https://openreview.net/pdf?id=zIUyj55nXR)
> + [ICML'22] ClofNet: [SE (3) Equivariant Graph Neural Networks with Complete Local Frames](https://proceedings.mlr.press/v162/du22e/du22e.pdf)
> + [NeurIPS'22] EGHN: [Equivariant Graph Hierarchy-Based Neural Networks](https://papers.nips.cc/paper_files/paper/2022/file/3bdeb28a531f7af94b56bcdf8ee88f17-Paper-Conference.pdf)
> + [NeurIPS'23] LEFTNet: [A new perspective on building efficient and expressive 3D equivariant graph neural networks](https://openreview.net/pdf?id=hWPNYWkYPN)
> + [ICML'24] FastEGNN: [Improving Equivariant Graph Neural Networks on Large Geometric Graphs via Virtual Nodes Learning](https://openreview.net/attachment?id=wWdkNkUY8k&name=pdf)
> + [NC'2401] ViSNet: [Enhancing geometric representations for molecules with equivariant vector-scalar interactive message passing
](https://www.nature.com/articles/s41467-023-43720-2)
> + [NeurIPS'24] Neural P$^3$M: [Neural P$^3$M: A Long-Range Interaction Modeling Enhancer for Geometric GNNs](https://arxiv.org/pdf/2409.17622)
> + [NeurIPS'24] HEGNN: [Are High-Degree Representations Really Unnecessary in Equivarinat Graph Neural Networks?](https://arxiv.org/pdf/2410.11443)



#### **High-Degree Steerable Models**
> + [arXiv:1802.08219] Tensor field networks: [Rotation- and translation-equivariant neural networks for 3D point clouds](https://arxiv.org/pdf/1802.08219.pdf)
> + [NeurIPS'19] Cormorant: [Covariant Molecular Neural Networks](https://papers.nips.cc/paper/2019/file/03573b32b2746e6e8ca98b9123f2249b-Paper.pdf)
> + [ICLR'22] SEGNN: [Geometric and Physical Quantities Improve E(3) Equivariant Message Passing](https://openreview.net/pdf?id=_xwr8gOBeV1)
> + [NC'2205] NequIP: [E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials](https://www.nature.com/articles/s41467-022-29939-5)
>   + [NC'2302] Allegro: [Learning local equivariant representations for large-scale atomistic dynamics](https://www.nature.com/articles/s41467-023-36329-y)
> + [NeurIPS'22] SCN: [Spherical Channels for Modeling Atomic Interactions](https://proceedings.neurips.cc/paper_files/paper/2022/file/3501bea1ac61fedbaaff2f88e5fa9447-Paper-Conference.pdf)
>   + [ICML'23] eSCN: [Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs](https://openreview.net/pdf?id=QIejMwU0r9)
> + [NeurIPS'22] MACE: [Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields](https://openreview.net/pdf?id=YPpSngE-ZU)


### **Geometric Graph Transformers**
> + [NeurIPS'20] SE(3)-Transformers: [3D Roto-Translation Equivariant Attention Networks](https://proceedings.neurips.cc/paper/2020/file/15231a7ce4ba789d13b722cc5c955834-Paper.pdf)
> + [NeurIPS'21] Graphormer: [Do Transformers Really Perform Bad for Graph Representation?](https://proceedings.neurips.cc/paper/2021/file/f1c1592588411002af340cbaedd6fc33-Paper.pdf)
> + [ICML'21] LieTransformer: [Equivariant self-attention for Lie Groups](https://proceedings.mlr.press/v139/hutchinson21a/hutchinson21a.pdf)
> + [ICLR'22] TorchMD-Net: [Equivariant Transformers for Neural Network based Molecular Potentials](https://openreview.net/pdf?id=zNHzqZ9wrRB)
> + [ICML'22] GVP-Transformer: [Learning inverse folding from millions of predicted structures](https://proceedings.mlr.press/v162/hsu22a/hsu22a.pdf)
> + [NeurIPS'22] Equiformer: [Equivariant Graph Attention Transformer for 3D Atomistic Graphs](https://openreview.net/pdf?id=_efamP7PSjg)
>   + [NeurIPS'23] EquiformerV2: [Improved Equivariant Transformer for Scaling to Higher-Degree Representations](https://openreview.net/pdf?id=3o4jU8fWVj)
> + [NeurIPS'22] So3krates: [So3krates: Equivariant attention for interactions on arbitrary length-scales in molecular systems
](https://proceedings.neurips.cc/paper_files/paper/2022/file/bcf4ca90a8d405201d29dd47d75ac896-Paper-Conference.pdf)
>   + [NC'2408] So3krates: [A Euclidean transformer for fast and stable machine learned force fields](https://www.nature.com/articles/s41467-024-50620-6)
> + [NeurIPS'23] Geoformer: [Geometric Transformer with Interatomic Positional Encoding](https://papers.nips.cc/paper_files/paper/2023/file/aee2f03ecb2b2c1ea55a43946b651cfd-Paper-Conference.pdf)
> + [arXiv:2402.12714v1] EPT: [Equivariant Pretrained Transformer for Unified Geometric Learning on Multi-Domain 3D Molecules](https://arxiv.org/pdf/2402.12714v1.pdf)

## **Geometric GNNs for Physics**
### **Particle**
#### **1. N-Body Simulation**
**Datasets:**
>+ $N$-Body——[NRI: Neural relational inference for interacting systems](https://arxiv.org/pdf/1802.04687.pdf)
>+ 3D $N$-Body——[EGNN: E(n) equivariant graph neural networks](https://arxiv.org/pdf/2102.09844.pdf)
>+ Constrained $N$-Body——[GMN: Equivariant graph mechanics networks with constraints](https://openreview.net/pdf?id=SHbhHHfePhP)
>+ Hierarchical $N$-Body——[EGHN: Equivariant graph hierarchy-based neural networks](https://openreview.net/pdf?id=ywxtmG1nU_6)


**Methods:**
> + [NRI: Neural Relational Inference for Interacting Systems](https://arxiv.org/pdf/1802.04687.pdf)
> + [IN: Interaction networks for learning about objects, relations and physics](https://arxiv.org/pdf/1612.00222.pdf)
> + [E-NFs: E(n) Equivariant Normalizing Flows](https://proceedings.neurips.cc/paper/2021/hash/21b5680d80f75a616096f2e791affac6-Abstract.html)
> + [EGNN: E(n) equivariant graph neural networks](https://arxiv.org/pdf/2102.09844.pdf)
> + [SEGNNs: Geometric and Physical Quantities improve E(3) Equivariant Message Passing](http://arxiv.org/abs/2110.02905)
> + [GMN: Equivariant Graph Mechanics Networks with Constraints](https://openreview.net/pdf?id=SHbhHHfePhP)
> + [EGHN: Equivariant Graph Hierarchy-Based Neural Networks](https://openreview.net/pdf?id=ywxtmG1nU_6)
> + [HOGN: Hamiltonian Graph Networks with ODE Integrators](https://arxiv.org/pdf/1909.12790.pdf)
> + [NCGNN: Newton-Cotes Graph Neural Networks: On the Time Evolution of Dynamic Systems](https://arxiv.org/pdf/2305.14642.pdf)


#### **2. Scene Simulation**
**Datasets:**
> + Physion——[SGNN: Learning Physical Dynamics with Subequivariant Graph Neural Networks](https://arxiv.org/pdf/2210.06876.pdf)
> + Kubric MOVi-A——[GNS*: Graph network simulators can learn discontinuous, rigid contact dynamics](https://proceedings.mlr.press/v205/allen23a/allen23a.pdf)
> + FluidFall & FluidShake & BoxBath & RiceGrip——[DPI-Net: Learning particle dynamics for manipulating rigid bodies, deformable objects, and fluids](http://arxiv.org/abs/1810.01566)
> + Water3D——[GNS: Learning to simulate complex physics with graph networks](https://proceedings.mlr.press/v119/sanchez-gonzalez20a/sanchez-gonzalez20a.pdf) 
> + MIT Pushing——[FIGNet: Learning rigid dynamics with face interaction graph networks](https://arxiv.org/pdf/2212.03574.pdf) 


**Methods:**
> + [SGNN: Learning Physical Dynamics with Subequivariant Graph Neural Networks](https://arxiv.org/pdf/2210.06876.pdf)
> + [GNS: Learning to simulate complex physics with graph networks](https://proceedings.mlr.press/v119/sanchez-gonzalez20a/sanchez-gonzalez20a.pdf)
> + [C-GNS: Constraint-based graph network simulator](https://proceedings.mlr.press/v162/rubanova22a/rubanova22a.pdf)
> + [GNS*: Graph network simulators can learn discontinuous, rigid contact dynamics](https://proceedings.mlr.press/v205/allen23a/allen23a.pdf)
> + [HGNS: Learning large-scale subsurface simulations with a hybrid graph network simulator](http://arxiv.org/abs/2206.07680)
> + [DPI-Net: Learning particle dynamics for manipulating rigid bodies, deformable objects, and fluids](http://arxiv.org/abs/1810.01566)
> + [HRN: Flexible Neural Representation for Physics Prediction](http://arxiv.org/abs/1806.08047)
> + [FIGNet: Learning rigid dynamics with face interaction graph networks](https://arxiv.org/pdf/2212.03574.pdf)
> + [EGHN: Equivariant Graph Hierarchy-Based Neural Networks](https://openreview.net/pdf?id=ywxtmG1nU_6)
> + [LoCS: Roto-translated Local Coordinate Frames For Interacting Dynamical Systems](https://proceedings.neurips.cc/paper/2021/file/32b991e5d77ad140559ffb95522992d0-Paper.pdf)
> + [EqMotion: Equivariant Multi-agent Motion Prediction with Invariant Interaction Reasoning](https://arxiv.org/pdf/2303.10876.pdf)
> + [ESTAG: Equivariant Spatio-Temporal Attentive Graph Networks to Simulate Physical Dynamics](https://openreview.net/pdf?id=35nFSbEBks)
> + [SEGNO: Improving Generalization in Equivariant Graph Neural Networks with Physical Inductive Biases](https://openreview.net/pdf?id=3oTPsORaDH)
___

## **Geometric GNNs for Biochemistry**
### **Small Molecule**
#### **1. Molecule Property Prediction**
**Datasets:**
> + QM9——[ATOM3D: Tasks On Molecules in Three Dimensions](https://arxiv.org/pdf/2012.04035.pdf)
> + MD17——[Schnet--a deep learning architecture for molecules and materials](https://arxiv.org/pdf/1712.06113.pdf)
> + OCP——[eSCN: Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs](https://arxiv.org/pdf/2302.03655.pdf)


**Methods:**
> + [Cormorant: Covariant Molecular Neural Networks](https://arxiv.org/pdf/1906.04015.pdf)
> + [TFN: Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds](https://arxiv.org/pdf/1802.08219.pdf)
> + [SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks](https://arxiv.org/pdf/2006.10503.pdf)
> + [NequIP：E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials](https://www.nature.com/articles/s41467-022-29939-5#Sec14)
> + [SEGNNs：Geometric and Physical Quantities improve E(3) Equivariant Message Passing](https://arxiv.org/pdf/2110.02905.pdf)
> + [LieConv: Generalizing convolutional neural networks for equivariance to lie groups on arbitrary continuous data](https://arxiv.org/pdf/2002.12880.pdf)
> + [Lietransformer: equivariant self-attention for lie groups](https://arxiv.org/pdf/2012.10885.pdf)
> + [Schnet--a deep learning architecture for molecules and materials](https://arxiv.org/pdf/1712.06113.pdf)
> + [DimeNet: Directional Message Passing for Molecular Graphs](https://arxiv.org/pdf/2003.03123.pdf)
> + [GemNet: Universal Directional Graph Neural Networks for Molecules](https://arxiv.org/pdf/2106.08903.pdf)
> + [PaiNN: Equivariant message passing for the prediction of tensorial properties and molecular spectra](https://arxiv.org/pdf/2102.03150.pdf)
> + [TorchMD-NET: Equivariant Transformers for Neural Network based Molecular Potentials](https://arxiv.org/pdf/2202.02541.pdf)
> + [Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs](https://arxiv.org/pdf/2206.11990.pdf)
> + [SphereNet: Learning Spherical Representations for Detection and Classification in Omnidirectional Images](https://openaccess.thecvf.com/content_ECCV_2018/papers/Benjamin_Coors_SphereNet_Learning_Spherical_ECCV_2018_paper.pdf)
> + [EGNN: E(n) equivariant graph neural networks](https://arxiv.org/pdf/2102.09844.pdf)
> + [Graphormer: Do Transformers Really Perform Bad for Graph Representation?](https://arxiv.org/pdf/2106.05234.pdf)
> + [SCN: Spherical channels for modeling atomic interactions](https://proceedings.neurips.cc/paper_files/paper/2022/file/3501bea1ac61fedbaaff2f88e5fa9447-Paper-Conference.pdf)
> + [eSCN: Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs](https://arxiv.org/pdf/2302.03655.pdf)


#### **2. Molecular Dynamics**
**Datasets:**
> + MD17——[GMN: Equivariant Graph Mechanics Networks with Constraints](https://openreview.net/pdf?id=SHbhHHfePhP)
> + OCP——[GemNet: Universal Directional Graph Neural Networks for Molecules](https://arxiv.org/pdf/2106.08903.pdf)
> + Adk——[EGHN: Equivariant Graph Hierarchy-Based Neural Networks](https://openreview.net/pdf?id=ywxtmG1nU_6)
> + DW-4 & LJ-13——[E-CNF: Equivariant Flows: Exact Likelihood Generative Learning for Symmetric Densities](https://arxiv.org/pdf/2006.02425.pdf)
> + Fast-folding proteins——[ITO: Implicit Transfer Operator Learning: Multiple Time-Resolution Models for Molecular Dynamics](https://arxiv.org/pdf/2305.18046.pdf)


**Methods:**
> + [EGNN: E(n) equivariant graph neural networks](https://arxiv.org/pdf/2102.09844.pdf)
> + [NequIP：E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials](https://www.nature.com/articles/s41467-022-29939-5#Sec14)
> + [GMN: Equivariant Graph Mechanics Networks with Constraints](https://openreview.net/pdf?id=SHbhHHfePhP)
> + [EGHN: Equivariant Graph Hierarchy-Based Neural Networks](https://openreview.net/pdf?id=ywxtmG1nU_6)
> + [NCGNN: Newton-Cotes Graph Neural Networks: On the Time Evolution of Dynamic Systems](https://arxiv.org/pdf/2305.14642.pdf)
> + [ESTAG: Equivariant Spatio-Temporal Attentive Graph Networks to Simulate Physical Dynamics](https://openreview.net/pdf?id=35nFSbEBks)
> + [SEGNO: Improving Generalization in Equivariant Graph Neural Networks with Physical Inductive Biases](https://openreview.net/pdf?id=3oTPsORaDH)
> + [ITO: Implicit Transfer Operator Learning: Multiple Time-Resolution Models for Molecular Dynamics](https://arxiv.org/pdf/2305.18046.pdf)
> + [E-CNF: Equivariant Flows: Exact Likelihood Generative Learning for Symmetric Densities](https://arxiv.org/pdf/2006.02425.pdf)
> + [E-ACF: SE (3) equivariant augmented coupling flows](https://arxiv.org/pdf/2308.10364.pdf)


#### **3. Molecule Generation**
**Datasets:**
> + GEOM——[CGCF: Learning Neural Generative Dynamics for Molecular Conformation Generation](https://arxiv.org/pdf/2102.10240.pdf)
> + QM9——[CGCF: Learning Neural Generative Dynamics for Molecular Conformation Generation](https://arxiv.org/pdf/2102.10240.pdf)


**Methods:**
> + [GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation](https://arxiv.org/pdf/2203.02923.pdf)
> + [GeoLDM: Geometric Latent Diffusion Models for 3D Molecule Generation](https://arxiv.org/pdf/2305.01140.pdf)
> + [ConfVAE: An End-to-End Framework for Molecular Conformation Generation via Bilevel Programming](https://arxiv.org/pdf/2105.07246.pdf)
> + [ConfGF: Learning Gradient Fields for Molecular Conformation Generation](https://arxiv.org/pdf/2105.03902.pdf)
> + [G-SchNet: Symmetry-adapted generation of 3d point sets for the targeted discovery of molecules](https://proceedings.neurips.cc/paper/2019/file/a4d8e2a7e0d0c102339f97716d2fdfb6-Paper.pdf)
> + [cG-SchNet: Inverse design of 3d molecular structures with conditional generative neural networks](https://arxiv.org/pdf/2109.04824.pdf)
> + [DGSM: Predicting Molecular Conformation via Dynamic Graph Score Matching](https://proceedings.neurips.cc/paper_files/paper/2021/file/a45a1d12ee0fb7f1f872ab91da18f899-Paper.pdf)
> + [E-NFs: E(n) Equivariant Normalizing Flows](https://arxiv.org/pdf/2105.09016.pdf)
> + [EDM: Equivariant diffusion for molecule generation in 3d](https://proceedings.mlr.press/v162/hoogeboom22a/hoogeboom22a.pdf)
> + [GeoMol: Torsional Geometric Generation of Molecular 3D Conformer Ensembles](https://arxiv.org/pdf/2106.07802.pdf)
> + [Torsional Diffusion: Torsional Diffusion for Molecular Conformer Generation](https://arxiv.org/pdf/2206.01729.pdf)
> + [EEGSDE: Equivariant Energy-Guided {SDE} for Inverse Molecular Design](https://arxiv.org/pdf/2209.15408.pdf)
> + [DMCG: Direct molecular conformation generation](https://openreview.net/pdf?id=kcrIligNnl)
> + [MDM: Molecular Diffusion Model for 3D Molecule Generation](https://arxiv.org/pdf/2209.05710.pdf)
> + [MolDiff: Addressing the Atom-Bond Inconsistency Problem in 3D Molecule Diffusion Generation](https://arxiv.org/pdf/2305.07508.pdf)
> + [EquiFM: Equivariant Flow Matching with Hybrid Probability Transport for 3D Molecule Generation](https://openreview.net/forum?id=hHUZ5V9XFu)
> + [Hierdiff: Coarse-to-Fine: a Hierarchical Diffusion Model for Molecule Generation in 3D](https://proceedings.mlr.press/v202/qiang23a.html)
> + [MPerformer: An SE (3) Transformer-based Molecular Perceptron](https://dl.acm.org/doi/10.1145/3583780.3614974)


#### **4. Molecule Pretraining**
**Datasets:**
> + QM9——[3D-Infomax: 3d infomax improves gnns for molecular property prediction](https://proceedings.mlr.press/v162/stark22a/stark22a.pdf)
> + GEOM & QMugs——[3D-Infomax: 3d infomax improves gnns for molecular property prediction](https://proceedings.mlr.press/v162/stark22a/stark22a.pdf)
> + PCQM4Mv2——[3D-PGT: Automated 3D pre-training for molecular property prediction](https://arxiv.org/pdf/2306.07812.pdf)
> + Uni-Mol——[Uni-Mol: A Universal 3D Molecular Representation Learning Framework](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/6402990d37e01856dc1d1581/original/uni-mol-a-universal-3d-molecular-representation-learning-framework.pdf)

**Methods:**
> + [3D-EMGP: Energy-Motivated Equivariant Pretraining for 3D Molecular Graphs](https://arxiv.org/pdf/2207.08824.pdf)
> + [GeoSSL-DDM: Molecular Geometry Pretraining with {SE}(3)-Invariant Denoising Distance Matching](https://arxiv.org/pdf/2206.13602.pdf)
> + [GraphMVP: Pre-training Molecular Graph Representation with 3D Geometry](https://arxiv.org/pdf/2110.07728.pdf)
> + [GNS-TAT: Pre-training via Denoising for Molecular Property Prediction](https://arxiv.org/pdf/2206.00133.pdf)
> + [3D-Infomax: 3d infomax improves gnns for molecular property prediction](https://proceedings.mlr.press/v162/stark22a/stark22a.pdf)
> + [Uni-Mol: A Universal 3D Molecular Representation Learning Framework](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/6402990d37e01856dc1d1581/original/uni-mol-a-universal-3d-molecular-representation-learning-framework.pdf)
> + [Transformer-M: One transformer can understand both 2d & 3d molecular data](https://arxiv.org/pdf/2210.01765.pdf)
> + [SliDe: Sliced Denoising: A Physics-Informed Molecular Pre-Training Method](https://arxiv.org/abs/2311.02124)
> + [Frad: Fractional Denoising for 3D Molecular Pre-training](https://proceedings.mlr.press/v202/feng23c.html)
> + [MGMAE: Molecular Representation Learning by Reconstructing Heterogeneous Graphs with A High Mask Ratio](https://dl.acm.org/doi/abs/10.1145/3511808.3557395)
> + [MoleculeSDE: A Group Symmetric Stochastic Differential Equation Model for Molecule Multi-modal Pretraining](https://proceedings.mlr.press/v202/liu23h/liu23h.pdf)
___

### **Protein**
#### **1. Protein Property Prediction**
**Datasets:**
> + GENE Ontology——[GearNet : Protein representation learning by geometric structure pretraining](http://arxiv.org/abs/2203.06125)
> + ENZYME——[GearNet : Protein representation learning by geometric structure pretraining](http://arxiv.org/abs/2203.06125)
> + SCOPe——[TAPE: Evaluating protein transfer learning with TAPE](https://arxiv.org/pdf/1906.08230.pdf)
> + UniProt——[DeepLoc: prediction of protein subcellular localization using deep learning](https://academic.oup.com/bioinformatics/article/33/21/3387/3931857)
> + PDB——[ATOM3D: Tasks On Molecules in Three Dimensions](https://arxiv.org/pdf/2012.04035.pdf)


**Methods:**
> + [DeepFRI: Structure-based protein function prediction using graph convolutional network](https://www.nature.com/articles/s41467-021-23303-9)
> + [LM-GVP: an extensible sequence and structure informed deep learning framework for protein property prediction](https://www.nature.com/articles/s41598-022-10775-y)
> + [GearNet : Protein representation learning by geometric structure pretraining](http://arxiv.org/abs/2203.06125)
> + [3DCNN: 3D deep convolutional neural networks for amino acid environment similarity analysis](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1702-0)
> + [TM-align: a protein structure alignment algorithm based on the TM-score](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1084323/)
> + [GVP: Learning from Protein Structure with Geometric Vector Perceptrons](http://arxiv.org/abs/2009.01411)
> + [PAUL: Hierarchical rotation-equivariant neural networks to select structural models of protein complex](https://onlinelibrary.wiley.com/doi/10.1002/prot.26033)
> + [EDN: Protein model quality assessment using rotation-equivariant transformations on point clouds](https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.26494)
> + [EnQA: 3D-equivariant graph neural networks for protein model quality assessment](https://www.biorxiv.org/content/10.1101/2022.04.12.488060v1.full.pdf)
> + [ScanNet: an interpretable geometric deep learning model for structure-based protein binding site prediction](https://www.nature.com/articles/s41592-022-01490-7)
> + [PocketMiner: Predicting locations of cryptic pockets from single protein structures using the PocketMiner graph neural network](https://www.nature.com/articles/s41467-023-36699-3.pdf)
> + [EquiPocket: an E(3)-Equivariant Geometric Graph Neural Network for Ligand Binding Site Prediction](https://arxiv.org/pdf/2302.12177.pdf)


####  **2. Protein Generation** 
**Datasets:**
> + CATH——[GVP: Learning from Protein Structure with Geometric Vector Perceptrons](http://arxiv.org/abs/2009.01411)
> + SCOPe——[ProstT5: Bilingual Language Model for Protein Sequence and Structure](https://www.biorxiv.org/content/10.1101/2023.07.23.550085v1.full.pdf)
> + AlphaFoldDB & PDB & ESM Atlas——[ESMFold: Evolutionary-scale prediction of atomic-level protein structure with a language model](https://www.science.org/doi/abs/10.1126/science.ade2574)
> + CASP——[ATOM3D: Tasks On Molecules in Three Dimensions](https://arxiv.org/pdf/2012.04035.pdf)



#####  **2.1 Protein Inverse Folding** 
**Methods:**
> + [GVP: Learning from Protein Structure with Geometric Vector Perceptrons](http://arxiv.org/abs/2009.01411)
> + [Generative models for graph-based protein design](https://proceedings.neurips.cc/paper/2019/hash/f3a4ff4839c56a5f460c88cce3666a2b-Abstract.html)
> + [ESM-IF1: Learning inverse folding from millions of predicted structures](https://proceedings.mlr.press/v162/hsu22a.html)
> + [GCA: Generative de novo protein design with global context](https://arxiv.org/pdf/2204.10673.pdf)
> + [ProteinMPNN: Robust deep learning based protein sequence design using ProteinMPNN](https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1)
> + [PiFold: Toward effective and efficient protein inverse folding](https://arxiv.org/pdf/2209.12643.pdf)
> + [LM-Design: Structure-informed Language Models Are Protein Designers](https://arxiv.org/abs/2302.01649)


#####  **2.2 Protein Folding** 

**Methods:**
> + [AlphaFold: Improved protein structure prediction using potentials from deep learning](https://discovery.ucl.ac.uk/id/eprint/10089234/1/343019_3_art_0_py4t4l_convrt.pdf)
> + [AlphaFold2:  Highly accurate protein structure prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2))
> + [RoseTTAFold: Accurate prediction of protein structures and interactions using a three-track neural network](https://www.science.org/doi/10.1126/science.abj8754)
> + [RoseTTAFold2: Efficient and accurate prediction of protein structure using RoseTTAFold2](https://www.biorxiv.org/content/10.1101/2023.05.24.542179v1.full.pdf)
> + [RFAA: Generalized Biomolecular Modeling and Design with RoseTTAFold All-Atom](https://www.biorxiv.org/content/10.1101/2023.10.09.561603v1)
> + [RFdiffusion: De novo design of protein structure and function with RFdiffusion](https://www.nature.com/articles/s41586-023-06415-8)
> + [EIGENFOLD: GENERATIVE PROTEIN STRUCTURE PREDICTION WITH DIFFUSION MODELS](https://arxiv.org/pdf/2304.02198.pdf)
> + [Chroma: Illuminating protein space with a programmable generative model](https://www.nature.com/articles/s41586-023-06728-8)
> + [ESMFold: Evolutionary-scale prediction of atomic-level protein structure with a language model](https://www.science.org/doi/abs/10.1126/science.ade2574)
> + [HelixFold-Single: MSA-free Protein Structure Prediction by Using Protein Language Model as an Alternative](https://arxiv.org/pdf/2207.13921.pdf)


#####  **2.3 Protein Structure and Sequence Co-Design** 
**Methods:**
> + [Chroma: Illuminating protein space with a programmable generative model](https://www.nature.com/articles/s41586-023-06728-8)
> + [RFdiffusion: De novo design of protein structure and function with RFdiffusion](https://www.nature.com/articles/s41586-023-06415-8)
> + [PROTSEED: Protein Sequence and Structure Co-Design with Equivariant Translation](https://arxiv.org/pdf/2210.08761.pdf)



####  **3. Pretraining**
**Datasets:**
>+ CATH——[S2F: Multimodal pre-training model for sequence-based prediction of protein-protein interaction](https://proceedings.mlr.press/v165/xue22a/xue22a.pdf)
>+ SCOPe——[ProSE: Learning the protein language: Evolution, structure, and function](https://pubmed.ncbi.nlm.nih.gov/34139171/)
>+ AlphaFoldDB——[GearNet : Protein representation learning by geometric structure pretraining](http://arxiv.org/abs/2203.06125)
>+ UniProt & BFD——[Prottrans: Toward understanding the language of life through self-supervised learning](https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf)
>+ NetSurfP-2.0——[Peer: a comprehensive and multi-task benchmark for protein sequence understanding](https://proceedings.neurips.cc/paper_files/paper/2022/file/e467582d42d9c13fa9603df16f31de6d-Paper-Datasets_and_Benchmarks.pdf)


**Methods:**
> + [ProtTrans: Toward understanding the language of life through self-supervised learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9477085)
> + [ProtGPT2: ProtGPT2 is a deep unsupervised language model for protein design](https://www.nature.com/articles/s41467-022-32007-7)
> + [PromptProtein: Multi-level Protein Structure Pre-training via Prompt Learning](https://openreview.net/pdf?id=XGagtiJ8XC)
> + [GearNet: Protein representation learning by geometric structure pretraining](http://arxiv.org/abs/2203.06125)
>   [xTrimoPGLM: Unified 100B-Scale Pre-trained Transformer for Deciphering the Language of Protein](https://www.biorxiv.org/content/10.1101/2023.07.05.547496v1.full.pdf)
> + [ProFSA: Self-supervised Pocket Pretraining via Protein Fragment-Surroundings Alignment](https://arxiv.org/pdf/2310.07229.pdf)
> + [DrugCLIP: Contrastive Protein-Molecule Representation Learning for Virtual Screening](https://arxiv.org/abs/2310.06367)
> + [Self-supervised Pocket Pretraining via Protein Fragment-Surroundings Alignment](https://arxiv.org/abs/2310.07229)
> + [HJRSS: Toward More General Embeddings for Protein Design: Harnessing Joint Representations of Sequence and Structure](https://www.biorxiv.org/content/10.1101/2021.09.01.458592v1.full.pdf)
> + [ESM-1b: Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences](https://www.biorxiv.org/content/10.1101/622803v4.full.pdf)
> + [ESM2: Evolutionary-scale prediction of atomic-level protein structure with a language model](https://www.biorxiv.org/content/biorxiv/early/2022/10/31/2022.07.20.500902.full.pdf)
___



### Mol + Mol
#### **1. Linker Design**
**Datasets:**
> + GEOM——[DiffLinker: Equivariant 3d-conditional diffusion models for molecular linker design](https://arxiv.org/pdf/2210.05274.pdf)
> + ZINC——[3DLinker: An E (3) Equivariant Variational Autoencoder for Molecular Linker Design](https://proceedings.mlr.press/v162/huang22g/huang22g.pdf)
> + CASF——[DeLinker: Deep Generative Models for 3D Linker Design](https://pubs.acs.org/doi/10.1021/acs.jcim.9b01120)

**Methods:**
> + [DiffLinker: Equivariant 3d-conditional diffusion models for molecular linker design](https://arxiv.org/pdf/2210.05274.pdf)
> + [DeLinker: Deep Generative Models for 3D Linker Design](https://pubs.acs.org/doi/10.1021/acs.jcim.9b01120)
> + [3DLinker: An E (3) Equivariant Variational Autoencoder for Molecular Linker Design](https://proceedings.mlr.press/v162/huang22g/huang22g.pdf)
___

#### **2. Chemical Reaction**
**Datasets:**
> + S<sub>N</sub>2-TS——[TSNet: predicting transition state structures with tensor field networks and transfer learning](https://pubs.rsc.org/en/content/articlelanding/2021/sc/d1sc01206a)
> + Transition1x——[OA-Reaction: Accurate transition state generation with an object-aware equivariant elementary reaction diffusion model](https://arxiv.org/pdf/2304.06174.pdf)


**Methods:**
> + [OA-Reaction: Accurate transition state generation with an object-aware equivariant elementary reaction diffusion model](https://arxiv.org/pdf/2304.06174.pdf)
> + [TSNet: predicting transition state structures with tensor field networks and transfer learning](https://pubs.rsc.org/en/content/articlelanding/2021/sc/d1sc01206a)
___



### **Mol + Protein**
#### **1. Ligand Binding Affinity**
**Datasets:**
> + CrossDocked2020——[GNINA: Three-Dimensional Convolutional Neural Networks and a Cross-Docked Data Set for Structure-Based Drug Design](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00411)
> + PDBbind——[ATOM3D: Tasks On Molecules in Three Dimensions](https://arxiv.org/pdf/2012.04035.pdf)



**Methods:**
> + [TargetDiff: 3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction](https://arxiv.org/pdf/2303.03543.pdf)
> + [GET: GENERALIST EQUIVARIANT TRANSFORMER TOWARDS 3D MOLECULAR INTERACTION LEARNING](https://arxiv.org/pdf/2306.01474.pdf)
> + [MaSIF: Deciphering interaction fingerprints from protein molecular surfaces using geometric deep learning](https://www.nature.com/articles/s41592-019-0666-6)
> + [ProtNet: Learning protein representations via complete 3d graph networks](https://arxiv.org/pdf/2207.12600.pdf)
> + [HGIN: Geometric Graph Learning for Protein Mutation Effect Prediction](https://dl.acm.org/doi/abs/10.1145/3583780.3614893)
> + [BindNet: Protein-ligand binding representation learning from fine-grained interactions](https://arxiv.org/pdf/2311.16160.pdf)

#### **2. Protein-Ligand Docking Pose Prediction**
**Datasets:**
> + PDBbind——[EquiBind: Geometric Deep Learning for Drug Binding Structure Prediction](https://proceedings.mlr.press/v162/stark22b/stark22b.pdf)

**Methods:**
> + [EquiBind: Geometric Deep Learning for Drug Binding Structure Prediction](https://proceedings.mlr.press/v162/stark22b/stark22b.pdf)
> + [DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking](https://arxiv.org/pdf/2210.01776.pdf)
> + [TANKBind: Trigonometry-Aware Neural NetworKs for Drug-Protein Binding Structure Prediction](https://proceedings.neurips.cc/paper_files/paper/2022/file/2f89a23a19d1617e7fb16d4f7a049ce2-Paper-Conference.pdf)
> + [DESERT: Zero-shot 3d drug design by sketching and generating](https://nips.cc/media/neurips-2022/Slides/54457.pdf)
> + [FABind: Fast and Accurate Protein-Ligand Binding](https://arxiv.org/pdf/2310.06763.pdf)


#### **3. Pocket-Based Mol Sampling**
**Datasets:**
> + CrossDocked2020——[TargetDiff: 3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction](https://arxiv.org/pdf/2303.03543.pdf)

**Methods:**
> + [Pocket2Mol: Efficient Molecular Sampling Based on 3{D} Protein Pockets](https://proceedings.mlr.press/v162/peng22b/peng22b.pdf)
> + [TargetDiff: 3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction](https://arxiv.org/pdf/2303.03543.pdf)
> + [SBDD: A 3D Generative Model for Structure-Based Drug Design](https://proceedings.neurips.cc/paper/2021/hash/314450613369e0ee72d0da7f6fee773c-Abstract.html)
> + [FLAG: Molecule Generation For Target Protein Binding with Structural Motifs](https://openreview.net/pdf?id=Rq13idF0F73)
> + [Zero-Shot 3D Drug Design by Sketching and Generating](https://proceedings.neurips.cc/paper_files/paper/2022/file/96ddbf813f042e8ff891b4d6f7149bb6-Paper-Conference.pdf)
___



### **Protein + Protein**
#### **1. Protein Interface Prediction**
**Datasets:**
> + DIPS——[ATOM3D: Tasks On Molecules in Three Dimensions](https://arxiv.org/pdf/2012.04035.pdf)
> + DIPS-plus——[DeepInteract: Geometric Transformers for Protein Interface Contact Prediction](https://arxiv.org/pdf/2110.02423.pdf)
> + BioGRID——[SYNTERACT: Protein-Protein Interaction Prediction is Achievable with Large Language Models](https://www.biorxiv.org/content/10.1101/2023.06.07.544109v1.full.pdf)


**Methods:**
>+ [SASNet: End-to-end learning on 3d protein structure for interface prediction](https://proceedings.neurips.cc/paper_files/paper/2019/file/6c7de1f27f7de61a6daddfffbe05c058-Paper.pdf)
>+ [dMaSIF： Fast end-to-end learning on protein surfaces](https://openaccess.thecvf.com/content/CVPR2021/papers/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.pdf)
>+ [DeepInteract: Geometric Transformers for Protein Interface Contact Prediction](https://arxiv.org/pdf/2110.02423.pdf)

#### **2. Binding Affinity Prediction**
**Datasets:**
> + DB5.5——[GET: GENERALIST EQUIVARIANT TRANSFORMER TOWARDS 3D MOLECULAR INTERACTION LEARNING](https://arxiv.org/pdf/2306.01474.pdf)
> + PDBBind——[GeoPPI: Deep geometric representations for modeling effects of mutations on protein-protein binding affinity](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009284)
> + SKEMPI 2.0——[mmCSM-PPI: predicting the effects of multiple point mutations on protein–protein interactions](https://academic.oup.com/nar/article-pdf/49/W1/W417/38842375/gkab273.pdf)


**Methods:**
>+ [mmCSM-PPI: predicting the effects of multiple point mutations on protein–protein interactions](https://academic.oup.com/nar/article-pdf/49/W1/W417/38842375/gkab273.pdf)
>+ [GeoPPI: Deep geometric representations for modeling effects of mutations on protein-protein binding affinity](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009284)
>+ [GET: GENERALIST EQUIVARIANT TRANSFORMER TOWARDS 3D MOLECULAR INTERACTION LEARNING](https://arxiv.org/pdf/2306.01474.pdf)


#### **3. Protein-Protein Docking Pose Prediction**
**Datasets:**
> + DB5.5 & DIPS——[Equidock: Independent {SE}(3)-Equivariant Models for End-to-End Rigid Protein Docking](http://arxiv.org/abs/2111.07786)


**Methods:**
> + [Equidock: Independent {SE}(3)-Equivariant Models for End-to-End Rigid Protein Docking](http://arxiv.org/abs/2111.07786)
> + [HMR: Learning Harmonic Molecular Representations on Riemannian Manifold](https://arxiv.org/pdf/2303.15520.pdf)
> + [HSRN: Antibody-antigen docking and design via hierarchical structure refinement](https://proceedings.mlr.press/v162/jin22a/jin22a.pdf)
> + [DiffDock-PP: Rigid Protein-Protein Docking with Diffusion Models](https://arxiv.org/pdf/2304.03889.pdf)
> + [dMaSIF-extension: Physics-informed deep neural network for rigid-body protein docking](https://openreview.net/pdf?id=5yn5shS6wN)
> + [AlphaFold-Multimer: Protein complex prediction with AlphaFold-Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.full.pdf)
> + [SyNDock: N Rigid Protein Docking via Learnable Group Synchronization](https://arxiv.org/pdf/2305.15156.pdf)
> + [ElliDock: Rigid Protein-Protein Docking via Equivariant Elliptic-Paraboloid Interface Prediction](https://arxiv.org/pdf/2401.08986.pdf)
> + [EBMDock: Neural Probabilistic Protein-Protein Docking via a Differentiable Energy Model](https://openreview.net/pdf?id=qg2boc2AwU)


#### **4. Antibody Design**
**Datasets:**
> + SAbDab & RAbD & Cov-abdab——[RefineGNN: Iterative Refinement Graph Neural Network for Antibody Sequence-Structure Co-design](https://arxiv.org/pdf/2110.04624.pdf)
> + SKEMPI 2.0——[ATOM3D: Tasks On Molecules in Three Dimensions](https://arxiv.org/pdf/2012.04035.pdf)


**Methods:**
> + [DiffAb: Antigen-Specific Antibody Design and Optimization with Diffusion-Based Generative Models for Protein Structures](https://proceedings.neurips.cc/paper_files/paper/2022/file/3fa7d76a0dc1179f1e98d1bc62403756-Paper-Conference.pdf)
> + [MEAN: Conditional Antibody Design as 3D Equivariant Graph Translation](https://arxiv.org/pdf/2208.06073.pdf)
> + [dyMEAN: End-to-End Full-Atom Antibody Design](https://arxiv.org/pdf/2302.00203.pdf)
> + [RefineGNN: Iterative Refinement Graph Neural Network for Antibody Sequence-Structure Co-design](https://arxiv.org/pdf/2110.04624.pdf)
> + [PROTSEED: Protein Sequence and Structure Co-Design with Equivariant Translation](https://arxiv.org/pdf/2210.08761.pdf)
> + [ADesigner: Cross-Gate MLP with Protein Complex Invariant Embedding is A One-Shot Antibody Designer](https://arxiv.org/pdf/2305.09480.pdf)
> + [AbBERT: Incorporating Pre-training Paradigm for Antibody Sequence-Structure Co-design](https://arxiv.org/pdf/2211.08406.pdf)
> + [AbODE: Ab initio antibody design using conjoined ODEs](https://proceedings.mlr.press/v202/verma23a.html)
> + [AbDiffuser: Full-Atom Generation of In-Vitro Functioning Antibodies](https://arxiv.org/pdf/2308.05027.pdf)
> + [tFold: Fast and accurate modeling and design of antibody-antigen complex using tFold](https://www.biorxiv.org/content/10.1101/2024.02.05.578892v1.full.pdf)


#### **5. Peptide Design**
**Datasets:**
> + PepBDB——[CAMP: A deep-learning framework for multi-level peptide–protein interaction prediction](https://www.nature.com/articles/s41467-021-25772-4)
> + LNR——[PDAR: Harnessing protein folding neural networks for peptide–protein docking](https://www.nature.com/articles/s41467-021-27838-9)
> + PepGLAD——[PepGLAD: Full-Atom Peptide Design with Geometric Latent Diffusion](https://arxiv.org/pdf/2402.13555v1.pdf)


**Methods:**
> + [HelixGAN: a deep-learning methodology for conditional de novo design of α-helix structures](https://academic.oup.com/bioinformatics/article-pdf/39/1/btad036/48959131/btad036.pdf)
> + [RFdiffusion: De novo design of protein structure and function with RFdiffusion](https://www.nature.com/articles/s41586-023-06415-8)
> + [PepGLAD: Full-Atom Peptide Design with Geometric Latent Diffusion](https://arxiv.org/pdf/2402.13555v1.pdf)
___




## **Other Domains**

### **Crystal Property Prediction**

**Datasets:**
> + Materials Project——[CGCNN: Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://arxiv.org/pdf/1710.10324.pdf)
> + JARVIS-DFT——[JARVIS-ML: Machine learning with force-field-inspired descriptors for materials: Fast screening and mapping energy landscape](https://arxiv.org/pdf/1805.07325.pdf)

**Methods:**
> + [CGCNN: Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://arxiv.org/pdf/1710.10324.pdf)
> + [MEGNet: Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals](https://arxiv.org/pdf/1812.05055.pdf)
> + [ALIGNN: Atomistic Line Graph Neural Network for Improved Materials Property Predictions](https://arxiv.org/pdf/2106.01829.pdf)
> + [ECN: Equivariant Networks for Crystal Structures](https://arxiv.org/pdf/2211.15420.pdf)
> + [Matformer: Periodic Graph Transformers for Crystal Material Property Prediction](https://arxiv.org/pdf/2209.11807.pdf)
> + [Crystal twins: Self-supervised Learning for Crystalline Material Property Prediction](https://www.nature.com/articles/s41524-022-00921-5)
> + [MMPT: A Crystal-Specific Pre-Training Framework for Crystal Material Property Prediction](https://arxiv.org/pdf/2306.05344.pdf)
> + [CrysDiff: A Diffusion-Based Pre-training Framework for Crystal Property Prediction](https://ojs.aaai.org/index.php/AAAI/article/download/28748/29440)

### **Crystal Generation**

**Datasets:**
> + Perov-5 & Carbon-24 & MP-20——[CDVAE: Crystal Diffusion Variational Autoencoder for Periodic Material Generation](https://github.com/txie-93/cdvae/tree/main/data)

**Methods**:
> + [CDVAE: Crystal Diffusion Variational Autoencoder for Periodic Material Generation](https://github.com/txie-93/cdvae/tree/main/data)
> + [DiffCSP: Crystal Structure Prediction by Joint Equivariant Diffusion](https://arxiv.org/pdf/2309.04475.pdf)
> + [DiffCSP++: Space Group Constrained Crystal Generation](https://arxiv.org/pdf/2402.03992.pdf)
> + [SyMat: Towards Symmetry-Aware Generation of Periodic Materials](https://arxiv.org/pdf/2307.02707.pdf)
> + [MatterGen: A Generative Model for Inorganic Materials Design](https://arxiv.org/pdf/2312.03687.pdf)


### **RNA Structure Ranking**

**Datasets:**
> + FARFAR2-Puzzles——[ARES: Geometric deep learning of RNA structure](https://www.science.org/doi/10.1126/science.abe5650)

**Methods**:
> + [ARES: Geometric deep learning of RNA structure](https://www.science.org/doi/10.1126/science.abe5650)
> + [PaxNet: Physics-aware graph neural network for accurate RNA 3D structure prediction](https://arxiv.org/pdf/2210.16392.pdf)

## **Related Surveys and Tutorials**
> + [Geometrically equivariant graph neural networks: A survey](https://arxiv.org/pdf/2202.07230.pdf)
> + [A Hitchhiker's Guide to Geometric GNNs for 3D Atomic Systems](https://arxiv.org/pdf/2312.07511.pdf)
> + [A Systematic Survey of Chemical Pre-trained Models](https://www.ijcai.org/proceedings/2023/0760.pdf)
> + [Graph-based Molecular Representation Learning](https://www.ijcai.org/proceedings/2023/0744.pdf)
> + [Geometric Deep Learning on Molecular Representations](https://arxiv.org/pdf/2107.12375.pdf)
> + [Artificial intelligence for science in quantum, atomistic, and continuum systems](https://arxiv.org/pdf/2307.08423.pdf)

