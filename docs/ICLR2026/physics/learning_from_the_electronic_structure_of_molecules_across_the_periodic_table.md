---
title: >-
  [论文解读] Learning from the Electronic Structure of Molecules across the Periodic Table
description: >-
  [ICLR 2026][物理/科学计算][Hamiltonian prediction] 本文提出 HELM——首个能扩展到 100+ 原子、58 种元素、含弥散函数大基组的"通用"哈密顿量矩阵预测模型，并配套发布迄今最大的分子哈密顿量数据集 OMol CSH 58k，进而把哈密顿量预训练得到的共享表示迁移到能量预测，在低数据场景下实现最高约 2× 的能量预测精度提升。
tags:
  - "ICLR 2026"
  - "物理/科学计算"
  - "Hamiltonian prediction"
  - "MLIP"
  - "electronic structure"
  - "图神经网络"
  - "pretraining"
  - "DFT"
---

# Learning from the Electronic Structure of Molecules across the Periodic Table

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=PS1YS8Wv4t](https://openreview.net/forum?id=PS1YS8Wv4t)  
**代码**: 待确认  
**领域**: AI for Science / 量子化学 / 机器学习原子间势  
**关键词**: Hamiltonian prediction, MLIP, electronic structure, equivariant GNN, pretraining, DFT  

## 一句话总结
本文提出 HELM——首个能扩展到 100+ 原子、58 种元素、含弥散函数大基组的"通用"哈密顿量矩阵预测模型，并配套发布迄今最大的分子哈密顿量数据集 OMol CSH 58k，进而把哈密顿量预训练得到的共享表示迁移到能量预测，在低数据场景下实现最高约 2× 的能量预测精度提升。

## 研究背景与动机
**领域现状**：机器学习原子间势（MLIP）通过拟合 DFT 算出的力和能量来逼近 Born–Oppenheimer 势能面，其性能随训练数据量持续增长，已有模型（如 Meta 的 UMA）训练在 4.59 亿条能量标签上。但 DFT 计算每个 N 原子体系时，除了产生 1 个能量、O(N) 个力标签外，还会算出一个 O(N²) 规模的哈密顿量矩阵 H，里面编码了激发态、电离能、电子密度、多极矩等远比力/能量丰富的信息——而这些"免费"数据迄今几乎没被用于训练大规模原子性质模型。

**现有痛点**：一方面，最先进的 MLIP 仍受限于数据，但当前最大模型已经吃掉超过 100 亿核时的 DFT 数据，靠继续堆数据量来提升性能在实践上越来越不可行；另一方面，已有的哈密顿量预测模型（PhiSNet、QHNet、SPHNet 等）只能处理小分子、小基组、少元素，无法扩展到 MLIP 所需的结构尺寸、基组（含 d/f 轨道）和元素多样性。

**核心矛盾**：哈密顿量矩阵蕴含的电子结构信息既"量大"（O(N²) 而非 O(N)）又"质优"（含远超力/能量的物理信息），却卡在两个工程瓶颈上无法利用——**模型扩展性**（高 lmax 下张量积的 O(lmax⁶) 复杂度、大结构的内存爆炸）和**数据缺失**（缺乏元素/尺寸多样的大规模哈密顿量数据集）。

**本文目标**：打通"哈密顿量预测"与"通用 MLIP"之间的鸿沟，提供一条把 H 中轨道相互作用数据整合进原子性质训练管线的可行配方。

**核心 idea**：**电子相互作用作为一种丰富且可迁移的数据源**——先在哈密顿量矩阵上训练一个可扩展的等变骨干网络，让它学到对原子环境的精细描述子，再把这个共享嵌入空间迁移（冻结/微调）到能量预测任务上，从而在能量标签稀缺时也能高效学习。

## 方法详解

### 整体框架
HELM（Hamiltonian-trained Electronic-structure Learning for Molecules）由一个**权重共享的特征提取骨干**加上**两个独立输出头**组成：一个哈密顿量头预测矩阵 H，一个能量头从原子结构预测总能量。典型训练流程是先在 H 上训练骨干，再把骨干特征直接复用或微调到能量预测——这正是"哈密顿量预训练"的实现方式。

```mermaid
flowchart LR
    A["分子图<br/>位移向量 r_ij + 原子类型 Z_i"] --> B["等变 GNN 骨干<br/>K 层 SO2 卷积 + gating<br/>节点+有向边嵌入"]
    B --> C["哈密顿量头<br/>z_i, z_ij → H 各子块 irreps"]
    B --> D["能量头<br/>z_i 的 l=0 分量 → 总能量 E"]
    C -.预训练后冻结/微调.-> D
```

### 关键设计

**1. 节点-边双预测的骨干，并让边单向依赖节点：** 与 MLIP 只做逐节点预测不同，学哈密顿量要同时预测节点（同原子内轨道相互作用 Hᵢᵢ）和有向边（原子间轨道相互作用 Hᵢⱼ）。HELM 基于等变消息传递 GNN，把节点/边嵌入初始化为多通道球谐系数（形状 $(l_{max}+1)^2 \times C$），经 K 层更新。关键的结构性约束是：第 $k+1$ 层节点嵌入由第 $k$ 层节点嵌入算出，但第 $k$ 层**边嵌入只由同层它所连的两个节点算出**（$z^{(k+1)}_{ij} = f_{edge}([y^{(k)}_i, y^{(k)}_j], r_{ij})$）。这种边对节点的单向依赖一举两得：训练时天然引入了 H 中库仑电子积分随 $1/r$ 衰减的物理先验；复用骨干做能量预测时又能直接省掉边更新，省算力。

**2. 用 SO(2) 卷积取代全张量积以驯服高 lmax：** 处理含 d、f 轨道的大基组需要 lmax 高达 6，而传统全张量积混合球谐系数的复杂度是 $O(l_{max}^6)$，根本扛不住。HELM 在嵌入更新里改用 Passaro & Zitnick 的 SO(2) 卷积，把复杂度从 $O(l_{max}^6)$ 降到 $O(l_{max}^3)$，这是模型能扩展到 100+ 原子、58 元素、含弥散函数大基组（def2-TZVPD）的核心使能技术。哈密顿量头还做了两处适配：对非零阶 irrep 乘以经 sigmoid 的可学习标量做门控非线性，以区分大基组下同主量子数的多个同型轨道壳层；对角块（同原子轨道间相互作用）额外施加宇称约束 $\delta((-1)^{\ell_1+\ell_2}, (-1)^{\ell_3})$，只计算唯一且非零的节点值。

**3. 适配大体系/多元素的损失与参考值预处理：** 朴素的逐元素 MAE/MSE 损失在跨尺寸、跨元素的大数据集上失效——重元素越多，H 矩阵元幅值分布越发散；逐分量 MAE 还会对旋转上等价的边产生偏置。而 Li et al. 的"波函数对齐损失"虽好，却要求每次前向后重组 H，在 100+ 原子时成为计算瓶颈。HELM 借鉴能量预测里的"每元素参考值"思路，先把节点标签的 $l=0$ 分量（大致来自元素特异的局域核电子态）做缩放和中心化以拉平不同元素间的方差，再用与 irrep 朝向无关的 root-MSE+MSE 损失训练，避开了重组 H 的开销。

**4. 直接从节点嵌入读出能量的能量头：** 虽然总能量原则上可由预测的 H 重新算出，但这需要把 HELM 内部表示重组回 H，且交换关联能还要在网格上数值积分电子密度，对大结构做基于梯度的力预测或直接能量微调都太慢。HELM 因此设计了一个直接映射 $E = f_E([z_i^{(K)}])$ 的能量头：用单个嵌入更新块，再对 $l=0$（标量）分量做线性变换 $w$，最后对分子内所有节点求和得到 $E = \sum_{i=1}^{N} w^\top z^{(K+1)}_{i,l=0}$。

## 实验关键数据

### 主实验表格
HELM 在两个公开哈密顿量基准上均达到 SOTA（矩阵元误差 Herr，单位 ×10⁻⁶ Eh）：

| 模型 | Water | Ethanol | Malondial. | Uracil | ∇²DFT-2k | ∇²DFT-5k | ∇²DFT-10k |
|------|-------|---------|-----------|--------|----------|----------|-----------|
| SchNOrb | 165.4 | 187.4 | 191.1 | 227.8 | 21500 | 20700 | 20700 |
| PhiSNet | 17.59 | 12.15 | 12.32 | 10.73 | 180 | 330 | 350 |
| QHNet | 10.79 | 20.91 | 21.52 | 20.12 | 840 | 730 | 520 |
| SPHNet | 23.18 | 21.02 | 20.67 | 19.36 | – | – | – |
| **HELM** | **9.33** | **5.79** | **4.86** | **3.61** | **60.33** | **57.41** | **59.21** |

在 ∇²DFT 上 HELM 的矩阵元误差约 60 µEh，比此前最佳模型好约 3–5 倍；用预测 H 重算总能量可在每分子 30 meV 内复现参考值。

### 消融实验表格
哈密顿量预训练对低数据能量预测的效果（∇²DFT 三个 split，能量 MAE，单位 meV，含 95% 置信区间）：

| Split | 训练法 | Train Eerr | Test Eerr |
|-------|--------|-----------|-----------|
| 2k | Direct | 97.16 | 791.02 |
| 2k | Pretrained-frozen | 217.58 | 324.64 |
| 2k | Finetuned | **76.30** | **266.23** |
| 5k | Direct | 240.63 | 592.54 |
| 5k | Finetuned | **116.52** | **199.50** |
| 10k | Direct | 285.39 | 506.80 |
| 10k | Finetuned | **176.75** | **198.40** |

在更难的 OMol CSH 58k 上，pretrained-frozen 比 direct 模型测试精度提升约 1.8×（OMol common 1k：3631→2119 meV；OMol all 5k：5976→3255 meV）。

### 关键发现
- **Direct 模型在有限能量标签上迅速过拟合**，而预训练（冻结或微调）同时降低过拟合并提升测试精度，且提升幅度在最小的 2k split 上最大。
- **UMAP 可视化**显示 pretrained-frozen 嵌入相比 direct 出现大量额外的清晰聚类，且对数据中欠采样的重元素区分更明显——证明从哈密顿量数据中确实学到了更精细的原子环境描述子；在 OMol 上微调会削弱这种精细结构，在 ∇²DFT 上微调则保留它，与各自测试精度趋势一致。
- 作者估算：要靠单纯堆力/能量数据达到同等提升，需要多出**一个数量级以上**的 DFT 计算。

## 亮点与洞察
- **变废为宝的视角**：把 DFT 计算中"顺带产出却被丢弃"的 O(N²) 哈密顿量矩阵当作监督信号，在数据墙逼近时另辟蹊径——这是一个数据效率上的杠杆点。
- **工程上真正扩展了哈密顿量预测的边界**：SO(2) 卷积 + 门控 + 宇称约束 + 参考值预处理这一套组合拳，把哈密顿量预测从"小分子玩具"推进到能与现代 MLIP 比肩的尺寸/元素/基组规模。
- **OMol CSH 58k 数据集本身是贡献**：58 种元素（覆盖前 83 号元素除首行过渡金属和镧系）、10–150 原子、def2-TZVPD 大基组、相互作用距离达 15 Å，是迄今结构尺寸、矩阵尺寸、元素数最大的分子哈密顿量数据集，并提供 OMol all 5k / OMol common 1k 两个测试 split。
- **预训练-表示分析闭环漂亮**：不仅给出精度数字，还用 UMAP 把"为什么预训练有用"可视化成嵌入空间的聚类结构，把性能提升归因到更精细的原子环境描述子。

## 局限与展望
- **内存仍是天花板**：预训练规模受限于内存——边标签数和基组（角动量）随体系增大而增长，作者特意排除了含 g 轨道的首行过渡金属和镧系元素以控制显存。
- **直接由 H 算能量对扰动极敏感**：矩阵元的微小误差会被放大到能量上，因此仍需在高精度能量标签上微调来补足精度。
- **OMol 上微调会损害精细表示**：在元素多样性高、重元素能量标签采样不足时，微调反而加剧过拟合、削弱表示结构，提升幅度（约 1.5×）低于冻结（约 1.8×）。
- **离实用 MLIP 精度尚有距离**：当前只是低数据受控实验，要达到可用精度还需在更大比例 OMol25 能量数据上微调，并需要计算创新来支撑更大标签的内存/数据处理、以及融入特征值/对称性损失。

## 相关工作与启发
- **等变 GNN 与哈密顿量预测谱系**：从 PhiSNet 的全张量积，到 QHNet/SPHNet 利用 CG 系数稀疏性提效，再到用 SO(2) 卷积降复杂度（Li et al., Yu et al.）——HELM 站在这条扩展性优化的主线上，并把它推到通用尺度。
- **通用 MLIP**：MACE、UMA 等在大规模异构数据上预训练单一模型覆盖周期表，HELM 试图为它们补上"电子结构信号"这块拼图。
- **损失函数方向**：波函数对齐损失（Li et al.）、对称性损失（Qian et al.）、以及在 QHNet 上加 flow-matching 把精度压到 5 µEh 以下（Kim et al.）都与本文正交，可叠加使用。
- **启发**：在任何"昂贵仿真顺带产出大量中间量"的科学计算场景（不止量子化学），都值得问一句——那些被丢弃的中间表示能否当作自监督/预训练信号来提升下游数据效率？

## 评分
- **新颖性**: ⭐⭐⭐⭐ 把被忽视的哈密顿量矩阵作为可迁移预训练信号这一视角新颖，且"哈密顿量预训练"提供了清晰可复用的范式。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 MD17/QM7 与 ∇²DFT 两个基准、三种训练方案的受控低数据实验、UMAP 表示分析，并配套发布数据集；但离实用 MLIP 精度仍有距离，重元素采样不足。
- **写作质量**: ⭐⭐⭐⭐ 动机—方法—实验逻辑清晰，架构图和损失曲线到位，物理先验（1/r 衰减、宇称约束）解释得当。
- **价值**: ⭐⭐⭐⭐ 在 MLIP 数据墙逼近的当下指出一条用电子结构数据破局的路径，数据集与模型对 AI for Science 社区都有实际价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Advancing Universal Deep Learning for Electronic-Structure Hamiltonian Prediction of Materials](advancing_universal_deep_learning_for_electronic-structure_hamiltonian_predictio.md)
- [\[ICLR 2026\] OXtal: An All-Atom Diffusion Model for Organic Crystal Structure Prediction](oxtal_an_all-atom_diffusion_model_for_organic_crystal_structure_prediction.md)
- [\[ICLR 2026\] MatRIS: Toward Reliable and Efficient Pretrained Machine Learning Interatomic Potentials](matris_toward_reliable_and_efficient_pretrained_machine_learning_interatomic_pot.md)
- [\[ICLR 2026\] DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs](dgnet_discrete_green_networks_for_data-efficient_learning_of_spatiotemporal_pdes.md)
- [\[ICLR 2026\] Beyond Structure: Invariant Crystal Property Prediction with Pseudo-Particle Ray Diffraction](beyond_structure_invariant_crystal_property_prediction_with_pseudo-particle_ray_.md)

</div>

<!-- RELATED:END -->
