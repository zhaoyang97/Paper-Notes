---
title: >-
  [论文解读] H2OFlow: Grounding Human-Object Affordances with 3D Generative Models and Dense Diffused Flows
description: >-
  [ICLR 2026][3D视觉][3D Affordance] H2OFlow 用 3D 生成模型造合成 HOI 数据、再以点云上的"稠密扩散流"（dense diffused flow）建模人体到目标姿态的逐点位移分布，**完全不用人工标注**就同时学出接触、朝向、空间占据三种 3D 可供性，并能泛化到真实点云。
tags:
  - "ICLR 2026"
  - "3D视觉"
  - "3D Affordance"
  - "Human-Object Interaction"
  - "Dense Diffused Flow"
  - "Transformer"
  - "点云"
  - "Synthetic Data"
---

# H2OFlow: Grounding Human-Object Affordances with 3D Generative Models and Dense Diffused Flows

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=QhqJ1DCp1X](https://openreview.net/forum?id=QhqJ1DCp1X)  
**代码**: 待确认  
**领域**: 3D 视觉 / 人-物交互（HOI）/ 三维可供性（Affordance）  
**关键词**: 3D Affordance, Human-Object Interaction, Dense Diffused Flow, Diffusion Transformer, Point Cloud, Synthetic Data  

## 一句话总结
H2OFlow 用 3D 生成模型造合成 HOI 数据、再以点云上的"稠密扩散流"（dense diffused flow）建模人体到目标姿态的逐点位移分布，**完全不用人工标注**就同时学出接触、朝向、空间占据三种 3D 可供性，并能泛化到真实点云。

## 研究背景与动机
- **领域现状**: 人-物可供性（affordance）是机器人、视觉、具身智能的核心能力——agent 要理解一个物体"能怎么用"。但现有 3D HOI 可供性方法绝大多数只学 **接触（contact）**，靠 RGB 图、点云或人体模型上密集标注的接触区域来监督。
- **现有痛点**: 一是接触标注**人工成本极高且难泛化**到新物体；二是只看接触太狭隘——人和物的交互还包含**朝向**（如人面对电视有偏好朝向、握锤子时手腕有特定角度）和**空间占据**（如人更可能站在微波炉前面而非后面）这些非接触几何模式。
- **核心矛盾**: 最近 COMA（Kim et al. 2024）提出"comprehensive affordance"概率化地建模接触+非接触关系，但它依赖把合成 RGB 图 **2D→3D 抬升**，需要复杂掩码、易失败，且要求**水密网格（watertight mesh）**算法向量，对真实带噪点云几乎不可用，泛化性差。
- **本文目标**: 在**零人工标注**、**只用点云**（无需水密网格）的前提下，同时学接触、朝向、空间三类可供性，并能泛化到真实采集的不完整点云。
- **核心 idea**: **【生成造数据 + 流做中间表示】** 用预训练 3D 生成模型直接生成 3D HOI 样本（绕开易错的 2D→3D 抬升），再用 **稠密扩散流**——在点云上学一个"从 T-pose 人体到目标交互姿态的逐点位移分布"——作为可泛化的中间表示，最后从采样出的流和点云里**解析地**推导三类可供性。

## 方法详解

### 整体框架
H2OFlow 分三段：① 用预训练 3D 生成模型 CHOIS 从文本生成多样 HOI 网格序列作为训练数据；② 把人/物网格采成点云，训练一个 DiT 扩散模型，学"以物体点云为条件、预测人体逐点位移流 $F$"的分布 $p_\theta(F\mid O)$；③ 推理时对未见物体点云采样多条流、重建大量目标人体配置，再在人-物点对分布上**解析地**算出接触、朝向、空间三种可供性分数。关键在于把"人怎么动"编码成稠密流分布，从而既建模交互多模态、又脱离对水密网格的依赖。

```mermaid
flowchart LR
    A[文本 prompt] --> B[预训练3D生成模型 CHOIS<br/>生成HOI网格序列]
    B --> C[采样点云<br/>人体H / 物体O]
    C --> D[训练 DiT 扩散模型<br/>学稠密扩散流 p_θ&#40;F|O&#41;]
    E[未见物体点云 O] --> F[采样多条流 F~p_θ&#40;F|O&#41;<br/>H=H0+F 重建目标人体]
    D --> F
    F --> G[点对分布 P_ij]
    G --> H1[接触可供性 C]
    G --> H2[朝向可供性 R]
    G --> H3[空间可供性 S]
```

### 关键设计

**1. 稠密扩散流表示：把人体交互编码成逐点位移分布** 核心是不直接预测人体姿态，而是预测**流场** $F=\{f_i\}$。固定一个标准 T-pose（0-pose）SMPL 人体点云 $H_0$，从目标交互网格采**同样 $N_H$ 个点**保证一一对应，于是流定义为逐点位移 $f_i := h_i - h_{0,i}$，即 $F := H - H_0$，重建时 $H = H_0 + F$。这种"流"表示对刚体和可变形体都适用、天然落在点云上，是 H2OFlow 能泛化到未见物体的根本原因——模型学的是**局部几何线索如何驱动人体点位移**，而非记忆全局网格模板。

**2. 用 DiT 扩散建模多模态流分布** 因为 HOI 高度多模态（左手/右手、不同区域都可能接触），单一确定性预测不够，作者学一个**条件分布** $p_\theta(F\mid O)$。采用标准扩散：前向加噪 $F_t = \sqrt{\bar\alpha_t}F_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$，反向学去噪 $p_\theta(F_{t-1}\mid F_t)$，用 Nichol & Dhariwal 的混合损失（噪声损失 + 基于 $\Sigma_\theta$ 的累积 KL 损失）监督。骨干用 **Diffusion Transformer（DiT）**：对噪声流 $F_t$、人体 $H$、物体 $O$ 各用共享权重 MLP 提逐点特征，把流+人体特征拼成 $f_{FH}$ 作输入、以物体特征 $f_O$ 作条件；每个 DiT block 先在 $f_{FH}$ 上做**自注意力**（人体内部协调流预测），再 $f_{FH}$ 与 $f_O$ 做**交叉注意力**（捕捉全局人-物交互）。这里的交叉注意力权重 $w_{ij}$ 后面会被复用进可供性打分。

**3. 从流解析推导三类可供性** 推理时对未见物体采样流得到一批目标人体，定义人点 $h_i$ 相对物点 $o_j$ 的条件分布 $P_{ij}:=p(h_i\mid o_j)$，三类可供性都是该分布上的期望：
- **接触**：用人-物点距加权交叉注意力，$c_{ij} = \mathbb{E}_{h_i\sim P_{ij}}\!\left[w_{ij}\cdot \exp(-\|d_{ij}\|)/\tau\right]$，其中 $d_{ij}=h_i-o_j$，越近分越高。
- **朝向**：不像 COMA 去算易受噪声影响的表面法向，而是**直接用流向量做方向代理**——取位移与流的叉积 $x_{ij} = (d_{ij}\times f_i)/\|d_{ij}\times f_i\|$，把单位球离散成 $n_b$ 个 bin 用高斯核估概率 $p_{x,ij}(n)$，再用**负香农熵** $H_{ij}=\mathbb{E}[\log p_{x,ij}(n)]$ 度量朝向一致性，最终 $R_{ij}=\mathbb{E}_{h_i\sim P_{ij}}[w_{ij}\cdot H_{ij}/\tau]$——熵低（朝向集中，如手）则可供性高，熵高（杂乱，如脚）则低。
- **空间**：在物体周围建体素网格 $G\in\mathbb{R}^{H\times W\times L}$，指示函数 $\delta_{ij}$ 标记体素是否被 $h_i$ 占据，$S_{ij}=\mathbb{E}_{h_i\sim P_{ij}}[\delta_{ij}]$ 即期望占据率，得到人体在物体周围的占据概率图。三者都只在采样的少量点上算、可 GPU 并行、不需高质量网格。

## 实验关键数据

### 主实验表格（OMOMO 数据集，与 COMA 等对比）

| 方法 | SIM-H↑ | SIM-O↑ | MAE-H↓ | MAE-O↓ | Precision@K↑ | MSE↓ |
|------|--------|--------|--------|--------|--------------|------|
| COMA | 41.3% | 56.9% | 0.22 | 0.14 | 42.9% | 0.14 |
| COMA-Recon（点云重建网格输入） | 20.7% | 31.8% | 0.62 | 0.51 | 9.1% | 0.66 |
| H2OSMPL（直接预测SMPL参数变体） | 57.3% | 68.0% | 0.21 | 0.15 | 53.6% | 0.14 |
| H2OFlow-NoAttn（去交叉注意力） | 67.3% | 76.4% | 0.15 | 0.09 | 69.2% | 0.12 |
| **H2OFlow** | **72.3%** | **81.0%** | **0.11** | **0.07** | **75.6%** | **0.12** |

H2OFlow 全指标大幅领先。值得注意的是 **COMA 在输入来自点云重建的网格图时性能崩塌**（COMA-Recon），印证它依赖干净水密网格、不适配真实点云。

### 消融实验表格（下游 HOI 推理任务）

| 变体 | mAP@1↑ | mAP@5↑ | mAP@10↑ | Top-5 Acc↑ | Collision↓ | Leakage↓ |
|------|--------|--------|---------|------------|-----------|----------|
| C（仅接触） | 0.52 | 0.60 | 0.66 | 0.41 | 0.31 | 0.27 |
| C+O（+朝向） | 0.57 | 0.63 | 0.69 | 0.55 | 0.26 | 0.22 |
| C+S（+空间） | 0.56 | 0.62 | 0.68 | 0.51 | 0.21 | 0.20 |
| **C+O+S（全部）** | **0.63** | **0.69** | **0.76** | **0.64** | **0.18** | **0.17** |
| Shuffled（打乱R/S对照） | 0.54 | 0.61 | 0.66 | 0.44 | 0.29 | 0.26 |

朝向和空间可供性各自都带来显著增益，三者叠加最好；**Shuffled 对照（保留接触但随机打乱朝向/空间）抹掉了增益**，证明提升源自结构化的朝向/空间信息本身，而非单纯增加特征容量。

### 关键发现
- **学流 > 直接学 SMPL 参数**：H2OSMPL 变体明显弱于 H2OFlow，说明逐点流表示更利于泛化。
- **交叉注意力是低采样下的纠错信号**：去掉注意力后接触/朝向可供性对称性变差；在只采少量样本时，注意力权重用学到的人-物几何关联补偿稀疏采样，产出更对称合理的可供性图。
- **真实点云鲁棒**：用 iPhone 深度相机+RealityKit 采集、FPS 下采样的真实点云上仍能输出多模态、语义合理的可供性（如背包不同部件、头部周围朝向），得益于训练时对物体点云加随机扰动和遮挡，无需完整扫描；COMA 在此场景退化成过简化的单模态图。
- **更省内存、更快**：因操作在稀疏点云上，内存和运行时都显著优于 COMA。

## 亮点与洞察
- **"造数据"换"标数据"**：用文本驱动的 3D 生成模型直接合成 HOI，绕开既贵又难泛化的人工接触标注，也避免 COMA 的 2D→3D 抬升失败模式。
- **流作为统一中间表示**：把"人怎么交互"统一成逐点位移分布，三种可供性都从同一个流分布**解析地**导出，框架简洁且互相一致。
- **用流向量替代法向算朝向**很巧：叉积 $d_{ij}\times f_i$ 把"相对位移方向"与"运动方向"耦合成一个稳定的方向量，避开了在噪声网格上算法向的不稳定与昂贵。
- **熵作为朝向一致性度量**直观可解释：手（一致朝向）熵低分高、脚（杂乱）熵高分低，符合直觉。

## 局限与展望
- **依赖预训练生成模型质量**：训练数据完全由 CHOIS 生成，生成模型的覆盖范围、物体类别与交互多样性会成为可供性学习的上界；冷门物体/交互可能受限。
- **0-pose 与物体的相对放置**需要先验设定（附录 B），这一步若不准会影响流的语义。
- **三类可供性的下游融合**用的是归一化线性加权 $\phi_{ij}=\lambda_c\hat c_{ij}+\lambda_o\hat R_{ij}+\lambda_s\hat S_{ij}$，权重需调，缺乏自适应融合机制。
- 评估主要在 OMOMO/BEHAVE 等室内家居物体上，更大尺度场景、多人或动态交互尚未验证。

## 相关工作与启发
- **COMA（Kim et al. 2024）**：最直接对标，首提 comprehensive affordance（接触+非接触概率化建模），但依赖 2D→3D 抬升 + 水密网格 + 表面法向，泛化与真实点云适配差；H2OFlow 在表示（流 vs 网格法向）和数据（生成 vs 抬升）两层都做了替换。
- **3D Flow（Eisner et al. 2022 / Xu et al. 2024 / Cai et al. 2024）**：流在策略学习与物体理解中证明有效，本文把刚体/铰接物的流思想**迁移到人体**，并用扩散建模其多模态。
- **3D HOI 生成（CHOIS / Diller & Dai / Peng et al.）**：本文复用 CHOIS 作为数据引擎，体现"用生成模型当合成数据源"的范式，对需要稀缺 3D 交互标注的任务有借鉴意义。
- 启发：当某任务标注昂贵且现有表示（如水密网格）限制真实部署时，"生成造数据 + 在点云上学可泛化中间表示（流）+ 解析导出目标量"是一条值得复用的范式。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 稠密扩散流作为 HOI 可供性中间表示、用流向量叉积+熵替代法向算朝向，组合新颖且解决了 COMA 的真实泛化痛点。
- **实验充分度**: ⭐⭐⭐⭐ OMOMO/BEHAVE 定量、Shuffled 对照消融、真实 iPhone 点云、内存/速度对比都有，证据链完整；多人/大场景未覆盖。
- **写作质量**: ⭐⭐⭐⭐ 问题动机清晰、三类可供性公式化定义到位、图示丰富；部分细节下放附录。
- **价值**: ⭐⭐⭐⭐ 零标注学多类型 3D 可供性、能上真实点云，对机器人/具身交互落地价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] QueryMe: Query-Driven Open-Vocabulary 3D Object Affordances Grounding from Multimodal Evidence](../../CVPR2026/3d_vision/queryme_query-driven_open-vocabulary_3d_object_affordances_grounding_from_multim.md)
- [\[ICLR 2026\] Generative Human Geometry Distribution](generative_human_geometry_distribution.md)
- [\[ICLR 2026\] SpatialHand: Generative Object Manipulation from 3D Perspective](spatialhand_generative_object_manipulation_from_3d_prespective.md)
- [\[CVPR 2026\] Affostruction: 3D Affordance Grounding with Generative Reconstruction](../../CVPR2026/3d_vision/affostruction_3d_affordance_grounding_with_generative_reconstruction.md)
- [\[ICLR 2026\] Scaling Sequence-to-Sequence Generative Neural Rendering](scaling_sequence-to-sequence_generative_neural_rendering.md)

</div>

<!-- RELATED:END -->
