---
title: >-
  [论文解读] Multi-modal Knowledge Distillation-based Human Trajectory Forecasting
description: >-
  [CVPR 2025][模型压缩][行人轨迹预测] 本文提出首个用于行人轨迹预测的多模态知识蒸馏框架——用轨迹+人体姿态+文本描述训练全模态教师模型，将其知识蒸馏到仅用轨迹或轨迹+姿态的学生模型，在JRDB/SIT/ETH-UCY三个数据集上最高提升约13%预测精度。 1. 领域现状：行人轨迹预测基于历史2D轨迹序列预测未来…
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "行人轨迹预测"
  - "多模态知识蒸馏"
  - "人体姿态"
  - "文本描述"
  - "瞬时预测"
---

# Multi-modal Knowledge Distillation-based Human Trajectory Forecasting

**会议**: CVPR 2025  
**arXiv**: [2503.22201](https://arxiv.org/abs/2503.22201)  
**代码**: [https://github.com/Jaewoo97/KDTF](https://github.com/Jaewoo97/KDTF)  
**领域**: 自动驾驶 / 轨迹预测  
**关键词**: 行人轨迹预测, 多模态知识蒸馏, 人体姿态, 文本描述, 瞬时预测

## 一句话总结
本文提出首个用于行人轨迹预测的多模态知识蒸馏框架——用轨迹+人体姿态+文本描述训练全模态教师模型，将其知识蒸馏到仅用轨迹或轨迹+姿态的学生模型，在JRDB/SIT/ETH-UCY三个数据集上最高提升约13%预测精度。

## 研究背景与动机
1. **领域现状**：行人轨迹预测基于历史2D轨迹序列预测未来运动，广泛应用于自动驾驶、移动机器人导航和监控系统。主流方法如HiVT（图注意力）和MART（Transformer）基于纯轨迹序列建模。近年来有研究尝试利用视觉线索（如人体姿态、边界框）增强预测。
2. **现有痛点**：(a) 仅凭2D坐标序列难以准确推断行人的运动意图，因为行人通过视觉信号（转身、抬臂等）传达意图；(b) 文本描述对模态融合效果显著但获取成本高——需要VLM在线生成，对资源受限系统不可行；(c) 在遮挡频繁的场景下，少量观测帧的瞬时预测更具挑战性。
3. **核心矛盾**：多模态（尤其是文本）能显著提升预测精度，但推理时获取额外模态的计算成本过高。如何在推理时不使用昂贵模态、同时保留其带来的性能提升？
4. **本文目标** (1) 如何将文本中蕴含的高层语义运动理解传递给轻量级模型？(2) 如何将agent内多模态融合+agent间交互两层知识分别蒸馏？(3) 如何在瞬时观测（仅1-2帧）的极端情况下也能有效预测？
5. **切入角度**：作者发现文本描述在整合不同模态时起关键"桥接"作用——文本弥合了轨迹与姿态之间的域差距，即使姿态噪声大也能通过文本的语义上下文充分利用姿态信息。基于此，设计KD框架让学生模型通过训练时对齐教师的嵌入空间来隐式获得这种语言驱动的理解。
6. **核心 idea**：训练时用全模态教师指导有限模态学生，通过分别对齐intra-agent和inter-agent嵌入空间来传递运动意图的多模态理解。

## 方法详解

### 整体框架
两阶段训练。第一阶段：训练全模态教师模型（轨迹 $\mathcal{X}$ + 3D姿态 $\mathcal{P}$ + 文本 $\mathcal{S}$），使用回归损失对完整/2帧/1帧三种观测设置联合训练。第二阶段：冻结教师模型，从头训练学生模型（仅 $\mathcal{X}$ 或 $\mathcal{X}+\mathcal{P}$），在回归损失基础上增加KD损失，对齐局部编码器输出 $Q$（intra-agent）和全局编码器输出 $H$（inter-agent）的分布。教师和学生共享相同网络结构，仅输入模态数不同。

### 关键设计

1. **模态嵌入与局部编码器（Intra-agent融合）**:

    - 功能：将各模态编码为统一嵌入并融合为单agent运动意图表示 $q_n$
    - 核心思路：用MLP分别编码轨迹 $z_x$、SMPL姿态参数 $z_p$，用预训练TinyBERT编码文本 $z_s$。对于HiVT：逐帧通过图网络处理，对每个agent融合自身模态与邻居模态（邻居的轨迹和姿态经旋转不变变换），然后用Transformer编码时序：$q_n^t = \psi_\mathcal{M}([(z_x,z_p,z_s)_i, (z_x,z_p,z_s)_j, (v_{ji})_e])$。对于MART：用Transformer同时在模态维和时序维做全局注意力，加class token汇聚：$q_n = \phi_{\mathcal{M},T_p}(\bar{q_n}, z_x, z_p, z_s)$
    - 设计动机：HiVT的图结构天然支持为每对agent-neighbor关系分别融入姿态信息（旋转不变），更精细地捕捉微妙交互暗示。SMPL表示比关节点更适合跨数据集泛化

2. **全局编码器（Inter-agent交互建模）**:

    - 功能：在 $q_n$ 基础上建模agent间交互关系，得到完整的运动意图表示 $H$
    - 核心思路：MART用标准Transformer注意力建模全局交互：$H = \phi_N(Q)$。HiVT利用图网络并将文本中的agent关系描述（如"他们在一起聊天"）编码为边属性：$H = \psi_N([(q_n)_i, (q_n)_j, (v_{ji}, s_{R,ji})])$，其中 $s_{R,ji}$ 是描述两个agent关系的文本嵌入
    - 设计动机：JRDB数据集有agent间关系的文本标注，HiVT的图结构可以自然地为每条边引入关系文本，使全局编码器也能受益于文本信息

3. **双层知识蒸馏（KD Loss设计）**:

    - 功能：分别传递intra-agent多模态融合知识和inter-agent交互知识
    - 核心思路：用KL散度对齐教师和学生的 $Q$ 和 $H$ 分布。对MART：$\mathcal{L}_{KD} = \mathcal{L}_{KL}(Q_\mathcal{T}\|Q_\mathcal{S}) + \mathcal{L}_{KL}(H_\mathcal{T}\|H_\mathcal{S})$。对HiVT需额外稳定性，使用余弦相似度+正则：$\mathcal{L}_{KD}^L = \lambda_{cos}\mathcal{L}_{cos}(Q_\mathcal{T}, Q_\mathcal{S}) + \mathcal{L}_{KL}(\mathcal{N}\|Q_\mathcal{S})$。三种观测设置（完整/2帧/1帧）各自独立计算KD损失
    - 设计动机：分层蒸馏让学生模型在agent本身的运动理解（$Q$）和社会交互理解（$H$）两个层面都向教师对齐，比仅对齐最终输出更有效

### 损失函数 / 训练策略
学生总损失：$\mathcal{L} = \lambda_{reg}L_{reg}^F + L_{reg}^2 + L_{reg}^1 + \mathcal{L}_{KD}^F + \mathcal{L}_{KD}^2 + \mathcal{L}_{KD}^1$，其中 $\lambda_{cos}=0.5, \lambda_{reg}=3$。HiVT用NLL回归损失，MART用L2损失。三种观测设置联合训练使模型同时擅长完整观测和瞬时预测。JRDB使用人工标注文本，SIT使用PLLaVa生成的文本，ETH/UCY使用规则生成的地图描述文本。

## 实验关键数据

### 主实验

| 数据集 | 模型 | 学生模态 | KD | ADE | ADE₁ | FDE | FDE₁ | Ave.+% |
|--------|------|---------|-----|-----|------|-----|------|--------|
| JRDB | HiVT | 𝒳 | ✗ | 0.221 | 0.342 | 0.432 | 0.632 | - |
| JRDB | HiVT | 𝒳 | ✓ | **0.220** | **0.326** | **0.438** | **0.604** | +2.38 |
| JRDB | HiVT | 𝒳+𝒫 | ✗ | 0.229 | 0.364 | 0.441 | 0.659 | - |
| JRDB | HiVT | 𝒳+𝒫 | ✓ | **0.232** | **0.308** | **0.445** | **0.560** | +4.98 |
| SIT | HiVT | 𝒳+𝒫 | ✗ | 0.518 | 0.531 | 0.979 | 1.006 | - |
| SIT | HiVT | 𝒳+𝒫 | ✓ | **0.414** | **0.500** | **0.789** | **0.951** | +13.03 |
| JRDB | MART | 𝒳 | ✗ | 0.286 | 0.395 | 0.545 | 0.753 | - |
| JRDB | MART | 𝒳 | ✓ | **0.259** | **0.366** | **0.495** | **0.684** | +7.61 |

多模态教师模型性能（JRDB+MART）：

| 模态 | ADE | ADE₁ | Ave.+% |
|------|-----|------|--------|
| 𝒳 | 0.286 | 0.395 | - |
| 𝒳+𝒫 | 0.287 | 0.366 | +2.02 |
| 𝒳+𝒮 | 0.261 | 0.301 | +12.41 |
| 𝒳+𝒫+𝒮 | **0.258** | **0.289** | **+14.98** |

### 消融实验

| KD-Local | KD-Global | ADE₁ | FDE₁ | Ave.+% | 说明 |
|----------|-----------|------|------|--------|------|
| ✗ | ✗ | 0.364 | 0.659 | - | 无KD基线 |
| ✓ | ✗ | 0.352 | 0.647 | +1.5 | 仅蒸馏intra-agent |
| ✗ | ✓ | 0.345 | 0.637 | +2.7 | 仅蒸馏inter-agent |
| ✓ | ✓ | **0.308** | **0.560** | **+4.98** | 双层蒸馏 |

### 关键发现
- **文本是多模态融合的关键桥梁**：HiVT中仅加姿态甚至有负面影响（-2.84%），但加文本后显著提升(+6.53%)，文本+姿态进一步提升到+8.38%。文本弥合了轨迹与嘈杂姿态之间的语义鸿沟
- 瞬时预测场景下KD效果最显著：ADE₁的改善通常远大于ADE，因为额外模态在观测不足时提供了关键的语义补充
- SIT数据集（较小）上KD效果最强（+13%），因为小规模数据下基础模型难以建立多模态关联，KD帮助传递了大规模预训练获得的理解
- 即使学生模型仅用轨迹（最简配置），KD仍能稳定提升性能，说明数值轨迹的潜力可被语义上下文知识释放

## 亮点与洞察
- **文本作为模态桥梁的洞察**是本文最有价值的发现：文本不仅本身提供语义信息，更重要的是它让模型能正确理解和利用嘈杂的姿态信号。这个insight可迁移到任何多模态融合问题——添加语义类型的模态可能比添加更多相似模态更有效
- **双层蒸馏设计**（intra+inter）优于单层：将轨迹预测分解为"理解个体运动意图"和"建模社会交互"两个层面分别对齐，比直接对齐最终输出更精准
- 三种观测设置（完整/2帧/1帧）联合训练使得一个模型同时擅长常规和瞬时预测，无需separate model

## 局限与展望
- 3D姿态提取依赖外部模型质量，噪声较大时反而有负面影响（HiVT中 $\mathcal{X}+\mathcal{P}$ 比纯 $\mathcal{X}$ 差2.84%）
- ETH/UCY的BEV视角下无法提取有效姿态和文本（VLM对BEV视角不擅长），退而使用CLIP图像特征和规则文本，效果有限（+1.55~3.80%）
- VLM生成文本的质量直接影响教师模型上限，未来可探索更强的VLM或多轮对话提取更精准的行为描述
- 教师-学生结构相同（仅输入模态不同），可尝试更小的学生网络实现进一步加速
- 未考虑场景语义（地图信息、障碍物），可扩展到trajectory+map+pose+text的更丰富多模态场景

## 相关工作与启发
- **vs SocialTransmotion**: ST也利用人体姿态增强轨迹预测，但直接concat输入；本文通过KD框架让学生推理时不需要姿态，实际部署更灵活
- **vs LLM-based轨迹预测(如LCF/DriveGPT)**: 它们推理时需要LLM参与，计算成本极高；本文通过KD在训练后彻底去除文本依赖
- 本框架可适配任何回归式轨迹预测模型（HiVT/MART仅为示例），通用性强

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将多模态KD引入轨迹预测，文本桥接模态融合的洞察有深度
- 实验充分度: ⭐⭐⭐⭐⭐ 3数据集×2模型×多观测设置×多模态组合，覆盖eco-view和BEV-view，非常全面
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，动机逻辑链完整
- 价值: ⭐⭐⭐⭐ KD框架通用性强，文本-模态桥接的发现对多模态学习有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SDKD: Frequency-Aligned Knowledge Distillation for Lightweight Spatiotemporal Forecasting](../../ICCV2025/model_compression/frequency-aligned_knowledge_distillation_for_lightweight_spatiotemporal_forecast.md)
- [\[ACL 2026\] MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation](../../ACL2026/model_compression/mta_multi-granular_trajectory_alignment_for_large_language_model_distillation.md)
- [\[CVPR 2025\] What Makes a Good Dataset for Knowledge Distillation?](what_makes_a_good_dataset_for_knowledge_distillation.md)
- [\[CVPR 2025\] Sketch Down the FLOPs: Towards Efficient Networks for Human Sketch](sketch_down_the_flops_towards_efficient_networks_for_human_sketch.md)
- [\[ICML 2025\] A Cross Modal Knowledge Distillation & Data Augmentation Recipe for Improving Transcriptomics Representations through Morphological Features](../../ICML2025/model_compression/a_cross_modal_knowledge_distillation_data_augmentation_recipe_for_improving_tran.md)

</div>

<!-- RELATED:END -->
