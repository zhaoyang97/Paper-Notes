---
title: >-
  [论文解读] You Share Beliefs, I Adapt: Progressive Heterogeneous Collaborative Perception
description: >-
  [ICCV2025][异构协同感知] 提出PHCP框架，首次在推理阶段解决异构协同感知的域差距问题——通过agent的伪标签做few-shot无监督域适应，自训练适配器对齐特征空间，无需联合训练即在OPV2V上仅用少量无标注数据达到接近SOTA(HEAL)的性能。 领域现状：协同感知(Collaborative Percep…
tags:
  - "ICCV2025"
  - "异构协同感知"
  - "少样本域适应"
  - "自训练"
  - "伪标签"
  - "推理阶段适配"
---

# You Share Beliefs, I Adapt: Progressive Heterogeneous Collaborative Perception

**会议**: ICCV2025  
**arXiv**: [2509.09310](https://arxiv.org/abs/2509.09310)  
**代码**: [GitHub](https://github.com/sihaoo1/PHCP)  
**领域**: 其他  
**关键词**: 异构协同感知, 少样本域适应, 自训练, 伪标签, 推理阶段适配

## 一句话总结
提出PHCP框架，首次在推理阶段解决异构协同感知的域差距问题——通过agent的伪标签做few-shot无监督域适应，自训练适配器对齐特征空间，无需联合训练即在OPV2V上仅用少量无标注数据达到接近SOTA(HEAL)的性能。

## 研究背景与动机

**领域现状**：协同感知(Collaborative Perception)让车辆通过V2X共享信息来扩展感知范围、穿透遮挡。中间特征层融合是当前主流方案，在准确率和带宽之间取得平衡。

**异构挑战**：现实中不同厂商的自动驾驶车辆使用不同传感器配置和感知模型，导致编码后的中间特征处于不同语义空间(域差距)。直接融合异构特征会严重降低性能——实验显示直接融合baseline的AP仅约53%。

**现有方法局限**：
   - MPDA/PnPDA/HEAL等方法通过训练适配器或统一特征空间来对齐特征
   - **关键痛点**：每当新agent加入协作，都需要在数据集上联合训练才能开始工作
   - 不可能为所有潜在协作者预先存储模型——不具备可扩展性

**核心问题**：能否在推理阶段动态调整模型参数来适应不同协作者，完全跳过联合训练？

## 方法详解

### 问题建模：Few-shot无监督域适应
- 将异构协同感知转化为少样本无监督域适应问题
- 目标：在推理阶段用少量无标注数据自适应调整适配器 $\Phi_{i \to ego}$
- 约束：无标注数据、极少帧数、需满足实时性

### 特征适配器设计
- 基于CBAM(Convolutional Block Attention Module)
- **通道注意力(CAM)**：对齐不同编码器在通道维度的特征分布差异
- **空间注意力(SAM)**：聚焦关键空间区域的差异
- 选择依据：可视化分析发现PointPillar和SECOND编码器的特征图在通道维度和关键区域存在系统性错位
- 轻量设计避免过拟合(训练数据极少)

### PHCP协作流程

**Stage I — 适配器微调(前k帧)**：
1. 确立协作关系后，agent在前k帧同时发送中间特征 $\mathbf{F}_i$ 和检测结果
2. 以agent的高置信度预测作为伪标签(保留置信度作为软标签)
3. 构建小样本训练集 $\mathcal{D}_i = \{(d_1, p_1), \dots, (d_k, p_k)\}$
4. 冻结融合网络和检测头，仅微调适配器 $\Phi_{i \to ego}$
5. 共训练20个epoch，使用warmup+multi-step decay学习率策略

**Stage II — 正常推理**：
1. agent仅发送中间特征(与标准中间协作相同)
2. ego用训练好的适配器转换特征：$\mathbf{F}_i' = \Phi_{i \to ego}(\mathbf{F}_i)$
3. 融合转换后的特征并进行最终预测

### 关键设计选择
- **仅微调适配器而非全模型**：多agent同时协作时，各适配器独立互不干扰
- **伪标签用软标签**：保留置信度信息比one-hot更鲁棒
- **k值参照few-shot learning**：1-shot、5-shot、10-shot

## 实验关键数据

### 数据集与设置
- OPV2V + OPV2V-H数据集(CARLA模拟)
- 两类异构agent：LP(PointPillars编码器)和LS(SECOND编码器)
- 16个场景，每个场景分support/query集
- 评价指标：mSAP@IoU 0.3/0.5/0.7

### vs 直接融合baseline

| 指标 | Direct Fusion | PHCP | 提升 |
|------|-------------|------|------|
| mSAP@0.3 | 59.7 | 92.9 | **+33.2** |
| mSAP@0.5 | 59.5 | 92.4 | **+32.9** |
| mSAP@0.7 | 53.0 | 85.9 | **+32.9** |

### vs 其他协同感知方法(mSAP@0.7)

| 方法 | mSAP@0.7 | 训练数据 |
|------|---------|---------|
| F-Cooper | 63.4 | 全量标注 |
| CoBEVT | 72.0 | 全量标注 |
| AttFusion | 77.3 | 全量标注 |
| V2X-ViT | 82.8 | 全量标注 |
| **PHCP(Ours)** | **87.1** | **少量无标注** |
| HEAL(SOTA) | 91.7 | 全量标注 |

- PHCP超过除HEAL外所有方法，且仅用少量无标注数据
- 与HEAL差距仅4.6个点，但HEAL需要全量标注数据训练

### 计算开销

| 阶段 | 配置 | 时间 | 显存 |
|------|------|------|------|
| 训练 | 1-shot | 1.49s | 1290MB |
| 训练 | 5-shot | 2.39s | 5604MB |
| 推理 | - | 0.07s | 798MB |

- 训练仅需1.5-2.4秒(20 iterations)，仅在建立新协作关系时执行一次

### Few-shot数量消融
- 1-shot即可提升约50% AP
- 随shot数增多性能持续提升，但增益递减
- 5-shot已达到很好的性价比

### 伪标签质量分析

| 置信度阈值 | mSAP@0.7 | wSAP@0.7 |
|-----------|---------|---------|
| 0.2 | 85.0 | 66.1 |
| **0.5** | **85.9** | **68.0** |
| 0.7 | 85.8 | 67.7 |
| soft | 85.4 | 67.0 |

- 伪标签质量对最终结果影响较小——归因于SAM能有效聚焦目标区域

### 异构 vs 同构对比

| 方法 | mSAP@0.5 | mSAP@0.7 |
|------|---------|---------|
| Direct Fusion(异构) | 59.5 | 53.0 |
| **PHCP(异构)** | **92.4** | **85.9** |
| SECOND同构 | 94.2 | 90.5 |
| PointPillar同构 | 95.8 | 93.1 |

- PHCP将异构性能差距从40+个点缩小到不到8个点

## 亮点与洞察

- **问题定义的转变**：将异构协同感知从"训练时域适应"重新定义为"推理时few-shot无监督域适应"——这个视角本身就很有价值。现实中确实不可能为每个潜在协作者预训练。
- **1-shot就有50%提升**：说明域差距虽大但结构化——简单的注意力适配就能大幅缓解，不需要复杂的对齐策略。
- **仅适配器训练的隔离性**：多agent协作时各适配器独立，避免了全模型微调的相互干扰——工程上非常实用。
- **伪标签对结果不太敏感**：空间注意力机制提供了天然的鲁棒性，降低了对伪标签质量的依赖。

## 局限与展望

1. **仅验证LiDAR编码器**：PointPillars和SECOND的异构性相对有限，未验证跨模态(LiDAR vs Camera)场景。
2. **仅模拟数据**：OPV2V基于CARLA模拟器，真实世界的噪声、通信延迟、定位误差等未充分考虑。
3. **固定k帧训练**：k值需要预设，缺乏自适应终止机制(如何判断适配器已收敛？)。
4. **单向适配**：当前设计只有ego端训练适配器，未考虑双向适配的可能。
5. **跨场景泛化性有限**：部分极端场景下跨场景测试性能下降明显。
6. **通信开销**：Stage I需要同时传输特征和检测结果，带宽需求翻倍(虽然只持续k帧)。

## 相关工作与启发

- **HEAL (Lu et al.)**：建立统一特征空间，新agent只需对齐到共享空间。性能最优但需全量训练——PHCP与之互补。
- **TFA (Wang et al.)**：两阶段微调范式，PHCP在此基础上改进为仅微调适配器。
- **CBAM (Woo et al.)**：通道+空间注意力模块，因轻量且适合少样本训练被选为适配器骨干。
- **F-Cooper/V2VNet/AttFusion**：经典协同感知方法，均假设同构模型——PHCP打破此假设。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次在推理阶段解决异构协同感知，问题定义有价值
- 实验充分度: ⭐⭐⭐⭐ 16场景+多消融，但仅OPV2V单数据集
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图示直观
- 价值: ⭐⭐⭐⭐ 实际部署意义大，但局限于模拟环境和LiDAR编码器

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] I Am Big, You Are Little; I Am Right, You Are Wrong](i_am_big_you_are_little_i_am_right_you_are_wrong.md)
- [\[NeurIPS 2025\] ResNets Are Deeper Than You Think](../../NeurIPS2025/others/resnets_are_deeper_than_you_think.md)
- [\[CVPR 2026\] Progressive Neural Architecture Generation](../../CVPR2026/others/progressive_neural_architecture_generation.md)
- [\[NeurIPS 2025\] Contextual Dynamic Pricing with Heterogeneous Buyers](../../NeurIPS2025/others/contextual_dynamic_pricing_with_heterogeneous_buyers.md)
- [\[CVPR 2025\] Regor: Progressive Correspondence Regenerator for Robust 3D Registration](../../CVPR2025/others/progressive_correspondence_regenerator_for_robust_3d_registration.md)

</div>

<!-- RELATED:END -->
