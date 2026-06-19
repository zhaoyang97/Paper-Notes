---
title: >-
  [论文解读] Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges
description: >-
  [NeurIPS 2025][图像生成][灵巧抓取迁移] 提出将跨手形态的视觉灵巧抓取迁移建模为 Schrödinger Bridge 问题，通过在潜空间中学习得分与流匹配（[SF]²M），并设计物理感知的最优传输代价函数（位姿/接触图/力旋量空间/雅可比可操作性），在无需配对数据的条件下实现不同机械手之间抓取意图的分布级迁移。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "灵巧抓取迁移"
  - "Schrödinger Bridge"
  - "最优传输"
  - "得分与流匹配"
  - "物理引导代价函数"
---

# Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges

**会议**: NeurIPS 2025  
**arXiv**: [2506.02489](https://arxiv.org/abs/2506.02489)  
**代码**: [grasp2grasp.github.io](https://grasp2grasp.github.io)  
**领域**: 机器人  
**关键词**: 灵巧抓取迁移, Schrödinger Bridge, 最优传输, 得分与流匹配, 物理引导代价函数  

## 一句话总结

提出将跨手形态的视觉灵巧抓取迁移建模为 Schrödinger Bridge 问题，通过在潜空间中学习得分与流匹配（[SF]²M），并设计物理感知的最优传输代价函数（位姿/接触图/力旋量空间/雅可比可操作性），在无需配对数据的条件下实现不同机械手之间抓取意图的分布级迁移。

## 研究背景与动机

灵巧多指机械手在抓取任务中拥有高自由度和灵活性，但其高维配置空间和复杂接触动力学使得数据驱动方法面临"手特定"瓶颈——为一种手训练的模型无法直接迁移到另一种手。现有抓取迁移方案存在以下痛点：

**遥操作/行为克隆方法**：直接复制关节角度或末端执行器位姿，当源手和目标手形态差异大时会产生物理不可行的抓取

**重定向方法**（如 Dex-Retargeting、CrossDex）：依赖人工标注的链接对应关系，限制了通用性，且是逐点映射而非分布级迁移

**RobotFingerPrint**：通过统一坐标空间做映射，但需要仿真预处理且与物体无关，忽略了功能性抓取意图

**缺乏配对数据**：获取源手→目标手的一一对应抓取数据极其困难

本文的核心动机是：**能否不需要配对数据，仅从各手的非配对抓取数据集出发，学习一个保持功能性抓取意图的分布级映射？**

## 方法详解

### 整体框架

方法采用两阶段流水线（如论文 Figure 3 所示）：

**阶段一：VAE 编码抓取观测**

- 使用 PVCNN 骨干的 VAE 将分割后的源手抓取视觉观测编码到潜空间 $z \in \mathcal{Z}$
- 解码器输出抓取配置 $\hat{g} = \text{Dec}(z)$，再通过可微运动学层 FK 得到手部 3D 网格顶点
- VAE 损失：$\mathcal{L}_{VAE} = \mathbb{E}[\|\hat{g} - g\|^2 + \alpha\|\hat{v} - \text{FK}(g)\|^2] + \beta \text{KL}(q_{\text{Enc}}(z|o) \| \mathcal{N}(0,I))$

**阶段二：潜空间 Schrödinger Bridge**

- 在 VAE 潜空间中建模源手分布 $q_0(z_{\text{source}}|o_{\text{obj}})$ 到目标手分布 $q_1(z_{\text{target}}|o_{\text{obj}})$ 的 Schrödinger Bridge
- 使用 U-ViT 作为骨干网络学习得分函数 $s_\theta(t,z)$ 和流场 $v_\theta(t,z)$
- 条件输入包括：物体点云（LION VAE 编码，1个全局token + 512个局部token）、手部观测潜变量（1个token）、5个接触锚点token

**推理流程**：源手观测 → VAE 编码 $z_0$ → 沿学到的 SDE 动力学演化 → 得到翻译后的潜码 $z_1$ → VAE 解码 → 目标手抓取配置

### 关键设计：物理感知 OT 代价函数

传统 Schrödinger Bridge 使用欧几里得距离作为传输代价，无法捕捉抓取的功能等价性。本文设计了四种物理感知代价：

**1. 基础位姿代价 $d_{\text{pose}}$**

$$d_{\text{pose}} = \|h_{\text{source}} - h_{\text{target}}\|_2^2 + \|R_{\text{source}} - R_{\text{target}}\|_F^2$$

平移部分用 L2 范数，旋转部分用 Frobenius 范数（旋转矩阵通过 6D 连续表示转换），鼓励工作空间中的粗对齐。

**2. 接触图相似度 $d_{\text{contact}}$**

$$d_{\text{contact}} = \text{Chamfer}(C_{\text{source}}, C_{\text{target}})$$

将接触图表示为 3D 点云,计算双向 Chamfer 距离，保持物体表面上的局部交互几何。

**3. 力旋量空间重叠 $d_{\text{wrench}}$**

$$d_{\text{wrench}} = 1 - \frac{\text{Vol}(\text{Hull}(GWS_{\text{source}}) \cap \text{Hull}(GWS_{\text{target}}))}{\text{Vol}(\text{Hull}(GWS_{\text{source}}) \cup \text{Hull}(GWS_{\text{target}}))}$$

基于接触位置和法线计算力旋量集合的凸包，用 IoU 衡量两个抓取的力学能力相似度。实际计算中降维到前 3 维（质心力），用 Monte Carlo 估计 IoU。

**4. 雅可比可操作性 $d_{\text{jac}}$**

$$d_{\text{jac}} = \|m_{\text{source}} - m_{\text{target}}\|^2$$

在可微仿真器 Warp 中执行抓取，计算物体位姿对关节角的雅可比矩阵，取各关节维度的最大值得到 6D 最大效应向量，鼓励翻译后的抓取保持相似的物体可控性。

### 损失函数

核心训练目标是 [SF]²M（Score and Flow Matching）损失：

$$\mathcal{L}_{[\text{SF}]^2\text{M}}(\theta) = \mathbb{E}\left[\|v_\theta(t,z) - u_t^\circ(z|z_0,z_1)\|^2 + \lambda(t)^2\|s_\theta(t,z) - \nabla\log p_t(z|z_0,z_1)\|^2\right]$$

其中 $(z_0, z_1) \sim \pi_\varepsilon^*$（通过 Sinkhorn 算法计算的熵正则化 OT 计划），$z \sim p_t(z|z_0,z_1)$（条件布朗桥采样）。采用小批量 OT 近似以提高计算效率。

## 训练与推理

### 训练流程

训练分为两个独立阶段，顺序执行：

**阶段一（VAE 训练）**：对每种手形态分别训练一个 PVCNN-based VAE。输入为分割后的手部点云观测 $o$，编码为潜变量 $z \sim \text{Enc}(o)$，解码为抓取配置 $\hat{g} = \text{Dec}(z)$，再经可微运动学层 FK 得到 3D mesh 顶点 $\hat{v} = \text{FK}(\hat{g})$。损失包含配置重建误差、mesh 顶点重建误差和 KL 正则化三项。训练完成后 encoder/decoder 参数冻结。

**阶段二（Schrödinger Bridge 训练）**：冻结 VAE 后，在潜空间中训练 U-ViT 网络学习 SB 动力学。每个 minibatch 的训练步骤为：
1. 从源手和目标手数据集中分别采样一批观测 $o_s, o_t$
2. 通过 VAE encoder 得到潜码 $z_s = \text{Enc}(o_s)$，$z_t = \text{Enc}(o_t)$
3. 使用 Sinkhorn 算法在当前 minibatch 上计算熵正则化 OT 计划 $\pi_\varepsilon^*$，代价矩阵由所选的物理感知代价函数决定
4. 从 $\pi_\varepsilon^*$ 中采样配对 $(z_0, z_1)$，在条件布朗桥上采样中间点 $z_t$
5. 用 [SF]²M 损失更新 $v_\theta$ 和 $s_\theta$

关键实现细节：使用 minibatch OT 避免精确 OT 的 $O(n^2)$ 计算开销；GWS 代价中将 6D 凸包降至前 3 维（质心力方向），用 Monte Carlo 估计 IoU；Jacobian 代价使用 Warp 可微仿真器在线计算。

### 推理流程

给定源手抓取的视觉观测 $(o_{\text{obj}}, o_{\text{hand}})$：
1. 通过源手 VAE encoder 得到初始潜码 $z_0 = \text{Enc}(o_{\text{hand}})$
2. 使用 Euler-Maruyama 积分沿学到的 SDE 动力学前向演化，步数可调（默认 100 步）
3. 得到翻译后的潜码 $z_1$
4. 通过目标手 VAE decoder 解码为目标手抓取配置 $g_{\text{target}} = \text{Dec}(z_1)$

推理时**无需 test-time finetuning**，整个过程是纯前向的摊销推理。条件信息（物体点云编码、接触锚点）在 U-ViT 的 cross-attention 中注入。

## 实验关键数据

### 主实验：抓取翻译质量（Table 1）

数据集：MultiGripperGrasp（3040万抓取，11种机械手，345个物体）。训练138物体，测试34个未见物体。

| 方法 | 成功率均值↑ | 多样性(rad)↑ | 6D GWH IoU↑ |
|------|-----------|-------------|-------------|
| CrossDex | 26.87% | 0.206 | 5.26% |
| Dex-Retargeting | 43.19% | 0.201 | 5.68% |
| RobotFingerPrint | 57.00% | 0.203 | 7.77% |
| Diffusion baseline | 60.83% | 0.271 | 9.58% |
| **Ours (pose)** | 64.32% | 0.250 | 10.68% |
| **Ours (contact)** | 62.75% | 0.272 | **14.54%** |
| **Ours (GWH)** | 66.34% | 0.266 | **14.63%** |
| **Ours (jacobian)** | **67.45%** | 0.258 | 10.88% |

关键发现：
- Jacobian 代价取得最高成功率（67.45%），比最强基线高 6.6 个百分点
- GWH 和 Contact 代价在功能对齐（GWH IoU）上大幅领先，约为基线的 1.5-2.5 倍
- 基于优化的方法（Dex-Retargeting/CrossDex）因忽略手掌碰撞导致大量物理无效抓取

### 消融实验

**扩散率 σ 的影响（Table 2，H→A 任务）：**

| σ | 成功率↑ | IoU↑ |
|---|--------|------|
| 0.01 | 77.16% | 16.72% |
| 0.1 | 74.78% | 15.89% |
| 1.0 | 64.39% | 13.83% |

较小的扩散率产生更精确的传输路径，但多样性略低。

**离散化步数的影响（Table 3，H→A 任务）：**

| 步数 | Ours 成功率 | Diffusion 成功率 | Ours IoU | Diffusion IoU |
|------|-----------|----------------|---------|--------------|
| 10 | 71.45% | 60.00% | 13.80% | 6.19% |
| 100 | 75.12% | 72.57% | 14.00% | 9.54% |
| 200 | 74.78% | 73.17% | 15.89% | 9.66% |

SB 方法在低步数下仍显著优于 Diffusion，体现了更稳定和样本高效的生成动力学。

### 物理功能对齐分析

6D GWH IoU 结果揭示了一个核心 trade-off：
- **Jacobian 代价**取得最高成功率（67.45%），说明可操作性度量最能捕捉稳定性所需的运动学关系
- **GWH 和 Contact 代价**在功能对齐指标上大幅领先（14.63% 和 14.54% vs 基线 9.58%），说明直接优化力学能力或接触模式更有效地传递了抓取的底层力学意图
- 这个 trade-off 意味着：稳定性（Jacobian）和功能等价性（GWH/Contact）之间存在张力，没有单一代价函数能同时最优化两者

### 效率对比

- RobotFingerPrint：>5秒/抓取（逐样本迭代优化）
- 本方法（100步）：~0.8秒/抓取（批量生成，摊销推理成本）

## 亮点与洞察

1. **问题建模创新**：首次将跨形态抓取迁移建模为 Schrödinger Bridge 问题，自然地处理了无配对数据的分布级映射
2. **物理感知代价设计**：四种代价函数各有侧重——Jacobian 最利于稳定性，GWH/Contact 最利于功能对齐，为不同应用场景提供选择
3. **无需手特定仿真预处理**：与 RFP 不同，本方法不需要为每种新手设计特定的坐标映射
4. **Simulation-free 训练**：避免了完整随机过程模拟，训练可扩展
5. **低步数鲁棒性**：即使 10 步也大幅超越 Diffusion 基线，部署友好

## 局限性

1. **薄壳物体性能下降**：接触区域模糊或极小的薄壳物体上表现不佳（如 Figure 4 右侧所示），有效抓取配置空间极度受限
2. **Shadow 手一致性偏低**：由于 Shadow 手更复杂的运动学和更大的工作空间，所有方法在 H→S 任务上性能均低于 H→A
3. **不支持推理时新手形态**：VAE 解码器隐式学习了目标手的运动学结构，面对未见手形态需要重新训练
4. **代价函数间的 trade-off**：稳定性最优（Jacobian）和功能对齐最优（GWH）不能同时达到，缺乏统一的开销函数
5. **评估仅在仿真中**：未展示真实机器人上的迁移效果

## 相关工作与启发

- **与 GenDexGrasp/UGG 的关系**：这些方法虽然也做多手抓取，但仍是手特定的生成模型，不做跨形态迁移
- **与 Schrödinger Bridge 文献的关系**：继承了 [SF]²M 框架的 simulation-free 训练思路，但创新在于将 OT 代价从通用几何距离替换为物理语义代价
- **启发**：物理感知的传输代价设计思路可推广到其他需要保持语义不变性的分布迁移任务（如跨域动作迁移、跨体态运动重定向）
- **潜在扩展**：可尝试组合多种代价函数（多目标 OT），或用条件化的代价权重实现任务自适应

## My Notes

### 方法论贡献评估
本文最有价值的贡献不在 Schrödinger Bridge 本身（框架直接沿用 [SF]²M），而在于**物理感知代价函数的设计**。在 OT/SB 框架中，ground cost 的选择决定了传输的语义意义——通用的欧几里得距离只能做几何对齐，无法保持功能意图。四种代价分别用位姿（粗对齐）、接触图（局部交互）、力旋量空间（力学能力）、雅可比（可控性）从不同物理维度编码抓取等价性，这种设计思路的可迁移性很强。

### 与现有方法的本质区别
传统重定向方法（Dex-Retargeting、CrossDex）是**逐点映射**——为每个源抓取找到一个目标抓取，且需要人工标注链接对应关系。本方法是**分布级映射**——学习源手抓取分布到目标手抓取分布的传输，自然地处理了一对多和多对一的情况，不需要显式的手部结构对应。

### 局限性的深层思考
最大局限在于 VAE decoder 隐式绑定了目标手的运动学，推理时不能直接迁移到未见手形态，与 "通用跨形态" 的终极目标仍有距离。一个可能的解法是将手形态作为条件输入 decoder，或使用 hypernetwork 根据手形态参数动态生成 decoder 权重。此外，四种代价的 trade-off 暗示可能存在 Pareto 最优解——多目标 OT 或加权组合是自然的下一步。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将 Schrödinger Bridge 引入抓取迁移是新颖的问题建模，物理感知代价设计也有原创性
- **实验充分度**: ⭐⭐⭐⭐ — 多种手形态组合、多指标评估、消融实验完整，但缺少真实机器人验证
- **写作质量**: ⭐⭐⭐⭐ — 数学框架清晰，从 SB 到具体应用的推导逻辑通顺
- **价值**: ⭐⭐⭐⭐ — 解决了实际痛点且方法通用，对机器人学习社区有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Schrödinger Bridge Matching for Tree-Structured Costs and Entropic Wasserstein Barycentres](schrödinger_bridge_matching_for_tree-structured_costs_and_entropic_wasserstein_b.md)
- [\[NeurIPS 2025\] Dynamic Diffusion Schrödinger Bridge in Astrophysical Observational Inversions](dynamic_diffusion_schrödinger_bridge_in_astrophysical_observational_inversions.md)
- [\[ICLR 2026\] Branched Schrödinger Bridge Matching](../../ICLR2026/image_generation/branched_schrödinger_bridge_matching.md)
- [\[ICML 2026\] Geometry-based Schrödinger Bridges for Trustworthy Multimodal Fusion](../../ICML2026/image_generation/geometry-based_schrödinger_bridges_for_trustworthy_multimodal_fusion.md)
- [\[NeurIPS 2025\] Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge](towards_general_modality_translation_with_contrastive_and_predictive_latent_diff.md)

</div>

<!-- RELATED:END -->
