---
title: >-
  [论文解读] Compositional Flows for 3D Molecule and Synthesis Pathway Co-design
description: >-
  [ICML 2025][计算生物][组合生成流] 提出 CGFlow（Compositional Generative Flows）——将 flow matching 扩展到组合对象的逐步生成，交织组合结构采样（合成路径）和连续状态传输（3D 构象），作为 3DSynthFlow 应用于可合成药物设计，在 LIT-PCBA 15个靶标上首次同时达到结合亲和力和可合成性的 SOTA。
tags:
  - "ICML 2025"
  - "计算生物"
  - "组合生成流"
  - "3D分子设计"
  - "合成路径"
  - "GFlowNet"
  - "药物设计"
---

# Compositional Flows for 3D Molecule and Synthesis Pathway Co-design

**会议**: ICML 2025  
**arXiv**: [2504.08051](https://arxiv.org/abs/2504.08051)  
**代码**: 无  
**领域**: 图像生成/分子设计  
**关键词**: 组合生成流, 3D分子设计, 合成路径, GFlowNet, 药物设计

## 一句话总结
提出 CGFlow（Compositional Generative Flows）——将 flow matching 扩展到组合对象的逐步生成，交织组合结构采样（合成路径）和连续状态传输（3D 构象），作为 3DSynthFlow 应用于可合成药物设计，在 LIT-PCBA 15个靶标上首次同时达到结合亲和力和可合成性的 SOTA。

## 研究背景与动机

**领域现状**：3D 分子生成模型（扩散/流匹配）在药物设计中表现出色，但一次性生成整个分子，无法保证可合成性。GFlowNet 可以按合成步骤构建分子但限于 2D。

**现有痛点**：
   - 扩散/流模型一次生成所有原子——无法 mask 无效生成动作，无法保证合成可行性
   - GFlowNet 按步构建但仅处理离散 2D 图——不建模 3D 构象（蛋白质-配体相互作用依赖 3D）
   - 自回归模型按步生成 3D 但早期步骤的微小误差会级联放大

**核心矛盾**：需要同时建模组合结构（合成路径的离散序列）和连续状态（3D 原子坐标），但现有方法只能处理其中一个。

**本文目标**：统一框架下联合生成合成路径和 3D 构象。

**切入角度**：将流匹配的插值过程扩展到组合状态转换——在组合步骤中逐步构建结构的同时，用流匹配传输对应的连续状态。

**核心 idea**：两个交织的流过程——(1) Compositional Flow 逐步拆解/构建组合结构; (2) State Flow 传输与各组件关联的连续状态。两者通过共享输入实现相互依赖。

## 方法详解

### 整体框架
CGFlow 由两个交织过程组成：
1. **Compositional Flow**：从完整分子向空状态渐进拆解（训练时正向），从空状态逐步构建（推理时反向）。每步对应一个合成反应步骤。
2. **State Flow**：标准最优传输 flow matching，但对不同组件采用不同的噪声水平——先被拆除的组件获得更高噪声。
推理时：GFlowNet 策略采样组合步骤（选择下一个合成反应）→ 条件流匹配生成对应的 3D 坐标。

### 关键设计

1. **组合流匹配插值的扩展**:

    - 功能：将标准流匹配的线性插值从"全量噪声→全量数据"扩展为"空结构→逐步构建完整结构"
    - 核心思路：在时间 $t_k$ 处执行第 $k$ 步合成反应，添加新的原子/片段；State Flow 对新添加的片段从噪声开始传输，对已有片段继续精化
    - 关键公式：$x_t = \alpha_t(c) \cdot x_1 + \sigma_t(c) \cdot \epsilon$，其中噪声级别 $\sigma_t(c)$ 依赖于组件 $c$ 被添加的时间
    - 设计动机：先添加的组件获得更多去噪时间→位置更精确；后添加的组件依赖先前组件的位置→自然的因果依赖

2. **GFlowNet 奖励引导采样**:

    - 功能：用 GFlowNet 按奖励比例采样合成路径（偏好高结合亲和力+高可合成性的路径）
    - 核心思路：$p(\text{pathway}) \propto R(\text{molecule})$，其中 $R$ 可以是结合分数、可合成性评分等
    - 设计动机：标准流匹配只能从训练分布采样，GFlowNet 支持奖励引导的探索——生成偏向高价值区域的分子

3. **3DSynthFlow 实例化**:

    - 功能：将 CGFlow 应用于可合成的靶向药物设计
    - 核心思路：合成路径由反应步骤序列定义（使用 Reaxys 反应模板），3D 构象在蛋白质口袋中生成
    - 训练数据：CrossDocked2020 + ZINC 合成路径
    - 关键创新：首个同时优化结合亲和力和可合成性的 3D 分子生成模型

### 损失函数 / 训练策略
- State Flow: 条件流匹配损失（CFM objective）
- Compositional Flow: GFlowNet 的轨迹平衡损失（trajectory balance）
- 两个损失联合训练
- 推理：交替执行合成步骤采样（GFlowNet）和 3D 坐标生成（flow matching ODE）

## 实验关键数据

### 主实验
LIT-PCBA 基准（15 个药物靶标）：

| 方法 | 平均 Vina Dock ↓ | 命中率 ↑ | AiZynth 可合成率 ↑ |
|------|----------------|---------|-----------------|
| TargetDiff | -7.84 | 12.3% | 18.5% |
| DiffSBDD | -8.21 | 15.7% | 22.3% |
| FlowSBDD | -8.56 | 18.2% | 28.1% |
| SynFlowNet (2D) | -7.12 | 8.5% | 42.3% |
| **3DSynthFlow** | **-9.42** | **24.5%** | **36.1%** |

### 采样效率

| 方法 | 找到高亲和力分子所需采样数 ↓ |
|------|----------------------|
| SynFlowNet (2D) | ~5000 |
| TargetDiff | ~3000 |
| **3DSynthFlow** | **~1200** (4.2× 加速) |

### 消融实验

| 配置 | Vina Dock | 可合成率 | 说明 |
|------|----------|---------|------|
| 仅 State Flow（无合成约束） | -8.56 | 22.3% | 退化为标准流匹配 |
| 仅 Compositional Flow（2D） | -7.12 | 42.3% | 无 3D 信息 |
| **CGFlow（两者交织）** | **-9.42** | **36.1%** | 最优平衡 |
| 无 GFlowNet（均匀采样路径） | -8.15 | 35.8% | 缺少奖励引导 |
| **+ GFlowNet 引导** | **-9.42** | **36.1%** | 偏好高价值分子 |

### 关键发现
- 3DSynthFlow 是首个在 Vina Dock(-9.42) 和 AiZynth(36.1%) 上同时达到 SOTA 的方法
- 采样效率提升 4.2×——GFlowNet 的奖励引导使搜索更聚焦
- 组合结构和连续状态的交织建模比独立建模显著更优（-9.42 vs -8.56/-7.12）
- 可合成率从纯 3D 方法的 ~22% 提升到 36%——合成路径约束有效
- 在所有 15 个 LIT-PCBA 靶标上一致优于现有方法——泛化性强

## 亮点与洞察
- **组合流 = 流匹配 × GFlowNet 的完美融合**——前者处理连续坐标，后者处理离散合成路径，通过时间轴上的交织实现相互依赖
- 不同组件不同噪声级别的设计极其自然——后添加的片段应该更"不确定"，因为它们依赖先前片段的位置
- 3D + 可合成性的联合优化对真实药物研发有直接价值——之前只能分别优化
- GFlowNet 的奖励引导使模型不仅学习数据分布，还偏向高价值区域——比纯似然学习更适合设计优化
- 框架通用性强——CGFlow 不限于分子设计，适用于任何组合对象+连续特征的生成

## 局限与展望
- 反应模板库有限——不在模板中的合成路径无法生成
- GFlowNet 训练复杂度随反应空间增长
- 3D 坐标预测的精度受限于 flow matching 的去噪质量
- Vina Dock 评分是近似的——与实际结合亲和力可能有偏差
- 未做 wet-lab 实验验证

## 相关工作与启发
- **vs TargetDiff/DiffSBDD**: 纯 3D 扩散模型，不保证可合成性
- **vs SynFlowNet**: 仅 2D 合成路径，不建模蛋白质-配体 3D 交互
- **vs AutoGrow**: 基于对接的碎片生长，不是端到端生成
- **启发**：组合+连续的联合生成范式可推广到其他科学设计问题（如新材料设计、蛋白质工程）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 组合流匹配的新范式具有广泛通用性
- 实验充分度: ⭐⭐⭐⭐⭐ 15靶标LIT-PCBA + CrossDocked + 效率分析
- 写作质量: ⭐⭐⭐⭐⭐ 框架图直观，数学清晰
- 价值: ⭐⭐⭐⭐⭐ 对计算药物设计有重大推进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment](scalable_non-equivariant_3d_molecule_generation_via_rotational_alignment.md)
- [\[ICML 2025\] WGFormer: An SE(3)-Transformer Driven by Wasserstein Gradient Flows for Molecular Generation](wgformer_an_se3-transformer_driven_by_wasserstein_gradient_flows_for_molecular_g.md)
- [\[ICML 2025\] UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design](unimomo_unified_generative_modeling_of_3d_molecules_for_de_novo_binder_design.md)
- [\[ICML 2026\] Demystifying Multimodal Biomolecular Co-design with Intrinsic Geodesic Coupling](../../ICML2026/computational_biology/demystifying_multimodal_biomolecular_co-design_with_intrinsic_geodesic_coupling.md)
- [\[NeurIPS 2025\] Amortized Sampling with Transferable Normalizing Flows](../../NeurIPS2025/computational_biology/amortized_sampling_with_transferable_normalizing_flows.md)

</div>

<!-- RELATED:END -->
