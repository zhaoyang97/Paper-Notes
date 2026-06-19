---
title: >-
  [论文解读] Attention Gathers, MLPs Compose: A Causal Analysis of an Action-Outcome Circuit in VideoViT
description: >-
  [AAAI2026][可解释性][mechanistic interpretability] 通过机械可解释性方法逆向工程 Video Vision Transformer（ViViT）的内部电路，揭示注意力头负责"收集证据"、MLP 模块负责"组合概念"的分工机制，证明模型在简单分类任务中隐藏了超越训练目标的语义知识。
tags:
  - "AAAI2026"
  - "可解释性"
  - "mechanistic interpretability"
  - "Transformer"
  - "Activation Patching"
  - "Circuit Analysis"
  - "Trustworthy AI"
---

# Attention Gathers, MLPs Compose: A Causal Analysis of an Action-Outcome Circuit in VideoViT

**会议**: AAAI2026  
**arXiv**: [2603.11142](https://arxiv.org/abs/2603.11142)  
**代码**: 待确认  
**领域**: 可解释性  
**关键词**: mechanistic interpretability, Video Vision Transformer, Activation Patching, Circuit Analysis, Trustworthy AI

## 一句话总结

通过机械可解释性方法逆向工程 Video Vision Transformer（ViViT）的内部电路，揭示注意力头负责"收集证据"、MLP 模块负责"组合概念"的分工机制，证明模型在简单分类任务中隐藏了超越训练目标的语义知识。

## 背景与动机

视频视觉 Transformer（ViViT）在视频分类任务上取得了优异表现，但和其他深度模型一样面临"黑箱"问题。对于需要部署在高风险场景（自动驾驶、医疗等）的视频 AI 系统，理解其内部推理过程是建立信任的关键前提。

现有的可解释性工作大多集中在语言模型和图像模型，视频领域由于时空维度更高而研究较少。机械可解释性（Mechanistic Interpretability）旨在通过逆向工程将模型内部计算还原为人类可理解的算法，但在视频 Transformer 上的应用几乎是空白。

本文的核心动机是：一个仅训练用于分类人类动作（如"保龄球"）的 ViViT 模型，是否在内部隐藏了更细粒度的语义理解（如动作的成功/失败）？这种"隐藏认知"对 AI 安全和可信部署有何启示？

## 核心问题

1. 预训练的 ViViT 在完成相同分类任务（输出均为"bowling"）时，是否对"全中"（strike）和"洗沟"（gutter）两种不同结果产生了不同的内部表征？
2. 如果存在这种内部信号，模型架构中的哪些组件（Attention vs. MLP）分别承担了什么角色？
3. 这种内部表征电路的鲁棒性如何，是否能被简单的消融实验破坏？

## 方法详解

### 实验设置

- **模型**：google/vivit-b-16x2-kinetics400，12 层 ViViT-B，使用 16×16 空间 + 2 帧时间的 tubelet embedding
- **数据**：从 Kinetics-400 "bowling" 类别中构造的最小对比对（contrastive pair）——一个"全中"视频和一个"洗沟"视频，模型对两者均正确分类为 bowling（Label 31）
- **固定随机种子**：42，确保实验可复现

### 观察性分析

1. **Direct Logit Attribution（DLA）**：分析 [CLS] token 在各层对最终分类 logit 的贡献，发现从 Layer 9 开始模型置信度显著增加
2. **Token-wise Heatmap**：可视化时空 token 对输出类别的贡献，发现主要集中在球和球瓶交互区域
3. **CLS Token Attention 可视化**：Layer 10 Head 8 作为语义"结果检测器"，在 strike 视频中追踪球的轨迹和撞击瞬间，在 gutter 视频中关注球沟和未被击倒的球瓶
4. **线性探针（Linear Probe）**：在所有 12 层上训练逻辑回归分类器区分 strike 和 gutter 的 [CLS] token 激活——结果从 Layer 0 就达到 100% 准确率，说明探针仅捕获了表面差异（"指纹扫描"），而非语义概念

### 信号定位：Delta Analysis

利用两段视频的激活差值定位内部信号：

$$\Delta = act_{strike} - act_{gutter}$$

计算每层 delta 的 L2 范数作为"信号强度"。结果显示从 Layer 5 到 Layer 11，L2 范数增长超过 300%（约 75 → 250+），呈现清晰的"信号放大级联"（amplification cascade）。与线性探针从 Layer 0 就检测到差异不同，delta analysis 显示语义信号在中深层才逐渐形成，表明模型计算的是高层语义抽象而非低层特征差异。

### 因果分析

1. **成分消融（Component Ablation）**：使用 DLA 识别贡献最大的前 10% token（313 个 patch），将其置零。结果：strike 视频的 bowling logit 仅下降 0.34（16.99→16.66），gutter 视频仅下降 0.02（16.52→16.50），分类几乎不受影响。这说明分类电路是高度分布式的，且"结果信号"电路独立于分类电路运作。

2. **Activation Patching**：将 strike 运行中的单个组件（Attention 或 MLP）激活替换到 gutter 运行中，测量在 Layer 11 恢复了多少"成功 vs. 失败"信号。信号恢复率计算如下：

$$\text{Recovery}(\%) = \frac{\|\Delta_{patch}\|}{\|\Delta_{strike}\|} \cdot \text{sign}(\Delta_{patch} \cdot \Delta_{strike}) \times 100$$

## 实验关键数据

### Activation Patching 结果（Layer 4-10）

| 层 | 组件 | 信号恢复率 |
|---|---|---|
| Layer 4 | Attention | 54.41% |
| Layer 4 | MLP | **60.17%** |
| Layer 5 | Attention | 50.22% |
| Layer 5 | MLP | **57.49%** |
| Layer 6 | Attention | 43.62% |
| Layer 6 | MLP | **49.11%** |
| Layer 7 | Attention | 40.38% |
| Layer 7 | MLP | **42.55%** |
| Layer 8 | Attention | 37.72% |
| Layer 8 | MLP | **42.10%** |
| Layer 9 | Attention | 44.43% |
| Layer 9 | MLP | **58.66%** |
| Layer 10 | Attention | 47.61% |
| Layer 10 | MLP | 43.39% |

### 关键发现

- **注意力头**恢复 37-54% 的信号，角色为"证据收集者"（Evidence Gatherers）
- **MLP 模块**恢复 42-60% 的信号，角色为"概念组合者"（Concept Composers），是生成"成功"信号的主要驱动力
- 没有单一组件能恢复 100% 信号，证明电路是**分布式和冗余的**
- 消融实验中分类几乎不受影响（logit 变化 < 0.34），验证了电路的鲁棒性

## 亮点

1. **首次在视频 Transformer 上进行系统性的机械可解释性分析**，将 MechInterp 从语言/图像模型拓展到视频领域
2. **揭示了清晰的分工模式**："Attention Gathers, MLPs Compose"——注意力负责聚合时空证据，MLP 负责组合语义概念，支持了 Transformer 内部功能分化的假说
3. **发现了"隐藏认知"现象**：模型仅训练用于分类"bowling"，却自发发展出区分动作结果的内部表征，这对 AI 安全具有重要警示意义
4. **方法论贡献**：展示了 delta analysis + activation patching 的组合方法，从信号定位到因果归因的完整分析流程
5. **线性探针的失败案例**分析很有教育意义——100% 准确率反而说明探针在捕获表面特征，强调了因果干预方法的必要性

## 局限与展望

- **样本规模极小**：仅使用一对对比视频（strike vs. gutter），无法确认发现的电路是否推广到更多样本或更多动作类别
- **单一架构**：仅在 ViViT-B 上验证，未测试 TimeSformer 等其他视频 Transformer
- **无法排除特征特异性**：发现的电路可能部分依赖于特定视频对的背景纹理等低层特征，而非纯粹的语义概念
- **缺乏与标准可解释性方法的定量对比**：未与 Integrated Gradients、CAV 等基线进行系统性比较
- 未来方向包括使用 Automated Circuit Discovery（ACDC）在大规模数据上验证，以及跨架构泛化实验

## 与相关工作的对比

| 方面 | 本文 | 传统可解释性方法 |
|---|---|---|
| 分析粒度 | 组件级因果分析（Attention vs. MLP） | 输入特征归因（梯度热力图） |
| 方法类型 | 因果干预（activation patching） | 观察性（saliency maps, IG） |
| 适用域 | 视频 Transformer | 主要在语言/图像模型 |
| 发现能力 | 能区分功能角色（gather vs. compose） | 只能指出"哪些输入重要" |

与 Eliciting Latent Knowledge（ELK）方向的工作（Burns et al. 2022, Mallen et al. 2023）相呼应：本文在视频域提供了模型隐藏知识的实证，而 ELK 主要在语言模型中探索。线性探针在 Layer 0 即达 100% 的失败案例也与 Mallen et al. 的发现一致——简单探针可能捕获的是浅层特征而非真正的隐藏知识。

## 启发与关联

- **对 AI 安全的启示**：即使是简单任务训练的模型也可能发展出超越训练目标的内部表征，标准的输出监控无法发现这些"隐藏认知"，需要机械可解释性工具进行深层审查
- **MLP 的冗余级联机制**使得简单的安全干预（如移除单个"有害"组件）可能无效，需要更精细的干预策略
- **分析框架可迁移**：delta analysis + activation patching 的方法论可以应用于其他视频理解任务和架构
- **与 Transformer 可解释性文献的衔接**：在语言模型中已有类似发现（attention 做信息路由，MLP 做知识存储/组合），本文在视频域提供了进一步的支持证据

## 评分

- 新颖性: ⭐⭐⭐⭐ （首次将完整的 MechInterp 流程应用于视频 Transformer，研究问题新颖）
- 实验充分度: ⭐⭐⭐ （方法论完整但样本量过小，仅一对对比视频，泛化性不足）
- 写作质量: ⭐⭐⭐⭐ （逻辑清晰，从观察到因果的叙述连贯，图表设计合理）
- 价值: ⭐⭐⭐⭐ （对 AI 安全和可信部署有重要启示，方法论有迁移价值，但需要大规模验证）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](../../NeurIPS2025/interpretability/causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)
- [\[CVPR 2026\] CIGMA: Causal Information-Gain Mechanistic Attribution of Attention Heads in Vision Transformers](../../CVPR2026/interpretability/cigma_causal_information-gain_mechanistic_attribution_of_attention_heads_in_visi.md)
- [\[NeurIPS 2025\] Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning](../../NeurIPS2025/interpretability/learning_to_focus_causal_attention_distillation_via_gradient-guided_token_prunin.md)
- [\[ICML 2026\] Where Computation Lives Inside TabPFN: Causal Localisation of Attention Head Function](../../ICML2026/interpretability/where_computation_lives_inside_tabpfn_causal_localisation_of_attention_head_func.md)
- [\[AAAI 2026\] Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning](attention_as_binding_a_vector-symbolic_perspective_on_transformer_reasoning.md)

</div>

<!-- RELATED:END -->
