---
title: >-
  [论文解读] PromptMoE: Generalizable Zero-Shot Anomaly Detection via Visually-Guided Prompt Mixing of Experts
description: >-
  [AAAI 2026][目标检测][零样本异常检测] PromptMoE 将提示学习从单体式（monolithic）范式转变为组合式（compositional）范式，通过视觉引导的混合专家（MoE）机制从可学习的语义原语库中动态组合实例自适应的正常/异常状态提示，在 15 个工业和医学数据集上实现 ZSAD SOTA。
tags:
  - "AAAI 2026"
  - "目标检测"
  - "零样本异常检测"
  - "CLIP"
  - "混合专家"
  - "组合式提示学习"
  - "视觉引导路由"
---

# PromptMoE: Generalizable Zero-Shot Anomaly Detection via Visually-Guided Prompt Mixing of Experts

**会议**: AAAI 2026  
**arXiv**: [2511.18116](https://arxiv.org/abs/2511.18116)  
**代码**: 无  
**领域**: 异常检测 / 视觉-语言模型  
**关键词**: 零样本异常检测, CLIP, 混合专家, 组合式提示学习, 视觉引导路由

## 一句话总结

PromptMoE 将提示学习从单体式（monolithic）范式转变为组合式（compositional）范式，通过视觉引导的混合专家（MoE）机制从可学习的语义原语库中动态组合实例自适应的正常/异常状态提示，在 15 个工业和医学数据集上实现 ZSAD SOTA。

## 研究背景与动机

零样本异常检测（ZSAD）旨在检测和定位训练阶段未见过的物体类别中的异常区域，在工业制造质量检测和医学诊断中具有关键应用价值。基于 CLIP 等视觉-语言模型的方法展现了潜力，但现有提示工程策略面临严重限制：

**单提示表示瓶颈**：无论是手工设计的固定提示（WinCLIP）还是学习的单一正常/异常提示对（AnomalyCLIP），一个固定的提示向量难以捕捉未见类别中的多样化正常和异常模式

**静态多提示过拟合**：简单增加可学习静态提示的数量会显著增加对辅助数据的过拟合风险——模型倾向于记忆训练集中特定的模式-提示组合，而非学习可推广的抽象概念

**单一动态映射的局限**：CoCoOp 式方法（AdaCLIP、VCP-CLIP）用单一映射网络从视觉实例动态生成提示，难以为特定的细粒度异常模式生成专门化的提示

**深层提示的泛化失败**：AnomalyCLIP 已经证明，当静态可学习提示插入文本编码器中间层时，仅在浅层有效，深层反而损害未见类别的零样本检测

核心洞察：**鲁棒的 ZSAD 需要组合式而非单体式的提示学习**——"学习如何组合"比"学习一个完整提示"更有效。

## 方法详解

### 整体框架

PromptMoE 基于冻结的 CLIP（ViT-L/14@336px）构建，核心创新是 **Visually-Guided Mixture of Prompt (VGMoP)** 模块。如 Figure 3 所示：

1. 视觉编码器从输入图像提取多层 patch 特征 $\mathbf{F}_x^{(l)}$ 和全局特征 $\mathbf{F}_x^{cls}$
2. VGMoP 以视觉特征为输入，动态生成实例特定的正常和异常文本提示 $\mathbf{T}_n^{(l)}$ 和 $\mathbf{T}_a^{(l)}$
3. 聚合文本嵌入与 patch 特征的逐层相似度，生成异常图 $\mathbf{M}$

### 关键设计

#### 1. 混合文本提示结构

为每个视觉实例构造两个混合文本提示：

$$\mathbf{T}_n = [\mathbf{S}_{\text{agg}}^n][\texttt{cls}][\mathbf{Q}_{\text{ctx}}]$$

$$\mathbf{T}_a = [\mathbf{S}_{\text{agg}}^n][\mathbf{S}_{\text{agg}}^a][\texttt{cls}][\mathbf{Q}_{\text{ctx}}]$$

其中：
- $\mathbf{S}_{\text{agg}}^n \in \mathbb{R}^{M_n \times D}$：聚合的正常状态提示（$M_n=5$）
- $\mathbf{S}_{\text{agg}}^a \in \mathbb{R}^{M_a \times D}$：聚合的异常状态后缀（$M_a=6$）
- $\mathbf{Q}_{\text{ctx}} \in \mathbb{R}^{M_q \times D}$：共享的可学习上下文 token（$M_q=8$）
- $[\texttt{cls}]$：类别名或通用占位符 "object" 的嵌入

异常提示在正常提示基础上追加异常后缀 $\mathbf{S}_{\text{agg}}^a$，设计灵感来自 PromptAD。

#### 2. 视觉引导的 MoE 状态聚合（VGMoP 核心）

对每一层 $l \in \mathcal{I}$，独立执行以下流程（见 Figure 4）：

**跨注意力视觉蒸馏**：可学习的状态查询 $\mathbf{q}^{(l)}$ 通过跨注意力主动从 patch 特征中蒸馏状态相关的视觉信号：

$$\mathbf{O}^{(l)} = \text{Softmax}\left(\frac{Q^{(l)}{K^{(l)}}^\top}{\sqrt{D}}\right)V^{(l)}$$

对 $\mathbf{O}^{(l)}$ 平均池化得到路由表示 $\mathbf{r}^{(l)} = \text{mean}(\mathbf{O}^{(l)})$。

**稀疏路由**：$\mathbf{r}^{(l)}$ 输入层特定的图像门控稀疏路由器 $G^{(l)}$（两层 MLP: Linear-ReLU-Linear），生成对专家提示池 $\mathcal{E}^{(l)} = \{\mathbf{s}_j^{(l)} \in \mathbb{R}^{M \times D}\}_{j=1}^E$ 的路由 logits。选择 top-$k$ 专家并加权聚合：

$$\mathbf{S}_{\text{agg}}^{(l)} = \sum_{i=1}^{k} \mathbf{w}_i^{(l)} \mathbf{s}_{\text{top},i}^{(l)}, \quad \mathbf{w}^{(l)} = \text{Softmax}(\mathbf{z}_{\text{top}}^{(l)})$$

正常和异常分别使用**独立的专家池** $\mathcal{E}_n$ 和 $\mathcal{E}_a$，避免负迁移。

#### 3. 辅助损失

**负载均衡损失** $\mathcal{L}_{\text{balance}}$：鼓励所有专家在 batch 上均匀贡献，防止路由退化为固定组合：

$$\mathcal{L}_{\text{balance}} = \alpha \sum_{l \in \mathcal{I}} \left(E \sum_{j=1}^{E} \left(\frac{1}{B}\sum_{i=1}^{B} \mathbf{p}_{i,j}^{(l)}\right)^2\right)$$

**专家解耦损失** $\mathcal{L}_{\text{decouple}}$：促进专家池内表示多样性，通过正交约束：

$$\mathcal{L}_{\text{decouple}} = \beta \sum_{l \in \mathcal{I}} \|\hat{\mathbf{S}}^{(l)}(\hat{\mathbf{S}}^{(l)})^T - \mathbf{I}_E\|_F^2$$

两个损失协同工作：$\mathcal{L}_{\text{decouple}}$ 确保专家多样性是负载均衡有效工作的前提（Figure 7）。

### 损失函数 / 训练策略

总损失由分类、分割和辅助三部分组成：

$$\mathcal{L}_{\text{total}} = \underbrace{\text{BCE}(s, c)}_{\text{分类}} + \underbrace{\text{Dice}(\mathbf{M}, \mathbf{m}) + \text{Focal}(\mathbf{M}, \mathbf{m})}_{\text{分割}} + \underbrace{\mathcal{L}_{\text{balance}} + \mathcal{L}_{\text{decouple}}}_{\text{辅助}}$$

异常分数结合峰值和全局相似度：$s = \frac{1}{2}(\max(\mathbf{M}) + \text{Softmax}(\mathbf{F}_x^{cls} \mathbf{F}_T^{(\max(\mathcal{I}))\top}/\tau'))$

训练设置：
- CLIP 完全冻结，仅训练 VGMoP 模块
- 图像 resize 至 518×518，提取第 {6, 12, 18, 24} 层特征
- 15 epochs，Adam 优化器，lr=0.001，前 3 epoch warmup
- $E=8$ 专家，top-$k=4$，$\alpha=0.01, \beta=0.005$

## 实验关键数据

### 主实验

在 15 个数据集上的综合评估（7 工业 + 8 医学），主要在 MVTec AD 上训练，零样本推理其余 14 个数据集。

| 数据集 (领域) | 指标 | PromptMoE | 之前SOTA | 提升 |
|--------------|------|-----------|----------|------|
| MVTec AD (工业) | I-AUROC | 93.8 | 92.0 (AdaCLIP) | +1.8 |
| VisA (工业) | I-AUROC | 85.0 | 84.5 (FAPrompt) | +0.5 |
| BTAD (工业) | I-AUROC | 93.4 | 92.0 (FAPrompt) | +1.4 |
| SDD (工业) | P-AUROC | 98.1 | 98.3 (FAPrompt) | -0.2 |
| HeadCT (医学) | I-AUROC | 98.2 | 94.8 (FAPrompt) | +3.4 |
| HeadCT (医学) | AP | 98.2 | 93.5 (FAPrompt) | +4.7 |
| 工业平均 | I-(AUC,AP) | (92.4, 93.4) | (91.7, 92.5) | +0.7/+0.9 |
| 工业平均 | P-(AUC,PRO) | (96.2, 89.2) | (96.2, 88.0) | 0/+1.2 |
| 医学 I 级平均 | (AUC,AP) | (97.4, 97.5) | (96.0, 95.5) | +1.4/+2.0 |

在工业和医学两个域均取得 SOTA，尤其在未见域的医学数据集上泛化效果出色。

### 消融实验

| 配置 | MVTec I-AUC | MVTec PRO | VisA I-AUC | VisA PRO |
|------|-------------|-----------|------------|----------|
| Static Prompt（基线） | 91.7 | 82.0 | 82.4 | 88.0 |
| +Static Ensemble | 92.2 | 82.9 | 83.3 | 88.3 |
| +VGMoP（单层，无辅助损失） | 93.1 | 83.3 | 84.1 | 89.0 |
| **PromptMoE（完整）** | **93.8** | **83.2** | **85.0** | **89.2** |

**其他消融结论**：
- $\alpha=0$（去掉负载均衡）→ MVTec I-AUC 降至 92.1，证明负载均衡不可或缺
- 共享专家池 → I-AUC 降至 91.4，证明正常/异常分离设计的必要性
- 共享两者（池+跨注意力）→ 降至 90.9，负迁移最严重
- 多层特征（{6,12,18,24}）优于仅用最后一层

### 关键发现

1. **动态组合 >> 静态集成**：VGMoP 比 Static Ensemble 提升显著（MVTec +0.9 I-AUC），证明核心优势来自视觉引导的动态组合而非简单增加提示数量

2. **正常/异常专家路由模式截然不同**（Figure 6）：
    - 正常状态：路由一致收敛到少数核心专家，表明模型学到了可泛化的"正常性"语义原语
    - 异常状态：路由高度动态且稀疏，不同数据集激活不同专家子集，体现灵活的异常组合能力

3. **两个辅助损失的协同效应**（Figure 7）：去掉 $\mathcal{L}_{\text{decouple}}$ 后专家表示退化为冗余，进而导致 $\mathcal{L}_{\text{balance}}$ 恶化，最终损害检测性能——专家多样性是负载均衡的前提

4. **从工业训练到医学推理的跨域泛化**：仅在 MVTec AD 训练，在 HeadCT 上 I-AUROC 达 98.2%，超过次优方法 3.4%，验证了组合提示学习概念级泛化的成功

## 亮点与洞察

1. **从单体到组合的范式转变**：将 MoE 从传统的"选择 MLP 层处理 token"重新定义为"选择语义原语构建提示"，是 MoE 在提示工程中的创新应用
2. **查询驱动的视觉蒸馏**：不直接用平均池化压缩视觉特征（会丢失关键局部信息），而是用可学习查询通过跨注意力主动蒸馏状态相关信号
3. **正常/异常路由模式的可解释性**：正常状态的稳定路由 vs 异常状态的动态路由提供了直觉上的合理性和可解释性
4. **实现简洁**：CLIP 完全冻结，仅训练轻量的 VGMoP 模块，在 RTX 3090 上即可训练

## 局限与展望

1. 虽然代码声称可用，但原论文中未提供 GitHub 链接，复现可能存在障碍
2. 训练仅在 MVTec AD 上进行（在 MVTec AD 本身评估时用 VisA 训练），辅助训练集的选择对性能有多大影响未充分讨论
3. MoE 中 $E=8, k=4$ 的超参数选择较为保守，Table 6 显示 $E=16$ 在 PRO 上有提升但 I-AUC 不一定更好，最优配置可能因场景而异
4. 在 SDD 和部分医学 P 级指标上未超过 FAPrompt，说明组合提示在某些特定异常模式上可能不如密集动态提示
5. 仅使用 ViT-L/14@336px，未探索更大 backbone（如 ViT-G）或更新的 VLM（如 SigLIP）的效果

## 相关工作与启发

- **AnomalyCLIP**：提示学习应用于 ZSAD 的开创性工作，但受限于静态提示的泛化失败
- **AdaCLIP / VCP-CLIP**：动态提示生成的先行者，但单一映射网络限制了专门化能力
- **Switch Transformer / MoE**：MoE 的稀疏激活机制天然适合缓解过拟合，本文将其巧妙迁移到提示空间
- 启发：在提示学习中，"组合性"（compositionality）比"数量"和"动态性"更重要——学会组合少量高质量原语比学习大量固定提示或依赖单一动态映射更有效

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 5 | MoE 作为提示组合器的范式创新，正常/异常路由模式的洞察深刻 |
| 技术深度 | 4 | VGMoP 设计精巧，辅助损失协同分析充分 |
| 实验充分性 | 5 | 15 个数据集、多维度消融、专家激活分析全面 |
| 实用价值 | 4 | CLIP 冻结+轻量模块，部署友好；但训练集选择影响待调查 |
| 写作质量 | 4 | 范式对比图（Figure 1）直观，整体叙事清晰 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection](../../CVPR2026/object_detection/moeclip_patch-specialized_experts_for_zero-shot_anomaly_detection.md)
- [\[CVPR 2026\] FB-CLIP: Fine-Grained Zero-Shot Anomaly Detection with Foreground-Background Disentanglement](../../CVPR2026/object_detection/fb-clip_fine-grained_zero-shot_anomaly_detection_with_foreground-background_dise.md)
- [\[CVPR 2026\] GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning](../../CVPR2026/object_detection/gs-clip_zero-shot_3d_anomaly_detection_by_geometry-aware_prompt_and_synergistic_.md)
- [\[CVPR 2026\] DLVP-CLIP: Enhancing Fine-Grained Zero-Shot Anomaly Detection via Dynamic Local Visual Prompting](../../CVPR2026/object_detection/dlvp-clip_enhancing_fine-grained_zero-shot_anomaly_detection_via_dynamic_local_v.md)
- [\[CVPR 2026\] VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](../../CVPR2026/object_detection/visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)

</div>

<!-- RELATED:END -->
