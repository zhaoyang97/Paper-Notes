---
title: >-
  [论文解读] JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models
description: >-
  [NeurIPS 2025][多模态VLM][jailbreak] 受 Eliciting Latent Knowledge (ELK) 框架启发，首次揭示 VLM 在 fusion layer 潜空间中存在可近似的安全决策边界，提出 JailBound 两阶段攻击框架（Safety Boundary Probing + Safety Boundary Crossing），通过联合优化图像和文本对抗扰动跨越该边界，在白盒和黑盒场景分别达到 94.32% 和 67.28% 平均攻击成功率，显著超越 SOTA。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "jailbreak"
  - "safety boundary"
  - "latent space attack"
  - "ELK"
  - "跨模态"
---

# JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.19610](https://arxiv.org/abs/2505.19610)  
**代码**: 待确认  
**领域**: 多模态VLM / AI安全 / 对抗攻击  
**关键词**: jailbreak, safety boundary, latent space attack, ELK, cross-modal perturbation

## 一句话总结
受 Eliciting Latent Knowledge (ELK) 框架启发，首次揭示 VLM 在 fusion layer 潜空间中存在可近似的安全决策边界，提出 JailBound 两阶段攻击框架（Safety Boundary Probing + Safety Boundary Crossing），通过联合优化图像和文本对抗扰动跨越该边界，在白盒和黑盒场景分别达到 94.32% 和 67.28% 平均攻击成功率，显著超越 SOTA。

## 研究背景与动机

**领域现状**：VLM 通过集成视觉编码器与 LLM 获得强大多模态能力，但视觉模态的引入显著扩大了攻击面。现有 jailbreak 攻击方法包括基于梯度的白盒攻击和基于查询反馈的黑盒攻击。

**现有痛点**：(a) 缺乏明确的攻击目标导致梯度优化容易陷入局部最优，缺少精确的方向指引；(b) 大多数方法将视觉和文本模态解耦处理，忽略跨模态交互。

**核心矛盾**：VLM 的安全对齐虽然抑制了有害输出，但模型内部仍然编码了与安全相关的知识（类似 ELK 研究中发现的"模型知道但不说"的现象）。这种潜在知识为攻击提供了可利用的结构。

**切入角度**：如果 VLM 在 fusion layer 的潜在表示中存在安全/不安全的决策边界，那么精确找到并跨越这个边界就能系统性地绕过安全机制。

**核心 idea**：先用线性分类器探测 fusion layer 中的安全决策超平面，再用三目标联合优化驱动图像+文本对抗扰动跨越该边界。

## 方法详解

### 整体框架
JailBound 分两个阶段：
- **Stage 1 - Safety Boundary Probing**：在每个 fusion layer 训练 logistic regression 分类器来近似安全决策超平面，获取法向量 $v^{(l)}$ 和跨越距离 $\varepsilon^{(l)}$
- **Stage 2 - Safety Boundary Crossing**：联合优化视觉扰动 $\delta_v^{\text{input}}$ 和文本后缀 $X_t^{\text{suffix}}$，使融合表示跨越决策边界进入不安全区域

### 关键设计

1. **Safety Boundary Probing**：

    - 功能：在每个 fusion layer 近似安全决策超平面
    - 核心思路：构造数据集 $\mathbb{D} = \{(h^{(i)}, y^{(i)})\}$，$h^{(i)} = \phi(x_v^{(i)}, x_t^{(i)})$ 为融合表示，$y^{(i)} \in \{0,1\}$ 为安全标签。训练 logistic regression $P_m(x_v, x_t) = \sigma(w^\top \phi(x_v, x_t) + b)$。决策边界为 $\mathcal{B}^{(l)}(w,b) = \{h^{(l)} | (w^{(l)})^\top h^{(l)} + b^{(l)} = 0\}$。法向量 $v^{(l)} = w^{(l)}/\|w^{(l)}\|_2$，跨越距离 $\varepsilon^{(i)} = |\sigma^{-1}(P_0) - (w^\top h^{(i)} + b)|/\|w\|_2$
    - 设计动机：100% 分类准确率证明 VLM 内部确实存在清晰的线性可分安全边界，这为后续攻击提供了精确目标，彻底解决了"梯度优化缺少方向"的问题

2. **Adversarial Alignment Loss $\mathcal{L}_{\text{align}}$**：

    - 功能：引导扰动后的融合表示向目标区域移动
    - 核心思路：$\mathcal{L}_{\text{align}}^{(l)} = \|\phi^{(l)}(\tilde{x}_v, \tilde{x}_t) - h_{\text{target}}^{(l)}\|_2^2$，其中 $h_{\text{target}}^{(l)} = \phi^{(l)}(x_v, x_t) - \varepsilon^{(l)} \cdot v^{(l)}$，即原始表示沿法向量方向偏移
    - 设计动机：提供了精确的优化目标，避免盲目梯度搜索

3. **Geometric Boundary Loss $\mathcal{L}_{\text{geo}}$**：

    - 功能：确保扰动方向沿法向量轨迹移动
    - 核心思路：$\mathcal{L}_{\text{geo}}^{(l)} = \|\frac{\Delta h^{(l)}}{\|\Delta h^{(l)}\|_2} - v^{(l)}\|_2^2$，其中 $\Delta h^{(l)} = \phi^{(l)}(\tilde{x}_v, \tilde{x}_t) - \phi^{(l)}(x_v, x_t)$
    - 设计动机：防止优化"走弯路"，确保扰动的几何效率最优

4. **Semantic Preservation Loss $\mathcal{L}_{\text{sem}}$**：

    - 功能：约束扰动大小以保持语义一致性
    - 核心思路：$\mathcal{L}_{\text{sem}} = \|\delta_v^{\text{input}}\|_2^2 + \mathcal{L}_{\text{suffix}}(X_t^{\text{suffix}})$，视觉扰动限制 $L_\infty$ 范数 $\leq 8/255$

5. **跨模态联合优化**：

    - 视觉扰动：连续空间梯度下降 $\delta_v^{\text{input}(k+1)} = \Pi_{\Gamma_v}[\delta_v^{\text{input}(k)} - \eta_v \nabla_{\delta_v} \mathcal{L}]$
    - 文本扰动：计算 embedding 空间梯度 $\delta_t^{\text{emb}} = -\eta_t \nabla_{x_t}\mathcal{L}$，然后通过最近邻搜索选择真实 token $t_j^{\text{suffix}} = \arg\min_{v\in V} \|E(v) - (x_t^{(j)} + \delta_t^{\text{emb}(j)})\|_2$
    - 设计动机：同时扰动两个模态，利用跨模态交互产生比单模态更强的攻击效果

### 训练策略
- 总损失 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{align}} + \lambda_1 \mathcal{L}_{\text{sem}} + \lambda_2 \mathcal{L}_{\text{geo}}$，$\lambda_1=2.0, \lambda_2=1.0$
- 安全阈值 $P_0 = 0.3$，视觉学习率 $\eta_v=0.001$，文本学习率 $\eta_t=0.0005$
- 文本后缀长度 20 tokens，优化 100-150 iterations
- 基于 MM-SafetyBench 数据集（13 类禁止内容，1719 样本）

## 实验关键数据

### 主实验 — 白盒攻击成功率 (ASR)

| 类别 | 方法 | Llama-3.2-11B | Qwen2.5-VL-7B | MiniGPT-4 |
|------|------|--------------|---------------|-----------|
| Illegal Activity | Baseline (I0+T0) | 51.47% | 2.94% | 42.65% |
| | 联合攻击 (I1+T1) | 88.24% | 64.71% | 95.59% |
| | **JailBound {I1,T1}** | **95.59%** | **82.35%** | **100.00%** |
| Hate Speech | Baseline | 63.16% | 12.28% | 56.14% |
| | **JailBound** | **95.61%** | **89.47%** | **96.49%** |
| Physical Harm | Baseline | 70.30% | 28.71% | 43.56% |
| | **JailBound** | **97.03%** | **87.13%** | **97.03%** |

### 攻击配置对比

| 配置 | 说明 | 白盒平均 ASR |
|------|------|-------------|
| I0+T0 | 无攻击 baseline | ~50% |
| I0+T1 | 仅文本攻击 | ~75% |
| I1+T0 | 仅视觉攻击 | ~72% |
| I1+T1 | 联合（非迭代） | ~87% |
| **{I1,T1}** | **迭代联合（JailBound）** | **~94%** |

### 黑盒迁移攻击

| 目标模型 | JailBound ASR | 对比 SOTA 提升 |
|----------|-------------|--------------|
| GPT-4o | 75.24% | +21.13% |
| Gemini 2.0 Flash | 70.06% | 显著超越 |
| Claude 3.5 Sonnet | 56.55% | 显著超越 |

### 关键发现
- 迭代联合攻击 {I1,T1} 比非迭代联合 I1+T1 平均高约 7%，证明迭代优化的交叉增强效果
- 在 Qwen2.5-VL-7B 上 baseline ASR 极低（部分低于 10%），但 JailBound 仍能达到 80%+，说明方法对安全对齐较强的模型同样有效
- Safety Boundary Probing 在所有 fusion layer 达到 100% 分类准确率，证实安全边界的线性可分性
- 黑盒迁移性极强，特别是对 GPT-4o 达到 75.24%，远超此前方法

## 亮点与洞察
- **ELK 到 VLM 安全的迁移**非常有洞察力：将"模型内部知道真相"的理论应用于安全场景，发现 VLM 的安全决策存在清晰的线性边界
- **三目标设计思路精巧**：alignment 给方向，geometric 给约束，semantic 给保真，三者互补形成稳健的优化
- **迭代交替优化**策略处理了连续（图像）和离散（文本）两种不同性质的优化问题
- 方法揭示了一个深层安全隐患：即使经过强安全对齐，决策边界仍是线性可分的，攻击者可以精确找到并跨越

## 局限与展望
- 攻击方法的公开可能被恶意使用（不过作为安全研究这也是推动防御进步的必要代价）
- 白盒阶段需要完整模型访问权限，实际部署场景中往往只有 API 访问
- 文本后缀长度固定为 20 tokens，可能不够灵活
- 100%的决策边界分类准确率可能暗示安全对齐过于简单——是否可以通过非线性安全嵌入来增强防御？
- 防御方面缺乏讨论——如何利用 probing 结果来加固安全边界？

## 相关工作与启发
- **vs VAJM**：VAJM 仅用图像对抗扰动绕过安全机制，但缺乏精确方向导引，容易陷入局部最优；JailBound 用 probed boundary 提供了精确的攻击目标
- **vs SCAV**：SCAV 在 LLM 潜空间操纵嵌入但限于单模态，不能操纵视觉输入也无法迁移到黑盒；JailBound 是跨模态的且有极强的迁移性
- **vs FigStep**：FigStep 使用排版图像绕过文本过滤器，属于 prompt 工程层面；JailBound 从潜空间层面系统性攻击，更底层更有效

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将 ELK 理论引入 VLM 安全攻击，safety boundary probing 概念新颖且实用
- 实验充分度: ⭐⭐⭐⭐ 6 个 VLM + 13 类安全场景 + 白盒/黑盒全面覆盖，但缺少防御方法对比
- 写作质量: ⭐⭐⭐⭐ 框架清晰，数学形式化完整，但部分符号较多可以精简
- 价值: ⭐⭐⭐⭐⭐ 对 VLM 安全领域有重要警示意义，揭示了线性安全边界的脆弱性，推动更强防御研究

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation](../../ACL2025/multimodal_vlm/do_vision-language_models_have_internal_world_models_towards_an_atomic_evaluatio.md)
- [\[ICCV 2025\] IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves](../../ICCV2025/multimodal_vlm/ideator_jailbreaking_and_benchmarking_large_visionlanguage_m.md)
- [\[ICML 2026\] Jailbreaking Vision-Language Models Through the Visual Modality](../../ICML2026/multimodal_vlm/jailbreaking_vision-language_models_through_the_visual_modality.md)
- [\[ICCV 2025\] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](../../ICCV2025/multimodal_vlm/jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)
- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)

</div>

<!-- RELATED:END -->
