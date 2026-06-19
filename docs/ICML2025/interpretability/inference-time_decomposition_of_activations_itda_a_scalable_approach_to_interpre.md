---
title: >-
  [论文解读] Inference-Time Decomposition of Activations (ITDA): A Scalable Approach to Interpreting Large Language Models
description: >-
  [ICML 2025][可解释性][稀疏自编码器] 提出 ITDA，一种基于匹配追踪（Matching Pursuit）的推理时激活分解方法，以仅 1% 的 SAE 训练成本实现可比的重构性能，可扩展到 405B 参数模型，并天然支持跨模型表示比较。 稀疏自编码器（SAE）是当前将 LLM 激活分解为可解释潜变量的主流方法…
tags:
  - "ICML 2025"
  - "可解释性"
  - "稀疏自编码器"
  - "匹配追踪"
  - "字典学习"
  - "表示相似性"
---

# Inference-Time Decomposition of Activations (ITDA): A Scalable Approach to Interpreting Large Language Models

**会议**: ICML 2025  
**arXiv**: [2505.17769](https://arxiv.org/abs/2505.17769)  
**代码**: [github.com/pleask/itda](https://github.com/pleask/itda)  
**领域**: 可解释性  
**关键词**: 稀疏自编码器, 可解释性, 匹配追踪, 字典学习, 表示相似性

## 一句话总结

提出 ITDA，一种基于匹配追踪（Matching Pursuit）的推理时激活分解方法，以仅 1% 的 SAE 训练成本实现可比的重构性能，可扩展到 405B 参数模型，并天然支持跨模型表示比较。

## 研究背景与动机

稀疏自编码器（SAE）是当前将 LLM 激活分解为可解释潜变量的主流方法，但存在两大核心瓶颈：

**训练成本高昂**：SAE 需要数亿乃至数十亿 token 的模型激活数据进行训练，且 SAE 自身的参数量可能超过被分析的 LLM（如 Gemma 2 2B 的 SAE 竟有 50 亿参数）。当前开源 SAE 仅覆盖 ≤27B 参数的模型。

**跨模型不可比**：SAE 的潜变量通过梯度下降在特定模型的激活空间中学习得到，不同模型的 SAE 潜变量之间没有天然的对应关系，无法直接进行模型间比较。

受相对表示相似性方法（Moschella et al., 2022）的启发——即不同模型的绝对表示虽然不同，但表示空间中元素间的角度关系保持不变——作者提出了一种轻量级的替代方案。

## 方法详解

### 整体框架

ITDA 的核心思路是：**不学习编码器，而是在推理时用匹配追踪算法将激活分解到一个由真实激活构成的字典上**。整体流程分为三步：

1. **字典构建**（离线，贪心采样）：从训练数据中迭代选取激活向量，构成字典 $\mathbf{D} \in \mathbb{R}^{n \times d}$
2. **稀疏编码**（在线，匹配追踪）：对目标激活 $\mathbf{x}$，用 Matching Pursuit 求解稀疏系数 $\mathbf{a} = \text{MP}(\mathbf{x}, \mathbf{D}, L_0)$
3. **重构**：$\hat{\mathbf{x}} = \mathbf{a}\mathbf{D}$

与 SAE 的对比：SAE 用学习的编码器+解码器 $\mathbf{f}(\mathbf{x}) = \sigma(\mathbf{W}^{\text{enc}}\mathbf{x} + \mathbf{b}^{\text{enc}})$，$\hat{\mathbf{x}} = \mathbf{W}^{\text{dec}}\mathbf{f} + \mathbf{b}^{\text{dec}}$；而 ITDA 完全去掉了参数化编码器，改用推理时优化。

### 关键设计

#### 1. 相对表示的理论基础

不同模型学到的绝对表示 $e^{(i)} = E_\theta(\mathbf{x}^{(i)})$ 可能存在旋转或仿射变换 $T$ 的差异。但角度关系是不变的：

$$\angle(\mathbf{e}^{(i)}, \mathbf{e}^{(j)}) = \angle(T\mathbf{e}^{(i)}, T\mathbf{e}^{(j)})$$

ITDA 利用余弦相似度 $S_C(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}$ 作为相似性函数，天然保持角度不变性。

#### 2. 推理时稀疏编码（Matching Pursuit）

给定输入 $\mathbf{x} \in \mathbb{R}^d$ 和字典 $\mathbf{D}$，求解稀疏编码问题：

$$\min_{\mathbf{a} \in \mathbb{R}^n} ||\mathbf{x} - \mathbf{a}\mathbf{D}|| \quad \text{s.t.} \quad ||\mathbf{a}||_0 \leq L_0$$

MP 算法的每次迭代：

- **选择**：找到与当前残差相关性最大的字典原子 $\mathbf{d}_j$
- **更新**：将残差投影到新选中的原子方向上并减去
- **重复**：迭代 $L_0$ 步以达到目标稀疏度

这里的相关性等价于未归一化的余弦相似度，与相对表示框架一致。

#### 3. 贪心字典构建

与 Moschella et al. 随机采样锚点不同，ITDA **确定性地、迭代地**构建字典：

**算法 1: ITDA 字典训练**

输入：训练数据 $\{x_i\}$，稀疏度 $L_0$，阈值 $\tau$

1. 初始化字典 $\mathbf{D}$（选取高频激活或随机采样）
2. 对每个训练 batch $\mathcal{B}$ 中的样本 $\mathbf{x}$：
    - 计算稀疏编码 $\mathbf{a} = \text{OMP}(\mathbf{x}, \mathbf{D}, L_0)$
    - 重构 $\hat{\mathbf{x}} = \mathbf{a}\mathbf{D}$
    - 计算重构损失 $\ell(\mathbf{x}) = ||\mathbf{x} - \hat{\mathbf{x}}||_2^2$
    - 若 $\ell(\mathbf{x}) > \tau$，将归一化的 $\mathbf{x}$ 加入字典
3. 过滤字典中的重复激活

关键参数 $\tau$（损失阈值）控制字典大小：**更低的 $\tau$ → 更大的字典 → 更低的重构误差**。这与 SAE 的固定字典大小设计形成对比——SAE 的字典大小在训练前预设，而 ITDA 由重构质量阈值自适应确定。

#### 4. 可解释标签

ITDA 字典原子的一个独特优势：每个原子自带可解释标签——即产生该激活的 **prompt + token** 对。SAE 的潜变量需要额外的自动化可解释性分析（如检查高激活样本）才能理解含义，而 ITDA 的标签提供了天然的语义信息。

#### 5. 跨模型表示相似性度量

基于 ITDA 字典的可解释标签，作者提出了一种新的表示相似性度量：

- 对两个模型分别构建 ITDA 字典，每个字典原子对应一个 (prompt, token) 对
- 用 **Jaccard 相似度（IoU）** 比较两个字典的 (prompt, token) 标签集合
- 这避免了直接比较不同模型的激活值（它们处于不同空间），转而比较"哪些输入对模型来说是重要的"

### 损失函数 / 训练策略

ITDA 没有传统意义上的损失函数优化过程。其"训练"本质是字典构建，核心超参数为：

- **稀疏度 $L_0$**：每个激活使用的字典原子数量
- **阈值 $\tau$**：决定何时新增字典原子的重构误差阈值
- **字典初始化**：使用最高频的激活作为初始字典元素

对比 SAE 的训练损失：$\mathcal{L}(\mathbf{x}) = ||\mathbf{x} - \hat{\mathbf{x}}||_2^2 + \lambda \mathcal{S}(\mathbf{f}(\mathbf{x})) + \alpha \mathcal{L}_{\text{aux}}$，ITDA 不需要稀疏正则项（稀疏性由 $L_0$ 硬约束保证），也不需要辅助损失。

## 实验关键数据

### 主实验

#### 训练效率对比

| 方法 | 训练 token 数 | GPT-2 训练时间 | 最大可处理模型 |
|------|-------------|--------------|-------------|
| SAE | 数亿~数十亿 | 数小时 | 27B (开源) |
| **ITDA** | **~100万** | **数分钟** | **405B** |
| 效率提升 | **~100×** | **~100×** | **~15×** |

#### 重构性能对比

| 模型 | 方法 | 重构效果 | 交叉熵退化 |
|------|------|---------|-----------|
| Pythia 系列 | SAE | 基准 | 基准 |
| Pythia 系列 | **ITDA** | **相当** | **相似或略差** |
| Gemma-2 | SAE | 基准 | 基准 |
| Gemma-2 | **ITDA** | 较差 | 明显更差 |
| Llama-3.1 70B | **ITDA** | ✓（首次） | — |
| Llama-3.1 405B | **ITDA** | ✓（首次） | — |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 阈值 $\tau$ ↓ | 字典变大，重构误差 ↓ | $\tau$ 是控制精度-效率 trade-off 的核心旋钮 |
| 阈值 $\tau$ ↑ | 字典变小，重构误差 ↑ | 极端情况可用极少原子 |
| 随机采样字典 | 重构效果差 | 贪心策略显著优于随机 |
| 确定性贪心字典 | 重构效果好 | 且保证跨运行可复现 |
| 稀疏度 $L_0$ ↑ | 重构 ↑，计算 ↑ | 控制稀疏码的非零元素数 |

### 关键发现

1. **表示相似性 SOTA**：基于 Jaccard 字典差异的 ITDA 度量在 Kornblith et al. 的 layer-matching benchmark 上超越了 CKA、SVCCA 和相对表示方法，达到当前最优。

2. **可扩展性突破**：ITDA 首次将稀疏字典分解应用到 70B 和 405B 参数的 LLM，比现有开源 SAE 覆盖的最大模型大一个数量级。

3. **自动化可解释性得分可比**：ITDA 字典原子的自动化可解释性评分与 SAE 潜变量相当，说明贪心采样的真实激活同样具有单语义（monosemantic）特性。

4. **模型依赖性**：ITDA 在 Pythia 系列模型上表现与 SAE 接近，但在 Gemma-2 上交叉熵退化明显更严重，说明方法的有效性存在模型依赖。

5. **层冻结实验复现**：ITDA 字典差异度量成功复现了 Raghu et al. (2017) 的层冻结实验结论，进一步验证了其作为表示相似性工具的可靠性。

## 亮点与洞察

1. **思路极其简洁**：用经典信号处理方法（Matching Pursuit）替代深度学习（SAE），以"不学习"的方式解决了一个计算瓶颈问题。100× 的加速来自于将"学习字典+编码器"简化为"采样字典+推理编码"。

2. **标签即可解释性**：ITDA 原子的 (prompt, token) 标签是一种优雅的设计——SAE 需要额外步骤解释每个潜变量的含义，而 ITDA 的字典本身就携带语义信息。

3. **跨模型比较的新范式**：将"比较不同模型的激活空间"（困难，涉及对齐问题）转化为"比较哪些输入被选入字典"（容易，只需比较标签集合的 Jaccard 相似度）。这个思路可能对 model merging、知识蒸馏等方向有启发。

4. **自适应字典大小**：用阈值 $\tau$ 控制字典规模而非预设大小，使得方法能根据激活空间的实际复杂度自适应调整——简单表示空间用小字典，复杂空间用大字典。

## 局限与展望

1. **Gemma-2 上效果较差**：交叉熵退化明显大于 SAE，说明匹配追踪在某些模型架构上的激活空间中表达能力不足，可能需要探索其他推理时优化算法（如 OMP 的改进变体或 FISTA）。

2. **推理时计算开销**：匹配追踪在推理时需要与整个字典计算相关性，字典越大推理越慢。SAE 的编码只需一次矩阵乘法 + 激活函数，推理效率更高。

3. **未探索 model diffing**：作者提到了用 ITDA 识别微调导致的行为差异（如 scheming、sycophancy）的潜力，但论文中没有实际实验，这是一个重要的未来方向。

4. **字典依赖训练数据**：字典原子都来自训练集的真实激活，可能无法覆盖分布外（OOD）输入的激活模式，而 SAE 学习到的解码器可以组合出未见过的方向。

5. **批处理中的重复问题**：同一 batch 中相似输入可能被重复添加到字典，虽然有后处理过滤，但效率不够优雅。

## 相关工作与启发

- **经典稀疏编码**（K-SVD, Matching Pursuit, FISTA）回归：深度学习时代倾向用神经网络解决一切，本文证明经典算法在特定场景仍有优势。
- **Crosscoders**（Lindsey et al., 2024）：在多模型表示上训练 SAE 以找跨模型特征，ITDA 的标签方法提供了更轻量的替代。
- **相对表示**（Moschella et al., 2022）：ITDA 是其思想的自然延伸——从简单的余弦相似度向量扩展为匹配追踪的稀疏分解。
- **未来方向**：将 ITDA 与 circuit analysis 结合，快速分析超大模型中的计算回路。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将经典匹配追踪引入 LLM 可解释性，思路新颖且合理
- **实验充分度**: ⭐⭐⭐⭐ — 多模型对比、可扩展性展示充分，但 Gemma-2 上的不足需更深入分析
- **写作质量**: ⭐⭐⭐⭐⭐ — 动机清晰，方法描述严谨，与 SAE 的对比贯穿全文
- **价值**: ⭐⭐⭐⭐ — 100× 加速和 405B 可扩展性有重大实用价值，跨模型比较开辟新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Supernova Event Dataset: Interpreting Large Language Models' Personality through Critical Event Analysis](supernova_event_dataset_interpreting_large_language_models_personality_through_c.md)
- [\[ICML 2026\] Scalable Circuit Learning for Interpreting Large Language Models](../../ICML2026/interpretability/scalable_circuit_learning_for_interpreting_large_language_models.md)
- [\[ACL 2025\] Mechanistic Interpretability of Emotion Inference in Large Language Models](../../ACL2025/interpretability/mechanistic_interpretability_of_emotion_inference_in_large_language_models.md)
- [\[ACL 2026\] FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models](../../ACL2026/interpretability/finesteer_a_unified_framework_for_fine-grained_inference-time_steering_in_large_.md)
- [\[ICML 2025\] A Reasoning-Based Approach to Cryptic Crossword Clue Solving](a_reasoning-based_approach_to_cryptic_crossword_clue_solving.md)

</div>

<!-- RELATED:END -->
