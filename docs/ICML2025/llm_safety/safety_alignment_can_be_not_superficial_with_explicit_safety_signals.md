---
title: >-
  [论文解读] Safety Alignment Can Be Not Superficial With Explicit Safety Signals
description: >-
  [ICML 2025][LLM安全][Safety Alignment] 通过在LLM中引入显式的安全二分类任务（[CLS] token），并设计策略性注意力机制和解码策略，在推理过程中动态评估安全性，以不到0.2x的额外开销将对抗攻击成功率从90%+降至接近0%。 现有LLM安全对齐方法（SFT/DPO/RLHF）被发现只…
tags:
  - "ICML 2025"
  - "LLM安全"
  - "Safety Alignment"
  - "Jailbreak Defense"
  - "Binary Classification"
  - "Strategic Decoding"
  - "Adversarial Robustness"
---

# Safety Alignment Can Be Not Superficial With Explicit Safety Signals

**会议**: ICML 2025  
**arXiv**: [2505.17072](https://arxiv.org/abs/2505.17072)  
**代码**: [https://sa-ess.github.io/](https://sa-ess.github.io/)  
**领域**: LLM对齐/安全  
**关键词**: Safety Alignment, Jailbreak Defense, Binary Classification, Strategic Decoding, Adversarial Robustness

## 一句话总结

通过在LLM中引入显式的安全二分类任务（[CLS] token），并设计策略性注意力机制和解码策略，在推理过程中动态评估安全性，以不到0.2x的额外开销将对抗攻击成功率从90%+降至接近0%。

## 研究背景与动机

现有LLM安全对齐方法（SFT/DPO/RLHF）被发现只是"表面对齐"——模型在面对精心设计的对抗攻击（jailbreak、prefill攻击、解码参数攻击等）时，安全机制很容易被绕过。先前工作（Li & Kim 2024）指出，安全对齐本质上可以归结为一个二分类任务（拒绝/执行），但现有方法让模型**隐式**学习这个任务，导致安全信号被其他优化目标（语气、风格、偏好等）稀释。

具体表现为：
- 对抗攻击下模型top-K logits表现出明显的犹豫和混乱（高熵、低sharpness）
- 现有数据增强方案（Qi et al. 2024, Yuan et al. 2024）只能处理简单的安全→不安全翻转，无法应对嵌套的、出现在回复中后段的有害内容
- 隐式安全信号在对抗场景下不够可靠，决策边界模糊

本文核心洞察：**将安全判断从隐式推理转变为显式二分类任务**，可以从根本上解决安全对齐的表面性问题。

## 方法详解

### 整体框架

在预训练和SFT阶段引入特殊的 [CLS] token，将其作为安全分类器；在推理阶段通过策略性注意力机制和策略性解码策略两个组件，隐式+显式地利用安全信号指导生成过程。整体设计分三步：
1. 训练阶段：将 [CLS] token 加入输入序列开头，输出经分类头判断benign/malicious
2. 推理阶段（隐式）：通过注意力机制让 [CLS] 的隐状态影响生成token
3. 推理阶段（显式）：利用 [CLS] 的分类预测直接介入解码过程

### 关键设计

#### 1. **显式安全二分类任务（[CLS] Token）**

受BERT启发，在每个输入序列开头插入 [CLS] token，其输出经分类头判定输入及已生成内容安全与否。为平衡分类与生成能力，设计了精细的注意力控制：

- **预训练阶段**：[CLS] 可注意所有token，但其他token不能注意 [CLS]，保持原始因果注意力逻辑不变
- **SFT阶段**：query token不能注意 [CLS]，response token可以注意 [CLS]；[CLS] 只能注意query token不能注意response token
- **两阶段均设小系数**控制分类loss权重，防止分类目标主导优化

数据集构建上，预训练用LLaMA3-Guard自动标注Wikipedia数据，SFT用Lima（benign）+Alert（malicious）+Alpaca采样构成均衡的29,600样本数据集。

#### 2. **策略性注意力机制（Strategic Attention Mechanism）**

在推理过程中动态重新评估 [CLS] token，根据当前生成状态调整其注意力范围，共设计四条规则：

- **Rule 1**（初始判恶意）：[CLS] 只关注输入token和前r₁个生成token，无需关注后续所有token
- **Rule 2**（初始判良性）：[CLS] 仅关注最新的r₂个生成token，聚焦新内容、降低计算量
- **Rule 3**（从良性转恶意）：记录转变点S_t，将注意力集中在 [S_t - r₂, S_t + r₃] 范围，提供容错机制防止关键词触发误分类，若误触发则自动回退到Rule 2
- **Rule 4**：跳过PAD、BOS、指令token等辅助token

超参数 r₁ = r₂ = r₃ = 10，可根据应用场景灵活调整。

#### 3. **策略性解码策略（Strategic Decoding Strategy）**

显式利用 [CLS] 分类输出指导解码，提出三级依赖方案：

- **低依赖**：完全依赖注意力机制，忽略分类预测（不够可靠）
- **高依赖**：一旦分类为恶意立即终止+固定拒绝回复（高误报率）
- **中依赖（本文方案）**：
    - 若初始分类为恶意 → 立即插入引导token（"Sorry, I cannot fulfill your request because..."）并解释原因
    - 若初始良性但后续连续τ步分类为恶意 → 在转变点插入引导token

设计理由：
- 解决决策边界模糊问题——对抗查询中分类概率常在0.5附近徘徊，解码策略可强制做出明确决定
- 采用Chain-of-Thought风格拒绝——不仅拒绝还解释原因，增强模型理解、减少反转可能和误报

### 损失函数 / 训练策略

预训练损失：$\mathcal{L}_{pretraining} = \mathcal{L}_{lm} + \lambda_1 \cdot \mathcal{L}_{cls}$

对齐损失：$\mathcal{L}_{alignment} = \mathcal{L}_{sft} + \lambda_2 \cdot \mathcal{L}_{cls}$

其中 $\mathcal{L}_{cls}$ 为 [CLS] token输出与ground truth的交叉熵，$\lambda_1 = 0.01$，$\lambda_2 = 0.1/0.01$，τ ≤ 3。

推理阶段采用**退火策略（Annealing）**减少重分类频率：早期频繁评估，逐步降低频率直至不再重评估，以不到0.2x的额外开销保持与每步评估相当的安全性能。

## 实验关键数据

### 主实验

基础模型：Llama2-7B；对齐模型：Mistral-7B-Instruct-v0.2。

| 数据集/攻击 | 指标(ASR↓) | Llama2-7B-CLS (本文) | Llama2-7B-Chat (RLHF) | 提升 |
|-------------|-----------|---------------------|----------------------|------|
| AdvBench / Prefill | ASR | **0.4%** | 39.62% | ~100x |
| HEx-PHI / Prefill | ASR | **1.2%** | 60.91% | ~50x |
| HarmBench / GCG | ASR | **0.0%** | 28.0% | 完全防御 |
| AdvBench / Decoding | ASR | **0.0%** | 87.0% | 完全防御 |
| MaliciousInstruct / Decoding | ASR | **0.0%** | 83.0% | 完全防御 |
| AdvBench / AutoDAN-T | ASR | **0.77%** | 61.3% | ~80x |
| AdvBench / PAP | ASR | **0.0%** | 28.26% | 完全防御 |

与SOTA数据增强方法对比（Qi et al. 2024）：

| 方法 | Prefill 5T | Prefill 40T | Decoding (HEx-PHI) | Decoding (MalInst) |
|------|-----------|------------|-------------------|-------------------|
| Llama2-7B-Chat | 42.1% | 57.0% | 54.9% | 84.3% |
| Llama2-7B-Chat-Aug | 2.8% | 4.5% | 11.3% | 1.0% |
| **Llama2-7B-CLS** | **0.9%** | **2.1%** | **0.0%** | **0.0%** |

### 消融实验

| 配置 | 关键指标(ASR) | 说明 |
|------|-------------|------|
| 去除预训练阶段 | 略有上升 | 预训练有帮助但增益有限，可能因LLaMA3-Guard标注噪声 |
| 去除策略性注意力 | 明显下降 | 注意力机制保证安全信号对推理方向变化的敏感性 |
| 去除策略性解码 | 明显下降 | 解码策略保证模型能及时有效响应安全变化 |
| FirstOnly重分类 | 最差 | 仅初始分类不足以应对后续攻击 |
| Periodic(每10步) | 接近最优 | 定期评估已有不错效果 |
| Annealing退火 | ≈Every | <0.2x开销达到每步评估相当性能 |
| Every(每步) | 最优 | 计算开销最高 |

### 关键发现

1. **探针实验**：随着对抗复杂度增加（Direct → Prefill → Nested），模型输出熵增加、sharpness降低，表明模型在对抗攻击下信心不足、安全推理不稳定
2. **GCG攻击完全无效**：动态重分类机制破坏了GCG依赖的静态对抗信号，使优化的对抗后缀失效
3. **跨模型族适用**：增强后的Mistral-7B-Instruct-v0.2-CLS首次在安全性上超过Llama2-7B-Chat，同时保持Mistral族在MT-Bench(7.38)、GSM8K(41.77)上的优势
4. **对采样不敏感**：方法在多轮生成中标准差接近零，对解码攻击天然免疫

## 亮点与洞察

1. **从隐式到显式的范式转换**：将安全判断从隐藏在生成过程中的隐式推理，转变为明确的二分类任务，思路清晰且效果显著
2. **动态重评估**: 不是一次性判断安全，而是在整个生成过程中持续监控，能应对嵌套攻击和中途出现的有害内容
3. **计算效率优秀**：Annealing策略使额外开销<0.2x，训练阶段仅多1个token，实际可部署
4. **GCG防御机理深刻**：动态重分类从根本上破坏了梯度优化攻击的前提——静态对抗信号
5. **与现有方法兼容**：可作为SFT/DPO/RLHF的**后增强阶段**叠加使用，而非替代

## 局限与展望

1. **仅限文本模态**：未验证在多模态（图文、语音等）场景下的有效性
2. **预训练增益有限**：受限于LLaMA3-Guard标注质量和计算资源，预训练阶段带来的额外收益不显著
3. **二分类可能过于简化**：恶意程度是连续谱，二分类在边界案例上可能不够nuanced
4. **超参数敏感性**：r₁/r₂/r₃/τ等超参需要针对不同场景调整，通用性待验证
5. **过度拒绝风险**：虽然通过CoT解释和τ连续步阈值缓解，但在实际部署中的误报率需更多评估

## 相关工作与启发

- **Li & Kim (2024)**：提出浅层安全对齐假说，本文是该假说的直接解决方案
- **Qi et al. (2024)**：数据增强方法增加训练多样性，但本文表明显式信号比更多数据更有效
- **BERT [CLS] token**：巧妙借鉴判别模型的设计用于生成模型的安全场景
- **启发**：安全对齐不应只依赖数据或训练策略，模型架构层面的显式设计可能是更根本的解决路径；类似思路可扩展到其他可靠性需求（真实性、公平性等）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 显式安全信号 + 动态重分类思路新颖，但整体框架相对直觉
- 实验充分度: ⭐⭐⭐⭐⭐ — 覆盖多种攻击类型、多基线对比、丰富消融实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，challenge-solution形式组织佳，但LaTeX公式稍显冗余
- 价值: ⭐⭐⭐⭐ — 安全对齐是重要问题，方法实用且可叠加现有方案，但多模态扩展性未验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Improving LLM Safety Alignment with Dual-Objective Optimization](improving_llm_safety_alignment_with_dual-objective_optimization.md)
- [\[ICML 2025\] Learning Safety Constraints for Large Language Models](learning_safety_constraints_for_large_language_models.md)
- [\[ACL 2026\] Reasoning Structure Matters for Safety Alignment of Reasoning Models](../../ACL2026/llm_safety/reasoning_structure_matters_for_safety_alignment_of_reasoning_models.md)
- [\[ACL 2026\] Into the Gray Zone: Domain Contexts Can Blur LLM Safety Boundaries](../../ACL2026/llm_safety/into_the_gray_zone_domain_contexts_can_blur_llm_safety_boundaries.md)
- [\[ICLR 2026\] Any-Depth Alignment: Unlocking Innate Safety Alignment of LLMs to Any-Depth](../../ICLR2026/llm_safety/any-depth_alignment_unlocking_innate_safety_alignment_of_llms_to_any-depth.md)

</div>

<!-- RELATED:END -->
