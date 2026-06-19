---
title: >-
  [论文解读] PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge LLM Inference via Semantic Sketch Collaboration
description: >-
  [AAAI 2026][LLM安全][隐私保护] 提出 PRISM 框架，通过上下文感知的软门控路由机制将用户 prompt 动态分配到云端/边缘/协作三种推理模式，并在协作模式中使用自适应两层本地差分隐私（LDP）和语义草图协作，实现隐私-效用-效率的三方平衡。 大语言模型（LLM）通常部署在云端以满足大规模推理需求…
tags:
  - "AAAI 2026"
  - "LLM安全"
  - "隐私保护"
  - "云边协同推理"
  - "LLM隐私"
  - "差分隐私"
  - "语义草图"
---

# PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge LLM Inference via Semantic Sketch Collaboration

**会议**: AAAI 2026  
**arXiv**: [2511.22788](https://arxiv.org/abs/2511.22788)  
**代码**: [https://github.com/Junfei-Z/PRISM](https://github.com/Junfei-Z/PRISM)  
**领域**: AI安全  
**关键词**: 隐私保护, 云边协同推理, LLM隐私, 差分隐私, 语义草图

## 一句话总结

提出 PRISM 框架，通过上下文感知的软门控路由机制将用户 prompt 动态分配到云端/边缘/协作三种推理模式，并在协作模式中使用自适应两层本地差分隐私（LDP）和语义草图协作，实现隐私-效用-效率的三方平衡。

## 研究背景与动机

大语言模型（LLM）通常部署在云端以满足大规模推理需求，但这带来了两个核心问题：

**隐私风险**: 用户 prompt 中常包含敏感个人信息（如医疗记录、金融数据），全量发送到云端存在隐私泄露风险

**通信开销**: 大量 prompt 传输带来显著的延迟和能耗

现有云边协同方案的局限性：

- **二元路由决策**: 基于简单阈值将 prompt 分为"本地处理"或"云端处理"，容易误分类——要么过度负担边缘设备，要么泄露隐私
- **统一噪声注入**: prompt 级别的差分隐私方案（如 Split-and-Denoise, DP-Forward）对所有 token/维度均匀加噪，不区分敏感度。这导致两个问题：
    - 对非敏感查询（如"法国首都是哪里？"）引入不必要的噪声
    - 统一扰动导致语义严重失真，云端 LLM 产生模糊或回避性回复（如 "I cannot provide information about [MASKED_ENTITY]"）

**核心动机**: 需要一种**感知 prompt 语义的自适应隐私保护机制**，根据每个 prompt 的具体敏感度动态选择保护策略。

## 方法详解

### 整体框架

PRISM 由四个阶段组成：

1. **敏感度分析（Sensitivity Profiling）**: 边缘设备对 prompt 进行实体级别的敏感度评估
2. **软门控路由（Soft Gating）**: 基于敏感度特征，用软分类器选择执行模式（云端/边缘/协作）
3. **自适应两层 LDP**: 对协作路径中的敏感实体，按类别风险分配差分隐私预算
4. **语义草图协作（Semantic Sketch Collaboration）**: 云端 LLM 从扰动 prompt 生成语义草图，边缘 SLM 结合原始上下文精炼最终回复

### 关键设计

#### 1. **敏感度分析模块**

轻量级边缘模块，包含两个输出：

- **风险评分 $R(P)$**: 反映 prompt 整体隐私敏感度

$$R(P) = \sum_{i=1}^m w_{c_i} \cdot \mathbb{I}(e_i)$$

其中 $w_{c_i}$ 是实体类别的预定义敏感度权重（如 $w_{\text{PERSON}} > w_{\text{NATIONALITY}}$）。

- **上下文指示器 $\Delta$**: 检测是否存在隐私语言线索（第一人称代词、人名实体）

$$\Delta = \max_{x_j \in P} \mathbb{I}(x_j \in \mathcal{F})$$

关键设计意图：同一实体在不同上下文中敏感度不同。"I plan to travel to Tokyo"中的 Tokyo 涉及个人信息（有"I"），而"Which country is Tokyo in?"中的 Tokyo 是公共知识。当 $\Delta > 0$ 时，所有实体保守地标记为需保护。

#### 2. **软门控路由机制**

与硬阈值路由不同，PRISM 使用熵正则化的软分类器：

$$\boldsymbol{\pi} = \text{softmax}(f_\theta(\mathbf{z})) \in \mathbb{R}^3$$

产生三个模式的概率分布 $\boldsymbol{\pi} = (\pi_{\text{cloud}}, \pi_{\text{collab}}, \pi_{\text{local}})$。

**训练损失** 包含任务损失和熵正则项：

$$\mathcal{L}_{\text{gating}} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{H}(\boldsymbol{\pi})$$

- 低熵：鼓励自信的路由决策
- 高熵：容纳模糊或不确定的情况
- 推理时取 $\arg\max$，确保确定性路由

#### 3. **自适应两层本地差分隐私**

这是本文最核心的技术贡献。针对简单匿名化（替换为 `<NAME>`）无法防止链接攻击的问题，以及统一 LDP 导致过度扰动的问题，提出分层自适应 LDP：

**预算分配**: 根据实体类别敏感度权重 $w_{c_i}$ 自适应分配：

$$\varepsilon_1 = \varepsilon_{\text{total}} \cdot \frac{w_{c_i}}{w_{c_i} + (1 - w_{c_i}) \cdot \alpha}, \quad \varepsilon_2 = \varepsilon_{\text{total}} - \varepsilon_1$$

- **高敏感实体**（如 NAME）: $w_{c_i} \to 1$，更多预算分给 $\varepsilon_1$（类别层），隐藏实体类型
- **低敏感实体**（如 LOCATION）: $w_{c_i} \to 0$，更多预算分给 $\varepsilon_2$（值层），保留语义一致性

两层均使用**随机响应（Randomized Response）**机制实现 $\varepsilon$-LDP。

**隐私保证**（Theorem 1）: 组合机制 $M = M_2 \circ M_1$ 满足 $(\varepsilon_1 + \varepsilon_2)$-LDP，由顺序组合定理保证。

#### 4. **语义草图协作**

扰动后的 prompt $P^*$ 以明文发送到云端（避免嵌入传输和分词器同步问题）。协作分两步：

**云端草图生成**: 使用 few-shot 提示，云端 LLM 从 $P^*$ 生成结构化语义草图 $S$：

$$S = \mathcal{G}_{\text{cloud}}(\mathcal{C}_{\text{cloud}})$$

草图采用简洁格式，省略被混淆的敏感实体，保持语义对齐。

**边缘精炼**: 边缘 SLM 结合草图 $S$ 和原始 prompt $P$（本地保留）重建最终回复：

$$\hat{R} = \mathcal{G}_{\text{edge}}(\mathcal{C}_{\text{edge}})$$

### 损失函数 / 训练策略

- 门控模块：交叉熵 + 熵正则化损失
- 隐私机制：数学证明保证 $\varepsilon$-LDP
- 推理采用 in-context learning，无需额外训练

## 实验关键数据

### 主实验

| 方法 | 完成时间(s) | 能耗(J) | 推理质量(IQ) | 说明 |
|---|---|---|---|---|
| Cloud-Only | 5.13 | 296 | 8.14 | 无隐私保护 |
| **PRISM** | **7.92** | **687** | **6.88** | 最优隐私保护方案 |
| Uniform LDP | 20.56 | 1708 | 5.72 | 统一加噪 |
| Selective LDP | 21.22 | 1771 | 5.94 | 选择性加噪 |
| Edge-Only | 17.84 | 1574 | 5.09 | 纯边端推理 |

**PRISM 仅需 Cloud-Only 的 1.54× 延迟和 2.32× 能耗**，同时提供强隐私保护，而 Uniform/Selective LDP 需要 4× 延迟和 6× 能耗。

### 消融实验

| 云端LLM + 边端SLM | 完成时间(s) | 能耗(J) | 推理质量 |
|---|---|---|---|
| GPT-4o + Phi-3.5-mini-3.5B | 8.29 | 684 | 7.00 |
| GPT-4o + Qwen1.5-1.8B | 7.08 | 632 | 6.91 |
| GPT-4o + StableLM-2-1.6B | 7.34 | 658 | **7.16** |
| Qwen3-235B + Phi-3.5-mini-3.5B | 8.59 | 739 | **7.22** |
| Qwen3-235B + TinyLLaMA-1.1B | 8.11 | 698 | 7.19 |

所有 8 种云边模型组合都保持 IQ ≥ 6.9（TinyLLaMA 除外），证明框架对异构部署的适应性。

### 关键发现

1. **自适应路由是效率关键**: PRISM 将非敏感 prompt 直接发云端、低风险 prompt 走协作、高风险 prompt 留本地，避免了"一刀切"的计算浪费
2. **隐私预算变化下 PRISM 性能稳定**: 不同 $\varepsilon$ 下，PRISM 的推理质量、能耗、延迟波动极小，而 Uniform/Selective LDP 随预算收紧性能显著下降
3. **语义草图设计有效**: 云端不直接生成完整回复，而是生成结构化草图，配合边端精炼既保护隐私又保持质量

## 亮点与洞察

- **实体级细粒度隐私保护**: 不再将 prompt 视为铁板一块，而是区分每个实体的敏感度
- **两层 LDP 的理论严谨性**: 给出完整的隐私保证证明和预算分配单调性证明
- **真实硬件评测**: 在实际 RTX 3070 端侧设备上测量能耗和延迟，而非仅靠模拟
- **构建了覆盖 4 领域的半合成数据集**: 涵盖旅游、医疗、银行、通用知识

## 局限与展望

1. **NER 依赖**: 敏感度分析依赖命名实体识别的准确性，NER 漏检会导致隐私泄露
2. **数据集规模有限**: 每个领域仅 40 条 prompt，缺乏大规模真实场景验证
3. **单边缘设备**: 未考虑多边缘设备协同推理场景
4. **草图质量受扰动程度影响**: 当隐私预算极低时，草图质量可能急剧下降
5. **推理质量评估依赖 GPT-4o**: 使用 LLM 作为评判存在偏差风险

## 相关工作与启发

- 云边协同 LLM 推理是隐私保护的重要方向，从简单的二元路由到 PRISM 的三模式软路由是重要进步
- 两层 LDP 的思想可推广到其他隐私场景（如联邦学习中的梯度保护）
- 语义草图协作的 "cloud generates sketch, edge refines" 模式可用于其他需要隐私保护的生成任务
- 未来可结合联邦学习训练个性化的敏感度分析模块

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (三模式路由 + 自适应两层LDP + 语义草图，贡献层次丰富)
- 实验充分度: ⭐⭐⭐⭐ (真实硬件评测，8种模型组合，但数据集规模偏小)
- 写作质量: ⭐⭐⭐⭐⭐ (理论证明严谨，系统架构描述清晰)
- 价值: ⭐⭐⭐⭐⭐ (解决LLM隐私保护的实际问题，方案具有落地前景)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Privacy-R1: Privacy-Aware Multi-LLM Agent Collaboration via Reinforcement Learning](../../ACL2026/llm_safety/privacy-r1_privacy-aware_multi-llm_agent_collaboration_via_reinforcement_learnin.md)
- [\[AAAI 2026\] RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-Wave Point Cloud Sequence](radarllm_empowering_large_language_models_to_understand_human_motion_from_millim.md)
- [\[AAAI 2026\] Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering](privacy-protected_retrieval-augmented_generation_for_knowledge_graph_question_an.md)
- [\[NeurIPS 2025\] CryptoMoE: Privacy-Preserving and Scalable Mixture of Experts Inference via Balanced Expert Routing](../../NeurIPS2025/llm_safety/cryptomoe_privacy-preserving_and_scalable_mixture_of_experts_inference_via_balan.md)
- [\[ACL 2026\] SharedRequest: Privacy-Preserving Model-Agnostic Inference for Large Language Models](../../ACL2026/llm_safety/sharedrequest_privacy-preserving_model-agnostic_inference_for_large_language_mod.md)

</div>

<!-- RELATED:END -->
