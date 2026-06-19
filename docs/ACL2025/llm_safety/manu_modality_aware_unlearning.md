---
title: >-
  [论文解读] Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models
description: >-
  [ACL 2025 (Long Paper)][LLM安全][机器遗忘] 提出 MANU——首个模态感知的 MLLM 遗忘框架，通过四种互补的神经元重要性函数（绝对/频率/方差/RMS）识别跨模态纠缠的知识载体神经元，选择性剪枝 top-α% 神经元实现多模态和纯文本输入下的均衡遗忘，无需任何梯度更新。
tags:
  - "ACL 2025 (Long Paper)"
  - "LLM安全"
  - "机器遗忘"
  - "MLLM"
  - "神经元剪枝"
  - "模态感知"
  - "隐私保护"
---

# Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2502.15910](https://arxiv.org/abs/2502.15910)  
**代码**: [GitHub](https://github.com/franciscoliu/MANU)  
**领域**: AI安全 / 多模态VLM  
**关键词**: 机器遗忘, MLLM, 神经元剪枝, 模态感知, 隐私保护

## 一句话总结

提出 MANU——首个模态感知的 MLLM 遗忘框架，通过四种互补的神经元重要性函数（绝对/频率/方差/RMS）识别跨模态纠缠的知识载体神经元，选择性剪枝 top-α% 神经元实现多模态和纯文本输入下的均衡遗忘，无需任何梯度更新。

## 研究背景与动机

**MLLM 训练在海量数据上不可避免地记忆敏感信息。** LLM 和 MLLM 的卓越能力源于大规模数据的预训练和微调，但这也带来隐私泄露和版权侵犯的风险。从头重新训练排除敏感数据计算成本过高，因此机器遗忘（Machine Unlearning）成为高效替代方案。

**现有 LLM 遗忘方法在 MLLM 上出现严重的模态不平衡问题。** Liu et al. (2024e) 揭示了一个关键发现：将 LLM 的遗忘方法（如 Gradient Ascent、Gradient Difference）直接应用到 MLLM 时，多模态输入（图+文）下的知识被成功遗忘，但纯文本输入下的同一知识仍然保留。例如，模型可能忘记了"看到这个人的照片后说出其名字"，但仍然能通过纯文本描述回答"这个人是谁"。

**根本原因在于模态纠缠的知识表示。** 不同模态的输入激活不同的神经元子集——多模态遗忘只影响了处理图文联合输入的神经元路径，而处理纯文本输入的神经元路径未被触及。热力图可视化清楚地展示了这种模态特异性的激活模式。

**MANU 的核心 idea：通过模态感知的神经元分析和选择性剪枝，同时移除两条模态路径上与目标知识相关的神经元。** 这是首个专门为 MLLM 跨模态遗忘设计的框架，且完全不需要梯度更新（training-free）。

## 方法详解

### 整体框架

MANU 分两阶段：**阶段一（重要神经元选择）**——将 forget set 和 retain set 分别转换为纯文本和多模态两种格式，前向传播收集所有 MLP 层的神经元激活统计，用四种重要性函数评估每个神经元的模态特异性贡献；**阶段二（选择性剪枝）**——用评分函数 $S_n$ 综合 forget vs retain 的相对重要性，剪枝 top-α% 神经元（权重置零）。

### 关键设计

1. **四种模态感知神经元重要性函数**:
    - 功能：从四个互补角度评估每个神经元在不同模态下的行为差异
    - 核心思路：
        - **绝对重要性** $I_{\text{abs}} = \frac{|\bar{Z}_{\text{multi}} - \bar{Z}_{\text{text}}|}{\bar{Z}_{\text{multi}} + \bar{Z}_{\text{text}} + \epsilon}$——衡量激活幅度的模态差异（归一化），捕获哪些神经元在不同模态下有截然不同的激活强度
        - **频率重要性** $I_{\text{freq}} = \frac{|N_{\text{multi}} - N_{\text{text}}|}{N_{\text{multi}} + N_{\text{text}} + \epsilon}$——衡量神经元在不同模态下被激活（超过阈值 $\tau$）的频率差异，捕获一致性而非幅度
        - **方差重要性** $I_{\text{var}} = \sqrt{\text{Var}_{\text{multi}} + \text{Var}_{\text{text}}}$——基于信息论原则，激活模式多样的神经元携带更多信息
        - **RMS 重要性** $I_{\text{rms}} = \sqrt{\frac{|\Delta Z^2|}{\Sigma Z^2 + \epsilon}}$——识别持续强激活且模态特异的神经元，过滤冗余的"无差别激活"神经元
    - 设计动机：单一指标不足以全面捕获模态特异性——幅度、频率、多样性、持续强度四个维度互补。四种函数的加和 $\mathcal{I}(\mathcal{D}, n) = \sum_{k \in \mathcal{K}} I_k(\mathcal{D}, n)$ 构成综合重要性度量

2. **相对重要性评分（Forget vs Retain）**:
    - 功能：确保剪枝的神经元主要服务于遗忘知识而非保留知识
    - 核心思路：评分函数取 forget 和 retain 重要性的比值：$S_n = \frac{\mathcal{I}(\mathcal{D}_f, n)}{\mathcal{I}(\mathcal{D}_r, n) + \epsilon}$。高分意味着该神经元对遗忘数据远比对保留数据重要
    - 设计动机：如果仅看绝对重要性，可能误剪与通用知识相关的神经元。比值设计确保"精准手术"——只移除服务于特定遗忘目标的神经元

3. **选择性剪枝——权重置零**:
    - 功能：将 top-α% 评分最高的神经元权重设为零，应用于语言和视觉 MLP 层
    - 核心思路：选择集合 $\mathcal{N} = \{n : S_n \text{ is among top } \alpha\%\}$，设置 $\theta' = 0$ if $n \in \mathcal{N}$
    - 设计动机：权重置零是最简单的剪枝方式，无需梯度更新——整个过程仅需一次前向传播收集激活统计

### 损失函数 / 训练策略

**完全无需训练（training-free）。** 仅需一次前向传播收集 forget set 和 retain set 的激活统计，然后执行剪枝。在 LLaVA-1.5-7B 和 Idefics2-8B 上验证，使用 3 张 NVIDIA A6000 GPU。

## 实验关键数据

### 主实验——MLLMU-Bench 上遗忘效果（LLaVA-1.5-7B, 5% Forget）

| 方法 | Forget 分类准确率 ↓ | Forget ROUGE ↓ | Retain 分类准确率 ↑ | Retain ROUGE ↑ | Real Celebrity 分类 ↑ |
|------|-------------------|---------------|-------------------|---------------|---------------------|
| Vanilla（无遗忘） | 51.70% | 0.645 | 46.11% | 0.632 | 51.80% |
| GA | 44.40% | 0.485 | 39.09% | 0.495 | 45.56% |
| Grad. Diff. | 43.60% | 0.507 | 41.07% | 0.508 | 46.52% |
| NPO | 45.61% | 0.525 | 42.61% | 0.515 | 49.51% |
| **MANU** | **41.25%** | **0.491** | **43.38%** | **0.542** | **49.57%** |

MANU 在遗忘效果（Forget集↓）和保留能力（Retain/Celebrity集↑）之间取得最佳平衡。

### 消融实验

| 配置 | Forget 效果 | Retain 保留 | 说明 |
|------|-----------|------------|------|
| 仅 $I_{\text{abs}}$ | 有效 | 较好 | 单一指标也有效 |
| 仅 $I_{\text{freq}}$ | 有效 | 较好 | 一致性视角互补 |
| 四种组合 | **最优** | **最优** | 多维度互补最全面 |
| α=1% | 遗忘不充分 | 保留最好 | 剪枝太少 |
| α=3% | **最优平衡** | 较好 | 最苋区间 |
| α=10% | 遗忘最强 | 损害通用能力 | 剪枝过多 |

### 关键发现

- **模态不平衡问题确认**：热力图明确可视化——GA/Gradient Difference 在多模态输入下遗忘有效（图 2b 颜色浅），但在纯文本输入下知识仍然保留（图 2a 颜色深）
- GA 类方法虽然遗忘效果有时优于 MANU，但以严重损害模型通用能力为代价——在 Retain Set 和 Real Celebrity Set 上性能大幅下降
- 最优剪枝比例 α 在 1-5% 之间，过多剪枝损害通用能力
- MANU 在 LLaVA-1.5-7B 和 Idefics2-8B 上表现一致，具备跨模型泛化能力
- MMMU 和 LLaVA-Bench 上 MANU 的通用能力下降最小

## 亮点与洞察

- **揭示并系统化了 MLLM 遗忘的新问题**：模态不平衡以前未被正式研究，热力图可视化提供了直观证据
- **完全无需训练的遗忘方法**：仅一次前向传播 + 统计 + 剪枝，极低计算成本
- **四种重要性函数的互补设计**：从幅度/频率/方差/RMS 四个维度全面捕获模态特异性
- **Forget/Retain 比值评分**：精准手术式剪枝，最小化对保留知识的误伤

## 局限与展望

- 仅在虚构人物遗忘场景验证，未测试概念级遗忘（如特定技能或知识域）
- 剪枝（权重置零）是粗粒度操作，更精细的权重修改（如缩放/衰减）可能更优
- 仅在 7B/8B 模型上验证，更大模型中神经元的模态角色分工可能不同
- 四种重要性函数的权重为等权加和，未探索自适应权重或学习权重
- 未探讨遗忘鲁棒性——遗忘后模型能否通过微调重新学会被遗忘的知识

## 相关工作与启发

- **vs Gradient Ascent (GA)**：通过反向梯度遗忘；在 MLLM 中导致模态不平衡且损害通用能力
- **vs Gradient Difference**：GA 改进版加入 retain set 梯度；仍出现模态不平衡
- **vs NPO (Negative Preference Optimization)**：将遗忘框架化为偏好优化；相对稳定但不如 MANU 平衡
- **vs 系统提示 (Prompting)**：简单提示可部分防止输出敏感信息，但知识仍存储在模型参数中
- **启发**：不同模态激活不同神经元的发现可用于模态特定的模型压缩——如果知晓哪些神经元专门处理视觉/文本，可做模态解耦的高效部署

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个模态感知的 MLLM 遗忘框架，四重要性函数设计系统全面
- 实验充分度: ⭐⭐⭐⭐ 双模型验证（LLaVA/Idefics2）、多种遗忘比例、详细消融和热力图可视化
- 写作质量: ⭐⭐⭐⭐ 动机通过热力图直观论证，方法公式化清晰
- 价值: ⭐⭐⭐⭐ 对 AI 安全和隐私保护有直接应用价值，揭示的模态不平衡现象有认知意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models](mmunlearner_reformulating_multimodal_machine_unlearning_in_the_era_of_multimodal.md)
- [\[AAAI 2026\] Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models](../../AAAI2026/llm_safety/cross-modal_unlearning_via_influential_neuron_path_editing_i.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)
- [\[NeurIPS 2025\] PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning](../../NeurIPS2025/llm_safety/pulse_practical_evaluation_scenarios_for_large_multimodal_model_unlearning.md)
- [\[ACL 2025\] ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging](zjuklab_at_semeval-2025_task_4_unlearning_via_model_merging.md)

</div>

<!-- RELATED:END -->
