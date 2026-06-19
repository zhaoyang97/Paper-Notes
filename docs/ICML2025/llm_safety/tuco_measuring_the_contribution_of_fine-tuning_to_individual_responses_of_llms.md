---
title: >-
  [论文解读] TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs
description: >-
  [ICML2025][LLM安全][fine-tuning分析] 提出 Tuning Contribution (TuCo) 指标，通过将微调后 LLM 的前向传播精确分解为预训练分量 (PTC) 和微调分量 (FTC)，首次实现在推理时逐 prompt 量化微调对模型输出的贡献，并揭示越狱攻击通过削弱 FTC 幅度来绕过安全防护。
tags:
  - "ICML2025"
  - "LLM安全"
  - "fine-tuning分析"
  - "可解释性"
  - "越狱攻击"
  - "残差分解"
  - "Transformer"
---

# TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs

**会议**: ICML2025  
**arXiv**: [2506.23423](https://arxiv.org/abs/2506.23423)  
**代码**: [github.com/FelipeNuti/tuning-contribution](https://github.com/FelipeNuti/tuning-contribution)  
**领域**: LLM分析 / AI Safety  
**关键词**: fine-tuning分析, 可解释性, 越狱攻击, 残差分解, Transformer内部表示

## 一句话总结

提出 Tuning Contribution (TuCo) 指标，通过将微调后 LLM 的前向传播精确分解为预训练分量 (PTC) 和微调分量 (FTC)，首次实现在推理时逐 prompt 量化微调对模型输出的贡献，并揭示越狱攻击通过削弱 FTC 幅度来绕过安全防护。

## 研究背景与动机

- **微调效果缺乏细粒度度量**：已有研究仅在数据集级别（如 benchmark 性能、机制可解释性）分析微调对 LLM 的影响，缺少针对单个 prompt 输出进行定量分析的方法。
- **越狱攻击的机制假说未被量化验证**：Wei et al. (2024) 和 Kotha et al. 提出越狱攻击利用预训练与微调目标间的"竞争"关系，但这一假说从未被直接形式化或测量。
- **隐藏状态 vs. 最终输出**：微调可能显著改变中间隐藏状态而不影响最终 token 预测，因此需要考察整个前向传播过程而非仅比较最终输出。

## 方法详解

### 1. 精确分解：PTC 与 FTC

对于具有 $L$ 层残差结构的 Transformer，给定微调模型 $\mathcal{T}^{\text{FT}}_\Theta$ 和对应预训练模型 $\mathcal{T}^{\text{PT}}_\phi$，每层的更新可分解为：

$$\mathbf{x}_{l+1} = \mathbf{x}_l + \underbrace{f^{\text{PT}}_\phi(\mathbf{x}_l, l)}_{\text{PTC}_l} + \underbrace{f^{\text{FT}}_\Theta(\mathbf{x}_l, l) - f^{\text{PT}}_\phi(\mathbf{x}_l, l)}_{\text{FTC}_l}$$

- **PTC (Pre-Training Component)**：预训练模型对应层的输出，代表预训练形成的计算电路
- **FTC (Fine-Tuning Component)**：微调模型与预训练模型在同一层、同一输入上的输出差值，代表微调新增的计算电路

该分解对所有残差结构 Transformer **精确成立**，无需假设已知特定电路分解。

### 2. TuCo 的定义

在所有层上累积最后一个 token 的 FTC 和 PTC：

$$I^{\text{FTC}} = \sum_{l=0}^{L-1} \text{FTC}_l[-1], \quad I^{\text{PTC}} = \sum_{l=0}^{L-1} \text{PTC}_l[-1]$$

Tuning Contribution 定义为：

$$\text{TuCo}(\mathbf{x}) = \frac{\|I^{\text{FTC}}\|}{\|I^{\text{PTC}}\| + \|I^{\text{FTC}}\|}$$

TuCo 取值范围 $[0, 1]$，值越高表示微调对该 prompt 响应的影响越大。

### 3. Grönwall 理论界

论文证明了离散 Grönwall 界：当 PTC 有界且 Lipschitz 时，

$$\|\mathbf{x}^{\text{FT}}_L - \mathbf{x}^{\text{PT}}_L\|_1 \leq L \|\text{PTC}\|_{\sup} (1 + \|\text{PTC}\|_{\text{Lip}})^L \frac{\beta}{1-\beta}$$

其中 $\beta = \max_l \frac{\|\overline{\text{FTC}}_l\|_1}{\|\overline{\text{PTC}}_l\|_1 + \|\overline{\text{FTC}}_l\|_1}$，理论上保证 FTC 较小时微调模型输出接近预训练模型。

### 4. FTC α-Scaling

通过缩放因子 $\alpha$ 调控微调分量的幅度：

$$\mathbf{x}_{l+1} = \mathbf{x}_l + \text{PTC}(\mathbf{x}_l, l) + \alpha \cdot \text{FTC}(\mathbf{x}_l, l)$$

$\alpha=1$ 恢复微调模型，$\alpha=0$ 近似回到预训练模型的行为。

## 实验关键数据

**模型覆盖**：Llama 2 (7B/13B)、Llama 3 (8B)、Gemma 7B、Vicuna v1.5 (7B/13B)、Mistral (V0.1/V0.2 7B)、Zephyr Gemma 7B，共 9 个开源模型。

### 实验1：FTC α-Scaling 控制模型行为

| 实验 | 指标 | 结果 |
|------|------|------|
| MMLU (57任务) | 最优 α 下准确率提升 | 1.03%–2.69% (71%任务显著) |
| MWE 行为评估 | 最大化行为一致性 | +1.55%–5.18% (所有模型显著) |
| MWE 行为评估 | 最小化行为一致性 | -2.80%~-25.24% |
| 基督教信仰认同 (Llama2 13B) | α=1.25 vs α=1.0 | +24% 认同度 |

### 实验2：Web 文本 vs. Chat 输入的 TuCo 区分度

| 模型 | AUC (OpenWebText vs HH-RLHF) |
|------|------|
| Llama 2 7B/13B | 1.00 |
| Vicuna 7B/13B | 0.99 |
| Gemma 7B | 0.93 |
| Llama 3 8B | 1.00 |

### 实验3：越狱攻击降低 TuCo

| 攻击类型 | 模型 | AUC (有攻击 vs 无攻击) |
|------|------|------|
| GCG 梯度攻击 | Llama 2 7B | 1.00 |
| GCG 梯度攻击 | Llama 2 13B | 0.80 |
| 共轭提示 (En vs Ml/Sw) | Llama 2 13B | 1.00 |
| Many-Shot | 所有模型 | TuCo 随 shot 数单调递减 |

### 实验4：成功越狱的 TuCo 更低

| 模型 | 成功越狱 AUC | vanilla 越狱率 | GCG 越狱率 |
|------|------|------|------|
| Llama 2 13B | **0.87** | 0.19% | 1.1% |
| Llama 2 7B | 0.83 | 0.19% | 16.36% |
| Gemma 7B | 0.94 | 6.92% | 7.42% |
| Vicuna 7B | 0.87 | 29.23% | 85.13% |

### 实验5：TuCo vs OutputCo 的差异

OutputCo 仅比较最终隐藏状态，而 TuCo 考察整个前向传播。在多次拒绝示例后接无害问题的实验中，OutputCo 随示例增多而降低（模型快速学会拒绝），但 TuCo 反而升高（反映内部微调电路活动增强），说明两个指标捕获了不同的信息。

## 亮点与洞察

1. **理论严谨**：从广义分量 (Generalized Component) 的形式化定义出发，证明任何微调 Transformer 都可精确分解为 PTC + FTC，无需假设电路结构。
2. **首个逐 prompt 的微调贡献度量**：推理时可计算，适用于十亿参数级模型，计算开销约为额外一次前向传播。
3. **越狱机制的定量证据**：三种主流攻击（GCG、共轭提示、Many-Shot）均显著降低 TuCo，且**攻击成功时 TuCo 降得更多**（AUC 高达 0.87），直接量化了"越狱=削弱微调效果"这一假说。
4. **低资源语言的 TuCo 排序与网络语料份额完全一致**：英语 > 日语 > 匈牙利语 > 斯瓦希里语/马拉雅拉姆语，揭示微调覆盖度与训练数据分布的直接关系。
5. **FTC α-Scaling 可实际调控模型行为**，在 MMLU 上获得 1–3% 的性能提升，尽管作者强调这不是目的而是验证。

## 局限与展望

1. **需要同时访问预训练和微调模型**：闭源模型（如 GPT-4、Claude）不可用，限制了实际部署场景。
2. **计算开销**：需要同时运行两个模型的前向传播，对大规模部署有一定负担。
3. **TuCo 非攻击检测工具**：虽然 AUC 高，但作者明确声明 TuCo 是分析工具而非防御机制，直接用于实时检测可能存在对抗绕过。
4. **模型规模有限**：仅验证至 13B 参数，对于 70B+ 或 MoE 架构的适用性未知。
5. **对 LoRA 等参数高效微调方法的适用性**：论文未专门讨论 LoRA/QLoRA 等方法下 FTC 的特性。
6. **因果性 vs 相关性**：TuCo 降低与越狱成功高度相关，但尚未建立因果关系。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首个逐 prompt 量化微调贡献的指标，理论推导完整
- 实验充分度: ⭐⭐⭐⭐ — 9 个模型 × 三种攻击 × 多种评估任务，消融充分但模型规模有限
- 写作质量: ⭐⭐⭐⭐⭐ — 理论动机→形式化→实验验证的逻辑链清晰严谨
- 价值: ⭐⭐⭐⭐ — 为 LLM 安全和可解释性提供新维度，但依赖预训练模型可用性限制了实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach](../../ACL2025/llm_safety/towards_context-robust_llms_a_gated_representation_fine-tuning_approach.md)
- [\[ICML 2025\] Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning](invariance_makes_llm_unlearning_resilient_even_to_unanticipated_downstream_fine-.md)
- [\[ACL 2025\] From Misleading Queries to Accurate Answers: A Three-Stage Fine-Tuning Method for LLMs](../../ACL2025/llm_safety/from_misleading_queries_to_accurate_answers_a_three-stage_fine-tuning_method_for.md)
- [\[ICML 2025\] Vulnerability-Aware Alignment: Mitigating Uneven Forgetting in Harmful Fine-Tuning](vulnerability-aware_alignment_mitigating_uneven_forgetting_in_harmful_fine-tunin.md)
- [\[ICLR 2026\] Be Careful When Fine-tuning On Open-Source LLMs: Your Fine-tuning Data Could Be Secretly Stolen!](../../ICLR2026/llm_safety/be_careful_when_fine-tuning_on_open-source_llms_your_fine-tuning_data_could_be_s.md)

</div>

<!-- RELATED:END -->
