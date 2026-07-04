---
title: >-
  [论文解读] MoRAgent: Parameter Efficient Agent Tuning with Mixture-of-Roles
description: >-
  [ICML2025][模型压缩][Agent Tuning] 提出 Mixture-of-Roles (MoR) 框架，将 Agent 能力分解为推理者、执行者、总结者三个角色，每个角色分配专门的 LoRA 组，以极少额外参数（0.16B–0.36B）实现接近甚至超越全参数微调的 Agent 性能。
tags:
  - "ICML2025"
  - "模型压缩"
  - "Agent Tuning"
  - "parameter-efficient fine-tuning"
  - "LoRA"
  - "Mixture-of-Experts"
  - "Function Calling"
---

# MoRAgent: Parameter Efficient Agent Tuning with Mixture-of-Roles

**会议**: ICML2025  
**arXiv**: [2512.21708](https://arxiv.org/abs/2512.21708)  
**代码**: [https://mor-agent.github.io/](https://mor-agent.github.io/)  
**领域**: 模型压缩  
**关键词**: Agent Tuning, parameter-efficient fine-tuning, LoRA, Mixture-of-Experts, Function Calling

## 一句话总结

提出 Mixture-of-Roles (MoR) 框架，将 Agent 能力分解为推理者、执行者、总结者三个角色，每个角色分配专门的 LoRA 组，以极少额外参数（0.16B–0.36B）实现接近甚至超越全参数微调的 Agent 性能。

## 研究背景与动机

- **Agent 微调的挑战**：现有 Agent 微调工作几乎全部采用全参数微调（full fine-tuning），计算资源开销巨大，同时会破坏原始模型的通用能力，限制了在通用任务与 Agent 任务间灵活切换。
- **PEFT 直接用于 Agent 效果差**：直接对 Agent 任务使用 LoRA 等参数高效方法微调，效果远不如全参数微调。原因在于 Agent 任务需要 LLM 同时具备推理、工具调用执行、对话总结等多种能力，而低秩矩阵难以同时学好这些异质能力。
- **多能力共学的低秩瓶颈**：核心洞察——Agent 任务本质上需要多种截然不同的能力协作，单一 LoRA 的参数空间不足以覆盖这些能力的分布差异。
- **多模型方案的局限**：之前 α-UMi 等工作用多个独立 LLM 分别承担不同角色，但训练和推理的资源开销过高，不实用。

## 方法详解

### 1. 能力分解（Capabilities Decomposition）

受 ReAct（Reason+Action）范式启发，将 Agent 能力分解为三个角色：

- **Reasoner（推理者）**：理解用户查询，分析执行轨迹，生成思考（thought），决定下一步激活哪个角色。形式化为：$T_t, Role_t = \boldsymbol{W}_r(p_r, q, \tau_{t-1})$
- **Executor（执行者）**：根据推理者的分析，选择合适的函数及参数进行调用。形式化为：$Fun_t, Param_t = \boldsymbol{W}_e(p_e, q, \tau_{t-1}, T_{t-1})$
- **Summarizer（总结者）**：在推理者判断任务完成或无法继续时，整理对话历史并向用户反馈。形式化为：$Sum = \boldsymbol{W}_s(p_s, q, \tau)$

三个角色通过规则门控（role-aware gate）交替激活，形成"推理→执行→（观测→推理→…）→总结"的工作流。

### 2. Mixture-of-Roles (MoR) 架构

MoR 框架部署在 Transformer 每层的线性层（注意力/FFN）上，冻结预训练权重，仅训练 LoRA 和路由器。

**前向计算**：给定输入 $\boldsymbol{u} \in \mathbb{R}^{len \times d_1}$，最终输出为：

$$\boldsymbol{h} = \boldsymbol{W}_0 \boldsymbol{u} + \Delta\boldsymbol{h}_r + \Delta\boldsymbol{h}_e + \Delta\boldsymbol{h}_s$$

关键约束：每个 token 仅由一个角色处理，即 $\mathbb{1}\{\Delta\boldsymbol{h}_r^i \neq 0\} + \mathbb{1}\{\Delta\boldsymbol{h}_e^i \neq 0\} + \mathbb{1}\{\Delta\boldsymbol{h}_s^i \neq 0\} = 1$。

**角色内部 LoRA 结构**：每个角色包含一个共享 LoRA 和多个路由 LoRA，以推理者为例：

$$\Delta\boldsymbol{W}_r = \boldsymbol{B}_r^0 \boldsymbol{A}_r^0 + \sum_{i=1}^{E_r} \boldsymbol{B}_r^i \boldsymbol{A}_r^i \boldsymbol{R}_r(\boldsymbol{u}_r)$$

其中 $\boldsymbol{R}_r$ 为 Top-K token-aware 路由器，根据输入动态选择具体 LoRA。

**LoRA 分配策略**：通过实验确定——Reasoner 和 Executor 各使用 5 个 LoRA（Top-4 激活），Summarizer 使用 4 个 LoRA（Top-3 激活），因为总结任务学习难度较低。

### 3. 训练目标

总损失由三部分组成：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha_1 \mathcal{L}_{\text{aux}} + \alpha_2 \mathcal{L}_{\text{orth}}$$

- $\mathcal{L}_{\text{CE}}$：标准交叉熵损失
- $\mathcal{L}_{\text{aux}}$：辅助负载均衡损失（继承自 Switch Transformer），防止 LoRA 间负载不均：$\mathcal{L}_{\text{aux}} = E_\triangledown \cdot \sum_{i=1}^{E_\triangledown} f_\triangledown^i \cdot P_\triangledown^i$
- $\mathcal{L}_{\text{orth}}$：正交损失，鼓励不同 LoRA 学习不同方向的特征分布：$L_{\text{orth}} = \sum_{i}\sum_{j>i} (\|\boldsymbol{A}^{iT}\boldsymbol{A}^j\|_F^2 + \|\boldsymbol{B}^{iT}\boldsymbol{B}^j\|_F^2)$

### 4. 多角色数据生成流水线

- **数据来源**：ToolBench、APIGen+ToolACE、glaive-function-calling-v2、MathGenie 等公开数据集
- **角色内容补全**：用 GPT-4o 补全数据中缺失的 thought 和 summary 内容
- **可靠性验证**：用 DeepSeek-V3 评估轨迹质量，过滤低质量样本；对执行错误（函数不在候选列表、参数数量/类型不匹配、函数选择错误）进行规则检测 + 人工修正
- **格式统一**：将所有数据统一为 JSON 格式，包含候选函数列表、系统提示和执行轨迹

## 实验关键数据

### StableToolBench 结果（DFS Pass Rate）

| 模型 | 额外参数 | I1-Inst | I1-Tool | I1-Cat | I2-Inst | I2-Cat | I3-Inst | AVG |
|------|---------|---------|---------|--------|---------|--------|---------|-----|
| GPT-4 | — | 59.2 | 65.7 | 61.7 | 55.2 | 55.6 | 66.1 | 60.6 |
| ToolLLaMA-v2-7B | — | 61.0 | 45.6 | 58.8 | 53.5 | 60.3 | 48.1 | 54.6 |
| Llama3.2-1B Base | — | 11.7 | 11.3 | 14.6 | 8.5 | 2.3 | 11.5 | 10.0 |
| **MoRAgent-Llama** | **+0.16B** | 54.6 | 45.5 | 53.2 | 46.8 | 68.2 | 58.8 | **54.5 (+44.5)** |
| Phi3.5-mini Base | — | 46.4 | 50.6 | 51.1 | 39.9 | 41.5 | 37.6 | 44.5 |
| **MoRAgent-Phi** | **+0.36B** | 55.9 | 56.6 | 60.9 | 55.4 | 59.7 | 63.6 | **58.7 (+14.2)** |

**核心发现**：
- MoRAgent-Llama 仅加 0.16B 参数，DFS Pass Rate 从 10.0% 飙升至 54.5%（+44.5%），接近 ToolLLaMA-v2-7B（54.6%）的水平，而后者参数量约为其 5.6 倍
- MoRAgent-Phi 在 Phi3.5-mini 上 DFS Pass Rate 提升 14.2 个百分点至 58.7%，超越 ToolLLaMA-v2-7B
- 在 CoT 模式下同样有显著提升：Llama +40.0%，Phi +26.3%

### 关键设计选择

- **LoRA 数量**：实验发现 Reasoner 和 Executor 各用 5 个 LoRA（Top-4）、Summarizer 用 4 个（Top-3）时性能-参数平衡最优
- **LoRA rank**：$d_3 = 16$
- **目标模块**：query, key, value, out, gate, up, down（共 7 个模块）
- **训练超参**：学习率 5e-5，4 epoch

## 亮点与洞察

1. **角色分解的设计直觉**：不同于之前多模型协作的重量级方案（如 α-UMi 用 3 个独立 LLM），MoR 在单一模型内通过多组 LoRA 实现角色分工，训练和推理成本极低。
2. **角色差异化的 LoRA 分配**：通过实验驱动的方式发现不同角色需要不同数量的 LoRA，Summarizer 比 Reasoner/Executor 需要更少的 LoRA——这符合"总结比推理和执行更简单"的直觉。
3. **双层路由机制**：规则级（role-aware gate，按角色类型分配）+ 学习级（token-aware router，在角色内部动态选 LoRA），兼顾了可解释性和灵活性。
4. **正交损失的引入**：显式鼓励同一角色内的不同 LoRA 学习正交方向，减少冗余，提升参数效率。
5. **数据质量控制流程完整**：用两个不同 LLM（GPT-4o 补全 + DeepSeek-V3 过滤）+ 人工修正的流水线，确保训练数据可靠性。

## 局限与展望

1. **角色定义较刚性**：三角色分解（Reasoner/Executor/Summarizer）是预定义的，不同 Agent 任务可能需要不同的角色划分。论文未探讨更灵活或自适应的角色发现机制。
2. **评估基准有限**：主要在 StableToolBench 和 BFCL 上评测，缺少对更复杂多轮交互场景（如 WebArena、OSWorld 等）的验证。
3. **数据构建依赖强 LLM**：多角色数据补全依赖 GPT-4o，可靠性验证依赖 DeepSeek-V3，数据质量受限于这些模型的能力上限，且成本不可忽视。
4. **仅测试小模型**：实验仅在 1B–3.8B 级别模型上验证，未在 7B+ 模型上测试，大模型上是否仍有同等增益尚不确定。
5. **推理延迟未报告**：引入多组 LoRA + 路由器后的实际推理时延增加未被量化分析。
6. **角色切换的错误传播**：Reasoner 判断下一角色的准确性直接影响整个流水线，若判断失误（如过早总结或遗漏执行步骤），会导致任务失败，论文未深入分析这类错误模式。

## 相关工作与启发

- **Agent Tuning 系列**：ToolLLM、AgentTuning、Gorilla 等全参数微调方法是本文的对标基线
- **多 LoRA 组合**：LoRAHub、MoA、MoLE 等多 LoRA MoE 方法提供了技术基础，但未针对 Agent 任务设计
- **多模型 Agent 协作**：α-UMi 的三角色分解（planner/caller/summarizer）是本文的直接灵感来源，MoR 用单模型内多 LoRA 替代多模型，大幅降低成本
- **启发**：角色分解 + 专用参数的思路可以推广到其他多能力任务（如 RAG、多模态 Agent），同一模型内不同能力用不同低秩空间建模

## 评分

- **新颖性**: 7/10 — 角色分解思路借鉴 α-UMi，创新点在于将多角色映射到多组 LoRA 的 MoE 架构上，正交损失和双层路由有增量贡献
- **实验充分度**: 7/10 — 多模型多基准测试 + 消融研究较完整，但缺少大模型和复杂 Agent 场景验证
- **写作质量**: 8/10 — 结构清晰，公式推导完整，图示直观
- **价值**: 7/10 — 为 Agent PEFT 提供了实用方案，小模型上效果显著，但角色定义的刚性限制了泛化性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](parameter-efficient_fine-tuning_of_state_space_models.md)
- [\[CVPR 2025\] Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](../../CVPR2025/model_compression/expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)
- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](../../ACL2025/model_compression/parameter-efficient_fine-tuning_via_circular_convolution.md)
- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](../../ACL2025/model_compression/state_offset_tuning_ssm_peft.md)
- [\[ICCV 2025\] Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations](../../ICCV2025/model_compression/generalized_tensor-based_parameter-efficient_fine-tuning_via_lie_group_transform.md)

</div>

<!-- RELATED:END -->
