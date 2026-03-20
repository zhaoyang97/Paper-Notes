# Reinforcement Learning for Long-Horizon Multi-Turn Search Agents

**会议**: NeurIPS 2025 Workshop  
**arXiv**: [2510.24126](https://arxiv.org/abs/2510.24126)  
**代码**: 无  
**领域**: Agent  
**关键词**: RL agent, multi-turn search, legal document retrieval, GRPO, tool use

## 一句话总结
展示 RL 训练的 14B 参数搜索 agent 在法律文档检索任务上通过多轮交互可以超越 frontier 模型（85% vs GPT o3 的 81%），关键在于精心设计的分段奖励结构和允许长 horizon 多轮交互。

## 研究背景与动机

1. **领域现状**：LLM agent 在工具使用和多步推理上展现了强大能力。多轮文档搜索是一个复杂的长 horizon 交互任务——agent 需要通过多轮搜索来定位特定信息。
2. **现有痛点**：(1) Prompt-based 方法虽然性能不错，但没有从经验中学习的能力；(2) 朴素 RAG（单次检索）在这类任务上效果很差（33%）；(3) 工具使用能力本身不足以获得好效果——base Qwen3-14B 有同样的工具但只得 53%。
3. **核心矛盾**：工具访问 ≠ 有效使用工具。agent 需要学会如何有效利用多轮交互机会来逐步缩小搜索范围。
4. **本文要解决什么？** 用 RL 训练 agent 学会在多轮交互中有效使用搜索工具。
5. **切入角度**：构建法律文档搜索 benchmark，设计分段奖励结构（分别奖励找到正确文档、正确引用、承认不知道，惩罚幻觉和格式错误），用 GRPO 训练 LoRA adapter。
6. **核心 idea 一句话**：通过精心设计的分段奖励和 GRPO 训练，14B 模型学会了有效利用多轮搜索交互，超越了 frontier 模型。

## 方法详解

### 整体框架
构建法律文档搜索 benchmark（2300 QA对） -> agent 拥有三个工具（关键词搜索/语义搜索/阅读文档内容） -> RL 训练（GRPO + 分段奖励）-> 评估不同 turn 限制下的性能。

### 关键设计

1. **三工具 agent 架构**:
   - 做什么：提供互补的文档搜索能力。
   - 核心思路：关键词搜索（BM25）返回文本片段 + section ID；语义搜索（FAISS + MiniLM-L6-v2 embeddings）返回概念匹配结果；阅读文档内容（根据 section ID 返回完整内容，ID 的层级结构支持导航：A:B:C -> A:B 可以向上跳转）。
   - 设计动机：两阶段搜索模式——先用关键词/语义搜索广泛探索，再用阅读工具深入提取。

2. **分段奖励设计**:
   - 做什么：为 RL 提供精细的学习信号。
   - 核心思路：[1.0, 2.0] 正确答案+正确引用（更少 turn/search 得更高奖励）；[0.0, 1.0] 模型回答"不知道"（比幻觉好）；[-1.0, 0.0] 错误答案（找到正确文档仍给 +0.1 部分奖励）；[-2.0, -1.0] 格式错误（无法执行工具调用）。
   - 设计动机：渐进式奖励让即使失败的 trajectory 也能提供学习信号。效率 bonus 鼓励更少搜索次数完成任务。**关键**：惩罚幻觉比承认不知道更重——训练模型在证据不足时说"不知道"。

3. **Turn-restricted 评估**:
   - 做什么：量化多轮交互对性能的影响。
   - 核心思路：在 turn N 时强制插入 <answer> 前缀迫使模型回答。0-turn 等价于朴素 RAG。
   - 设计动机：理解 agent 如何利用额外的搜索机会，以及 RL 训练如何改变这种利用能力。

### 损失函数 / 训练策略
GRPO (Group Relative Policy Optimization)。Base model: Qwen3-14B + LoRA adapter。Reward model: Gemini 2.5 Pro 做二元质量判断。group_size=6, 8 groups per step。YaRN 扩展 context 到 128K tokens。

## 实验关键数据

### 主实验

| 模型 | 准确率 | 平均 Turns |
|------|--------|-----------|
| Naïve RAG (Gemini 2.5 Pro) | 33% | 1.0 |
| Qwen3-14B (base) | 53% | 3.7 |
| Gemini 2.5 Flash | 66% | 3.4 |
| Gemini 2.5 Pro | 78% | 5.3 |
| OpenAI o3 | 81% | 7.1 |
| **Qwen3-14B + RL** | **85%** | **6.2** |

### Turn 限制分析

| 分析 | 发现 |
|------|------|
| Base Qwen3-14B | 6 turns 后性能饱和 |
| RL-trained Qwen3-14B | 10 turns 仍在提升 |
| Gemini 2.5 Pro | 10 turns 仍在提升 |
| 训练时限制 turns | 限制到 4 turns 训练的 agent 在 10 turns 推理时也利用不好后续 turns |

### 关键发现
- **14B RL 模型超越所有 frontier 模型（85% vs o3 的 81%）**——工具使用能力可以通过 RL 从小模型中"挤出来"。
- **工具访问 ≠ 有效工具使用**：没有 RL 训练的 Qwen3-14B 只有 53%，有了 RL 后跃升到 85%。
- **RL agent 更善于利用多轮交互**：base 模型 6 turns 后饱和，RL 模型 10 turns 仍在提升——RL 学会了"不急于回答"和"有计划地搜索"。
- **训练时的 turn 限制影响推理时的利用能力**：必须在训练时给予足够的 turn budget。

## 亮点与洞察
- **"工具访问 ≠ 有效工具使用"** 是该论文最重要的洞察：同样的工具，RL 训练的模型多出 32 个百分点。
- **分段奖励设计**极具参考价值："不知道" > 幻觉的价值排序应该成为所有搜索 agent 的标准。
- **小模型 + RL 可以超越大模型**：展示了在特定任务上，RL 训练的专家模型比通用 frontier 模型更强的路线。

## 局限性 / 可改进方向
- **单一法律领域**：未验证在其他领域的泛化性。
- **依赖 Gemini 2.5 Pro 作为 reward model**：高成本，且可能引入偏见。
- **训练数据由 LLM 生成**：QA 对的质量和多样性受限于生成模型。
- **Workshop paper**：实验规模相对有限。

## 相关工作与启发
- **vs 朴素 RAG**: 单次检索(33%) vs 多轮交互(85%)，差距巨大，说明多轮交互对复杂检索任务至关重要。
- **vs Prompt-based agents**: Prompt agent 可以达到不错效果（Gemini Pro 78%），但 RL 可以进一步推高性能。
- **vs Chain-of-Retrieval**: 类似思路但用 RL 学习最优检索策略，而非预设流程。

## 评分
- 新颖性: ⭐⭐⭐⭐ RL 训练多轮搜索 agent 的实证研究，turn 限制分析新颖
- 实验充分度: ⭐⭐⭐ Workshop paper 规模，单一领域
- 写作质量: ⭐⭐⭐⭐ 简洁清晰
- 价值: ⭐⭐⭐⭐⭐ 小模型超越 frontier 的结果非常有启发性
