# AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play

**会议**: NeurIPS 2025  
**arXiv**: [2509.24193](https://arxiv.org/abs/2509.24193)  
**代码**: [GitHub](https://github.com/ritaranx/AceSearcher/) / [HuggingFace](https://huggingface.co/AceSearcher)  
**领域**: LLM Agent / LLM推理  
**关键词**: 搜索增强LLM, 多跳推理, 问题分解, 自我博弈, 迭代DPO  

## 一句话总结
提出 AceSearcher——一种协作式自我博弈框架，让单个 LLM 同时扮演**问题分解者**（将复杂查询拆解为子问题引导检索）和**求解者**（整合检索上下文生成答案），通过 SFT + 迭代 DPO 两阶段训练，仅用最终答案作为奖励信号，在 10 个数据集上平均 EM 提升 7.6%，32B 模型匹配 DeepSeek-V3（<5% 参数）。

## 研究背景与动机

1. **领域现状**：RAG 已成为弥补 LLM 知识缺陷的标准方法，但大多数 RAG 方法只处理简单的单跳检索问题。
2. **现有痛点**：复杂推理任务需要 (a) 多跳检索收集多个证据片段，(b) 推理整合多信息生成答案。现有方法要么依赖强闭源 LLM 做多轮 prompting（成本高），要么用在线 RL 训练（内存密集），要么仅在 QA 数据上监督（泛化受限）。
3. **核心矛盾**：多跳检索需要"知道该搜什么"（问题分解能力），而搜到结果后还需要推理能力——两者耦合但现有方法分开处理。
4. **切入角度**：让一个模型同时学两个角色——分解者和求解者，通过协作式自我博弈互相提升。
5. **核心 idea 一句话**：问题分解者的好坏由求解者的准确率衡量，求解者的好坏由最终答案匹配度衡量——无需中间步骤标注。

## 方法详解

### 整体框架
给定复杂问题 $q$ → 分解者生成子问题序列 $z = (z_1, ..., z_n)$ → 对每个子问题用检索器获取上下文 $\mathcal{D}_i$ → 求解者逐步生成中间答案 $w_i$ → 最终整合生成答案 $a'$。分解者和求解者是**同一个 LLM**，通过不同 prompt 模板控制角色。

### 关键设计

1. **两角色协作 (Decomposer + Solver)**:
   - 分解者 $\rho$：$z \sim p_\theta(\cdot|q)$，将原始问题拆为子问题模板
   - 求解者 $\pi$：$w_i \sim p_\theta(\cdot|z_i, w_{<i}, \mathcal{D}_i)$，基于子问题+之前答案+检索上下文生成
   - 用一个统一 LLM 实现，通过输入模板区分角色
   - 联合目标：$p_\theta(a|q) = \sum_z p_\theta(z|q) \sum_w p_\theta(a|q,z,w) p_\theta(w|q,z)$

2. **Stage I: SFT 数据混合**:
   - 做什么：建立基础能力——检索使用、问题分解、推理
   - 数据组成（共 180K 样本）：
     - Context-rich QA：NQ, SQuAD, DROP, NarrativeQA 等——学会从上下文提取答案
     - 问题分解：GSM8K, ConvFinQA, StrategyQA——学会拆解多步问题
     - CoT 推理：MathInstruct (CoT + PoT 风格)——增强多步推理能力
   - 设计动机：现有 RAG 微调仅关注答案提取，缺乏分解和推理能力的训练

3. **Stage II: 迭代 DPO 强化微调**:
   - 做什么：在只有最终答案标注的数据上，同时优化两个角色
   - 核心思路：
     - 对每个问题采样 $m$ 个分解方案，每个分解采样 $m'$ 个求解轨迹
     - 根据最终 EM 奖励 $r(q,a',a) = \text{EM}(a',a) \times \mathbb{I}(f(q,a')=1)$ 构建偏好对
     - 分解者：按期望奖励 $\bar{r}(q, z^{(i)})$ 选最好/最差分解
     - 求解者：按最终答案匹配度选最好/最差轨迹
     - 联合偏好集：$\mathcal{D}_{pref} = \mathcal{D}_{decompose} \cup \mathcal{D}_{subq} \cup \mathcal{D}_{final}$
   - 理论保证：Theorem 4.1 证明迭代 DPO 的最小化器收敛到真实最优参数 $\theta^*$
   - 关键优势：不需要中间步骤标注，不需要在线 RL 的高内存开销

4. **多评估环境**:
   - RAG 环境：HotpotQA, 2WikiMHQA, HOVER——需要多跳检索
   - Context-Rich 推理环境：GSM8K, TabMWP, ConvFinQA——需要上下文推理
   - 统一的奖励信号：仅基于最终答案的 EM

### 训练策略
- 骨干模型：Qwen-2.5-Instruct (1.5B/14B/32B) 和 Llama-3.1-8B-Instruct
- 32B 用 LoRA (r=8, α=16)，其余全量微调
- Batch size 64，max tokens 2048，SFT 1 epoch，RFT 2 轮迭代
- 每子问题最多 $\lfloor 15/n \rfloor$ 篇文档

## 实验关键数据

### 主实验 (Multi-hop QA + Fact Verification)

| 方法 | 参数 | 2WikiMHQA | HotpotQA | 平均 |
|------|------|-----------|----------|------|
| Search-R1 | 7B | 基线 | 基线 | 基线 |
| DeepResearcher | 7B | 基线 | 基线 | 基线 |
| **AceSearcher-8B** | 8B | +显著 | +显著 | **+7.6% avg** |
| AceSearcher-1.5B | 1.5B | 超过 9× 参数的方法 | - | 极高效 |
| AceSearcher-32B | 32B | 匹配 DeepSeek-V3 | - | <5% 参数 |

### Document-level Reasoning (DocMath-Eval)

| 方法 | TAT-QA | FinQA | 平均 |
|------|--------|-------|------|
| GPT-4o | 基线 | 基线 | 基线 |
| DeepSeek-V3 | 强 | 强 | 强 |
| **AceSearcher-32B** | **匹配 V3** | **匹配 V3** | **~相当** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 仅 SFT | 基线 | 基线能力 |
| SFT + 1 轮 DPO | +显著 | 强化微调有效 |
| SFT + 2 轮 DPO | +进一步 | 迭代收益 |
| 去除分解数据 | 下降 | 分解训练数据关键 |
| 去除 CoT/PoT 数据 | 下降 | 推理数据关键 |
| 去除 QA 数据 | 下降 | 检索使用能力关键 |

### 关键发现
- **1.5B 匹配 10× 参数模型**：AceSearcher-1.5B 在多跳 QA 上超过多个更大的搜索增强 LLM
- **自我博弈产生正反馈循环**：更好的分解 → 更好的检索 → 更好的答案 → 更好的奖励信号
- **无需中间标注**：仅用 EM 奖励就能同时训练分解和求解能力
- **迭代 DPO vs 在线 RL**：迭代 DPO 内存友好且效果可比，不需要同时维护 policy + value model

## 亮点与洞察
- **协作式自我博弈的 RAG 应用**：分解者和求解者互相提升——分解者的奖励由求解者的表现决定，求解者的奖励由最终答案决定。这种"mutual bootstrapping"范式可迁移到其他需要规划+执行的场景
- **数据混合的重要性**：SFT 阶段混合检索、分解、推理三类数据，每类都不可或缺（消融实验证实），这个 data recipe 本身就有参考价值
- **不依赖闭源模型**：使用开源模型 + 公开数据集，无需 GPT-4 蒸馏或中间标注
- **实用的迭代 DPO**：相比在线 RL 内存需求小很多，适合资源受限环境

## 局限性 / 可改进方向
- **检索器固定**：使用固定的 E5 / OpenAI Embedding 检索器，未联合训练检索器
- **子问题数量固定**：分配给每个子问题的文档数按 $\lfloor N/n \rfloor$ 均匀分配，不够灵活
- **仅 EM 奖励**：对开放式回答/长文本生成场景不友好
- **改进方向**：(1) 联合训练检索器；(2) 自适应文档分配；(3) 使用 process reward model 替代 EM 奖励

## 相关工作与启发
- **vs Search-R1 / R1-Searcher**：这些方法用在线 RL（GRPO 等），内存密集；AceSearcher 用迭代 DPO，更高效
- **vs IRCOT**：IRCOT 是 prompt-based 的多跳方法，依赖强 LLM；AceSearcher 训练小模型达到更好效果
- **vs Self-RAG**：Self-RAG 用 reflection token 决定何时检索；AceSearcher 显式分解问题指导检索
- **vs DeepResearcher**：DeepResearcher 用 GRPO 训练，AceSearcher 用迭代 DPO + 两角色设计

## 评分
- 新颖性: ⭐⭐⭐⭐ 两角色自我博弈 + 迭代 DPO 的组合新颖，理论收敛保证是加分项
- 实验充分度: ⭐⭐⭐⭐⭐ 3 类任务、10 个数据集、4 种模型规模、大量 baseline 对比、充分消融
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，数学推导完整，实验设计合理
- 价值: ⭐⭐⭐⭐⭐ 提供了高效且通用的搜索增强 LLM 训练方案，1.5B 模型即可实用
