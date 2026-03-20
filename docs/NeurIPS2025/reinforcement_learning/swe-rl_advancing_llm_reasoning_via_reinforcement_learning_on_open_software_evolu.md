# SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution

**会议**: NeurIPS 2025  
**arXiv**: [2502.18449](https://arxiv.org/abs/2502.18449)  
**代码**: [facebookresearch/swe-rl](https://github.com/facebookresearch/swe-rl)  
**领域**: LLM 强化学习 / 软件工程  
**关键词**: 强化学习, 软件演化数据, GRPO, SWE-bench, 推理能力泛化, 代码编辑, Pull Request

## 一句话总结

首次将强化学习 (RL) 应用于真实世界软件工程任务（GitHub PR/Issue 修复），仅用基于规则的序列相似度奖励训练 Llama-3.3-70B，在 SWE-bench Verified 上达到 41.0% 解决率（中等规模模型 SOTA），且 RL 训练仅在 issue-solving 数据上进行，却涌现出在代码推理、数学、通用语言理解等域外任务上的泛化推理能力。

## 研究背景与动机

1. **SWE-bench 是真实世界 SE 的核心挑战**：SWE-bench 要求模型解决真实 GitHub issue，涉及完整仓库理解、bug 定位、patch 生成，远超竞赛编程难度。现有方法严重依赖 GPT-4o / Claude-3.5 等闭源模型，开源模型表现落后。

2. **DeepSeek-R1 证明 RL 可增强推理**：DeepSeek-R1 通过 RL + 规则奖励显著提升了 LLM 在竞赛编程和数学上的推理能力，但其 671B 参数规模难以复现，且在 SE 任务上的效果有限。

3. **现有 RL 训练依赖执行反馈，难以扩展到 SE**：竞赛编程中可用测试用例做执行反馈，但真实 SE 任务的代码执行环境搭建成本极高（依赖复杂、环境不一致），执行反馈难以规模化获取。

4. **已有开源 SE 模型均依赖 SFT + 闭源蒸馏**：Lingma-SWE-GPT、SWE-Gym、SWE-Fixer 等均在训练数据中混入 GPT-4o 或 Claude-3.5-Sonnet 的蒸馏输出，且全部基于 SFT，没有探索 RL 路径。

5. **SFT 的局限性**：SFT 将模型拟合到特定任务分布上，容易在域外任务（如数学、通用理解）上性能下降，且需要精心设计数据配比来维持泛化性。

6. **软件演化数据是天然的 RL 训练场**：GitHub 上有海量高质量 PR 数据，每个 PR 包含 issue 描述、代码上下文和 oracle patch，可构造自然的 RL 训练信号而无需执行环境。

## 方法详解

### 整体框架

SWE-RL 整体流程：(1) 从 GitHub 收集 273K 高质量 PR 作为 seed 数据集；(2) 每条数据包含 issue 描述 + 代码上下文（完整文件内容）+ oracle patch；(3) 策略 LLM 通过推理生成 search/replace 格式的代码编辑；(4) 使用基于规则的奖励计算相似度分数；(5) 通过 GRPO 算法优化策略。

### 关键设计

- **轻量级规则奖励**：不依赖代码执行，而是用 Python `difflib.SequenceMatcher` 计算预测 patch 与 oracle patch 的序列相似度（0~1 的连续值）。格式错误的输出直接给 -1 惩罚。这种连续奖励能捕捉"部分正确"，优于离散的精确匹配奖励。
- **完整文件上下文**：输入 prompt 包含相关文件的完整内容，隐式迫使模型先推理故障定位（在哪个位置改），再生成修复编辑，同时教会模型诊断和修复两种能力。
- **PR seed 筛选启发式**：要求 PR 至少关联一个 issue、issue 描述 bug 修复请求、代码变更涉及编程文件，以确保数据质量。
- **格式约束**：模型输出需为 search/replace 格式的代码编辑（沿用 Agentless 格式），格式错误直接 -1 惩罚促使模型快速学会格式规范。

### 训练策略

- **基座模型**: Llama-3.3-70B-Instruct
- **优化算法**: GRPO (Group Relative Policy Optimization)，每个 batch 32 个问题，每个问题采样 16 个 rollout（group size G=16），全局 batch size 512
- **上下文窗口**: 16K tokens
- **训练步数**: 1600 步，512 张 H100 GPU，约 32 小时 wall-time
- **推理框架 Agentless Mini**：基于 Agentless 简化——仅做文件级定位，将详细推理留给修复步骤。支持多个 reproduction test 的扩展采样和重排序

## 实验关键数据

### 主实验：SWE-bench Verified (pass@1)

| 模型 | 脚手架 | 解决率 |
|------|--------|--------|
| GPT-4o | Agentless | 38.8% |
| o1-preview | Agentless | 41.3% |
| DeepSeek-R1 (671B) | Agentless | 49.2% |
| Claude-3.5-Sonnet | OpenHands | 53.0% |
| SWE-Fixer-72B | SWE-Fixer | 32.8% |
| Llama3-SWE-SFT-70B | Agentless Mini | 36.2% |
| **Llama3-SWE-RL-70B** | **Agentless Mini** | **41.0%** |

→ 中等规模 (<100B) 模型 SOTA，与 GPT-4o (38.8%) 和 o1-preview (41.3%) 相当，显著超越所有开源同级别模型。

### 消融：修复能力对比 (oracle 文件定位，贪心解码)

| 模型 | 格式正确率 | 修复成功率 |
|------|------------|------------|
| Llama-3.3-70B-Instruct (贪心) | 12.2% | 5.4% |
| Llama-3.3-70B-Instruct (多数投票) | 44.6% | 16.6% |
| Llama3-SWE-SFT-70B | 96.2% | 29.6% |
| **Llama3-SWE-RL-70B** | **95.6%** | **34.8%** |

→ RL 模型修复性能大幅超越 SFT (+5.2%)，基座模型格式正确率仅 12.2%。

### 域外泛化能力 (零样本贪心解码)

| 任务 | Llama-3.3-70B | SFT-70B | **RL-70B** |
|------|---------------|---------|------------|
| HumanEval+ | 76.2 | 73.2 | **79.9** |
| CRUXEval-I | 60.5 | 68.4 | **71.6** |
| CRUXEval-O | 61.9 | 75.1 | **75.5** |
| MATH (strict) | 63.2 | 54.0 | **73.7** |
| MMLU | 86.49 | 85.26 | **86.82** |

→ RL 在 5 个域外任务上全面超越基座模型和 SFT；SFT 在 MATH/HumanEval 等任务上性能下降。

### 关键发现

- **连续奖励 vs 离散奖励**：连续奖励（34.8%）显著优于离散精确匹配奖励（29.0%），因为真实 patch 高度多样，精确匹配信号过于稀疏
- **采样扩展**：从 20 个 repair 样本扩展到 160 个带来显著提升（33.6%→40.0%），160 以后边际递减
- **Aha moment 涌现**：RL 训练后模型自发出现自我反思、探索替代方案、分治策略等推理行为

## 亮点与洞察

- 🔑 **第一个将 RL 用于真实 SE 任务的工作**：证明不需要执行环境，仅用 patch 相似度就能有效训练，极大降低了 RL 在 SE 领域的应用门槛
- 🔑 **RL > SFT 的明确证据**：同一基座、同源数据，RL 不仅在域内超越 SFT，更在域外任务全面领先，而 SFT 甚至导致平均性能下降
- 🔑 **推理能力的跨域涌现**：仅在 issue-solving 上做 RL，就涌现出数学推理 (+10.5)、代码推理 (+11.1)、通用理解等能力的提升，印证了 DeepSeek-R1 的 aha moment 发现可推广到 SE 领域
- 🔑 **不依赖闭源模型蒸馏**：训练数据完全来自公开 PR，不包含任何 GPT-4o/Claude 蒸馏数据，在同级模型中是唯一的"自主进化"路径
- 🔑 **连续奖励 > 离散奖励**：精巧的奖励设计证明部分正确的梯度信号对 RL 学习至关重要

## 局限性 / 可改进方向

1. **奖励函数基于序列相似度而非语义等价**：可能惩罚功能等价但表达不同的 patch，限制了策略探索语义正确的多样解法
2. **文件级定位过于简化**：Agentless Mini 仅做文件级定位，缺乏函数/行级精细定位，大文件场景下效率受限
3. **Pipeline 架构限制整体推理**：各步骤独立推理，模型无法从交互反馈中学习，也无法全局统筹考虑定位-修复-验证的关联
4. **训练成本仍然较高**：512 张 H100 训练 32 小时，对学术界仍是高门槛
5. **未探索更小模型**：仅实验了 70B 模型，7B/13B 级别的 SWE-RL 效果尚不清楚
6. **缺少与 agentic 方法的融合**：Pipeline-based 与 agent-based 能否通过 RL 统一尚未探索

## 相关工作与启发

- **DeepSeek-R1**：RL + 规则奖励提升推理能力的先驱，本文将其从竞赛编程/数学推广到真实 SE 任务
- **Agentless**：Pipeline-based SE 工具的代表，本文基于其简化版 (Agentless Mini) 做推理框架
- **GRPO (DeepSeekMath)**：本文采用的 RL 优化算法，通过组内标准化计算优势函数，无需 critic 网络
- **Magicoder (OSS-Instruct)**：SFT baseline 的数据生成灵感来源，但本文证明 RL 路径更优
- **SWE-Gym / SWE-Fixer / Lingma-SWE-GPT**：均为 SFT + 闭源蒸馏路线的代表，SWE-RL 在不依赖蒸馏的情况下超越它们
- **启发**：为大规模软件演化数据（commits、PRs、code reviews）驱动 LLM 自主进化开辟了新范式，暗示"学习修 bug 的过程"本身就是一种高效的通用推理训练

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将 RL 方法成功应用于真实 SE 任务，开辟全新方向
- 实验充分度: ⭐⭐⭐⭐ — 主实验 + 消融 + 扩展分析 + 域外泛化全面，但缺少错误条分析
- 写作质量: ⭐⭐⭐⭐⭐ — 结构清晰，动机充分，图表信息量大
- 价值: ⭐⭐⭐⭐⭐ — 对 SE + RL 社区有重要推动作用，证明了软件数据驱动推理能力涌现的可行性
