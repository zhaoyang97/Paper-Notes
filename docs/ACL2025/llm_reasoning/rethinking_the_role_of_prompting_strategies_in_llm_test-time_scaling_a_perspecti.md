# Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory

**会议**: ACL 2025  
**arXiv**: [2505.10981](https://arxiv.org/abs/2505.10981)  
**代码**: [GitHub](https://github.com/MraDonkey/rethinking_prompting)  
**领域**: LLM推理  
**关键词**: test-time scaling, majority voting, prompting策略, Chain-of-Thought, 概率论分析

## 一句话总结
本文在 6 个 LLM × 8 种 prompting 策略 × 6 个 benchmark 上系统实验发现，随着 majority voting 采样次数增加，简单的 CoT 始终超越复杂 prompting 策略；并从概率论角度给出理论证明，提出 $O(1)$ 复杂度的 scaling 性能预测方法和两种改进策略。

## 研究背景与动机

1. **领域现状**：Test-time scaling（推理时扩展计算）是近期 LLM 推理的热门方向，通过增加推理阶段的计算量（如多次采样 + majority voting）来提升推理能力。同时，研究者设计了大量 prompting 策略（CoT、ToT、Self-Refine、Step-Back 等）来增强推理。

2. **现有痛点**：复杂 prompting 策略在 pass@1 上通常优于简单 CoT，但**当 test-time scaling 时（大量采样 + majority voting），不同策略的相对表现如何变化**，尚未被系统研究。人们默认"pass@1 更好 = scaling 后也更好"，但这一假设未被验证。

3. **核心矛盾**：复杂策略（ToT、MAD 等）虽然单次推理更准，但每次推理成本也更高。在相同计算预算下，它们能比简单 CoT 多用几倍的 token？

4. **本文要解决什么？**
   - 哪种 prompting 策略在 test-time scaling 下表现最好？
   - 为什么会出现这种现象？有没有理论解释？
   - 能否高效预测不同策略在不同采样次数下的性能？

5. **切入角度**：从概率论视角出发，将 majority voting 建模为多项分布的 mode 选择问题，分析"简单题"和"难题"的比例如何决定不同策略的 scaling 曲线。

6. **核心 idea 一句话**：简单 CoT 在 test-time scaling 中持续胜出，因为它有更多"简单题"（正确答案概率最高）且错误答案的分布更分散。

## 方法详解

### 整体框架

输入：LLM $\mathcal{M}$、prompting 策略 $\mathbf{P}_i$、采样次数 $N$、数据集 $\mathfrak{D}$  
处理：对每个问题采样 $N$ 次，用 majority voting 选出最终答案  
评估：在固定采样次数 $N$ 或固定推理成本 $O$ 下比较不同策略的准确率

### 关键设计

1. **问题难度定义（Definition 1）**:
   - 做什么：将问题按对某个 prompting 策略的难度分为三类
   - 核心思路：设答案空间 $\mathcal{A} = \{a_1, ..., a_m\}$，$p_{i,j}$ 是策略 $\mathbf{P}_i$ 给出答案 $a_j$ 的概率。若正确答案 $a_1$ 的概率唯一最大，则为**简单题**；若概率并列最大，则为**中等题**；若某个错误答案概率最大，则为**难题**
   - 设计动机：直觉上，简单题 scaling 后趋向 100%，难题趋向 0%

2. **Scaling 收敛定理（Theorem 1-3）**:
   - 做什么：证明三种难度问题在 $N \to \infty$ 时的行为
   - 核心思路：**简单题**极限为1；**中等题**极限为 $1/|\mathcal{S}|$；**难题**极限为0
   - 设计动机：CoT 有更多简单题、更少难题（Table 1 验证）

3. **策略翻转定理（Theorem 4）**:
   - 做什么：证明在什么条件下一个初始更差的策略会在大 $N$ 时反超
   - 核心思路：如果策略的正确答案概率与最大错误答案概率的差距更大，则在大 $N$ 后反超
   - 设计动机：CoT 的错误分布更分散，不会集中在某个错误答案上

4. **$O(1)$ Scaling 性能预测方法**:
   - 做什么：从少量采样估计概率分布，预测任意 $N$ 下的性能
   - 设计动机：避免大 $N$ 下的高成本实际推理

### 改进策略

- **自适应 scaling**：简单题少采样、难题多采样
- **动态策略选择**：对每个问题选最优 prompting 策略

## 实验关键数据

### 主实验（CoT vs 其他策略，Qwen2.5-7B 为例）

| 策略 | 简单题比例 | 难题比例 | 极限准确率 |
|------|-----------|---------|-----------|
| CoT | **88.1%** | **11.6%** | **88.2%** |
| L2M | 87.4% | 12.3% | 87.6% |
| SBP | 87.1% | 12.8% | 87.2% |
| DiP | 86.3% | 13.4% | 86.4% |

### 消融实验（改进策略效果）

| 配置 | GSM8K Maj@10 | MATH-500 Maj@10 |
|------|-------------|-----------------|
| 纯 CoT | 86.0% | 15.2% |
| + 两种改进结合 | **97.4%** | **61.0%** |

### 关键发现
- 在所有 6 个 LLM 上，CoT 在 $N$ 足够大时都是最优策略，约 80% 的组合符合趋势
- Self-Refine 在 scaling 下表现最差，甚至不如 Direct Prompting
- 在强模型上，Direct Prompting 也能在大 $N$ 时表现最佳

## 亮点与洞察

- **反直觉发现**：pass@1 更好的策略在 test-time scaling 下不一定更好
- **概率论分析优雅而实用**：将 majority voting 建模为多项分布的 mode 问题
- **$O(1)$ 预测方法**：可在实际部署中快速选择最优策略和采样次数

## 局限性 / 可改进方向

- **仅考虑 majority voting**：未分析 Best-of-N、Process Reward Model 等方法
- **答案提取依赖正则**：对开放式生成任务难以直接应用
- **理论假设单题独立**：实际中不同问题的分布可能有相关性

## 相关工作与启发

- **vs Self-Consistency (Wang et al., 2023)**: 本文在 Self-Consistency 框架下系统比较了多种 prompting
- **vs OpenAI o1/o3**: o1 的 test-time scaling 通过 trained reasoning，本文聚焦 prompting 层面
- **vs Scaling Laws**: 本文的预测方法可看作 test-time majority voting 的 scaling law

## 评分
- 新颖性: ⭐⭐⭐⭐ 反直觉发现+理论证明，实验规模大而全面
- 实验充分度: ⭐⭐⭐⭐⭐ 6 LLM × 8 策略 × 6 benchmark
- 写作质量: ⭐⭐⭐⭐ 理论部分清晰，但符号较多
- 价值: ⭐⭐⭐⭐ 对 test-time scaling 实践有直接指导意义
