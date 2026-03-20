# Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning

## 基本信息
- **arXiv**: 2506.01939
- **会议**: NeurIPS 2025
- **作者**: Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, et al.
- **机构**: Tsinghua University, Alibaba
- **代码**: 未开源

## 一句话总结
从 token 熵模式的全新视角分析 RLVR，发现 CoT 推理中仅约 20% 的高熵"分叉 token"决定推理方向，仅在这些 token 上做梯度更新即可匹配甚至大幅超越全量更新（Qwen3-32B 上 AIME'25 +11.04），揭示 RLVR 本质是优化推理决策点。

## 背景与动机
RLVR (Reinforcement Learning with Verifiable Rewards, 如 DeepSeek-R1 的 GRPO) 已被证明能显著提升 LLM 推理能力，但其机制仍不清楚：
- RL 到底改变了模型的什么？
- 是所有 token 都同等重要，还是某些关键 token 起主导作用？
- 能否利用这一理解来改进 RLVR？

## 核心问题
RLVR 在 token 层面到底在优化什么？能否只优化最关键的 token 来提升效率和效果？

## 方法详解

### 1. Token 熵模式分析
对 CoT 推理过程中每个 token 位置计算策略熵 $H(p_\theta(\cdot | x, y_{<t}))$：
- **关键发现 1**：绝大多数 token（~80%）的熵极低（模型很自信怎么继续）
- **关键发现 2**：仅少数 token（~20%）有高熵——这些是**"分叉 token"(forking tokens)**，模型在这些位置面临推理方向的选择
- **关键发现 3**：高熵 token 对应推理链中的决策点（如选择解题策略、决定论证方向）

### 2. RLVR 训练动态分析
观察 RLVR 训练过程中 token 熵的变化：
- RLVR **基本保持**基础模型的熵分布模式
- 主要调整的是**高熵 token 的熵值**（向正确推理方向降低熵）
- 低熵 token 几乎不受影响
- 这说明 RLVR 的本质作用是在推理"分叉口"引导模型做出更好的选择

### 3. Forking Token Gradient（核心方法）
基于上述洞察，提出仅在高熵 forking token 上计算梯度更新：
- 对每个生成 token，计算策略熵
- 选择熵最高的 top-20% token
- **仅在这些 token 上应用 policy gradient**
- 其余 80% token 不参与梯度计算

### 4. 实验结果惊人
- **20% token 更新 ≈ 100% token 更新**（Qwen3-8B 上性能相当）
- **20% token 更新 >> 100% token 更新**：
  - Qwen3-32B: AIME'25 +11.04, AIME'24 +7.71
  - Qwen3-14B: AIME'25 +4.79, AIME'24 +5.21
- **反证**：仅在 80% 低熵 token 上训练 → 性能显著下降
- 展示了强劲的 **scaling trend**——模型越大，forking token 优化的优势越明显

## 实验关键数据

### AIME 基准
| 模型 | 方法 | AIME'25 | AIME'24 |
|---|---|---|---|
| Qwen3-32B | 全量梯度 RLVR | baseline | baseline |
| Qwen3-32B | **Forking token (20%)** | **+11.04** | **+7.71** |
| Qwen3-14B | 全量梯度 RLVR | baseline | baseline |
| Qwen3-14B | **Forking token (20%)** | **+4.79** | **+5.21** |

### Token 类型消融
| 训练 token | 比例 | 效果 |
|---|---|---|
| 全量 (100%) | 100% | baseline |
| 高熵 top-20% | 20% | ≥ baseline（大模型上大幅超越）|
| 低熵 bottom-80% | 80% | **显著下降** |

## 亮点
1. **全新视角**：从 token 熵模式理解 RLVR，简洁而深刻
2. **违反直觉的 80/20 规律**：20% token 足以（甚至更好地）驱动 RL 训练
3. **强 scaling trend**：模型越大优势越大，暗示 forking token 优化是通往 AGI 推理的正确方向
4. **实践价值**：减少 80% 梯度计算，显著降低 RLVR 训练成本
5. **理论洞察**：RLVR 本质是优化推理决策点，而非全面改写生成模式

## 局限性
1. 高熵 token 的阈值选择（top-20%）是经验性的
2. 分析主要在数学推理任务，对代码/自然语言推理的泛化性待验证
3. 熵计算需要额外前向传播，引入一定开销
4. 对基础模型的 token 熵分布质量有依赖

## 与相关工作的对比
- **vs. Does Thinking More Help? (本批之前写的)**：后者发现 overthinking 增加方差，本文发现 RLVR 的关键是优化高熵（高方差）token——两者互补，共同说明方差/熵是推理的核心信号
- **vs. DAPO/GRPO**：标准 RLVR 对所有 token 均匀更新，forking token 方法更高效精准
- **vs. Token-level reward 方法**：过程奖励模型 (PRM) 希望在 step 级给反馈，本文直接在 token 级识别关键位置
- **vs. ThinkPrune/Thinkless**：这些方法减少无用 thinking token，本文从 RL 训练端识别关键 token

## 启发与关联
- **与 Overthinking 研究的联系**：高熵 token 是推理"十字路口"，overthinking 可能在这些位置引入过多分叉 → 组合使用可能更有效
- **稀疏 RL 更新的前景**：如果只需 20% token 的梯度，RLVR 的计算和内存成本都可大幅降低
- **对 RL scaling 的启示**：scaling up RL 时，应该关注如何在更大模型上利用 forking token 信号，而非简单增加 sample 数

## 评分
- 新颖性：★★★★★ — token 熵视角理解 RLVR 是开创性洞察
- 技术深度：★★★★☆ — 分析深入，方法简洁但有效
- 实验完整度：★★★★★ — 3 个模型规模 × 多 benchmark × 消融 × scaling analysis
- 写作质量：★★★★★ — 叙事层层递进，从观察到解释到方法环环相扣
