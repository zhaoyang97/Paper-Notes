# Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test

**会议**: ICLR 2026  
**arXiv**: [2506.21551](https://arxiv.org/abs/2506.21551)  
**代码**: 无  
**领域**: 对齐RLHF  
**关键词**: grokking, memorization, generalization, MoE pathway, pretraining dynamics

## 一句话总结
首次在实际规模 LLM（7B MoE）的近单遍预训练中验证 grokking 现象——不同数据组异步记忆、延迟泛化；通过分析 MoE routing pathway 的演化（从 instance-specific 到 structured/shared），提出两个零成本指标来监控泛化进度，无需 instruction tuning 和 benchmark 评估。

## 研究背景与动机

1. **领域现状**：Grokking（延迟泛化）是训练 Transformer 时观察到的反直觉现象——训练 loss 收敛后很久，泛化性能才开始急剧提升。现有 grokking 研究限于小模型在算法数据上训练数千 epoch。

2. **现有痛点**：(a) LLM 预训练是近单遍的（~1 epoch），没有重复回放数据，loss 收敛机制与多 epoch 训练截然不同；(b) LLM 在异构跨域数据上训练，不同数据的记忆速度和泛化关系可能不同；(c) 监控 LLM 泛化性能代价极高——需要先做 instruction tuning 再跑 benchmark。

3. **核心矛盾**：预训练 loss 收敛后模型内部仍在发生什么变化？为什么 loss 不变但泛化在提升？有没有不依赖外部评估的指标来追踪泛化？

4. **本文要解决什么？** (a) 验证 grokking 是否在实际 LLM 预训练中存在；(b) 揭示记忆到泛化转变的内部机制；(c) 提供零成本泛化监控指标。

5. **切入角度**：MoE 架构天然将计算组织为 expert 选择序列（pathway），可以追踪每个样本的 pathway 如何演化——从随机/instance-specific（记忆）到结构化/跨样本共享（泛化）。

6. **核心 idea 一句话**：Grokking 在 LLM 预训练中以局部、异步的形式存在；MoE pathway 从个体特异到跨样本共享的演化是记忆到泛化转变的可观测信号。

## 方法详解

### 整体框架
基于 OLMoE-7B 的开源预训练 checkpoint 序列，跟踪训练数据的记忆时间点和下游 benchmark 的泛化时间点，验证局部 grokking。然后分析 MoE routing pathway 的动态变化，开发两个指标量化 pathway 复杂度，证明它们与泛化性能强相关。

### 关键设计

1. **局部异步 Grokking 的验证**:
   - 做什么：将训练数据按记忆时间点 $t_i^*$ 分组，将 benchmark 样本按预测变正确的时间点分组，通过 Hungarian 匹配配对
   - 核心发现：不同数据组在不同步骤被记忆，泛化通常在记忆之后以滞后方式出现。数学和代码任务需要记忆更多样本才能开始泛化，而常识 QA 泛化更快
   - 设计动机：证明 LLM 中的 grokking 不是全局同步的，而是局部的、数据异质的

2. **Pathway 编辑距离（样本间相似度）**:
   - 做什么：度量不同训练样本在 MoE 各层的 expert 选择序列的相似度
   - 核心思路：每个样本构建 pathway 字符串 $s_i = \text{concat}(e_1^{(i)}, ..., e_L^{(i)})$，计算样本对的 Levenshtein 编辑距离 $D_{path}(s_i, s_j)$
   - 关键发现：早期 pathway 几乎相同（低编辑距离）→ 记忆阶段产生分歧（高编辑距离）→ **记忆后编辑距离下降**——语义相关的样本开始收敛到相似的 pathway，标志着共享知识的出现

3. **Pathway 一致性（层间平滑度）**:
   - 做什么：度量单个样本在相邻层之间的 expert 选择一致性
   - 核心思路：计算相邻层所选 expert embedding 的加权余弦相似度
   - 关键发现：记忆后 pathway 一致性增加——expert 选择在层间变得更平滑、更结构化

4. **理论支撑**:
   - 在单层 MoE 上建立了 pathway 复杂度与泛化界之间的联系
   - 更结构化的 pathway → 更紧的泛化界

### 损失函数 / 训练策略
- 分析基于 OLMoE-7B 的 10 个等间隔预训练 checkpoint
- 泛化评估：每个 checkpoint 做 LoRA instruction tuning → 跑标准 benchmark
- 指标计算在训练数据上完成，零额外成本

## 实验关键数据

### 主实验

**Grokking 现象验证**（4 个领域 × 多个数据组）:

| 领域 | 记忆后泛化延迟 | 数据难度效应 |
|------|-------------|------------|
| 数学 | 长延迟（需记忆大量样本）| 越晚记忆，延迟越长 |
| 代码 | 长延迟 | 同上 |
| 常识 QA | 短延迟 | 相对容易泛化 |
| 领域 QA | 中等延迟 | 中等 |

### 消融实验

| 指标 | 与泛化性能相关性 | 说明 |
|------|----------------|------|
| Pathway 编辑距离 | 强负相关 | 编辑距离下降→泛化提升 |
| Pathway 一致性 | 强正相关 | 一致性增加→泛化提升 |
| 训练 loss | 无显著相关 | loss 收敛后无法预测泛化 |

### 关键发现
- **Grokking 在 LLM 预训练中确实存在**，但是局部的、异步的——不同数据组的记忆和泛化时间点不同
- **训练 loss 不能预测泛化**：loss 收敛后泛化仍在提升，且提升幅度因领域/难度而异
- **Pathway 从个体化到结构化的转变**：记忆完成后，模型继续在"更聪明地记忆"——发现跨样本可迁移的知识结构
- **深度依赖的重组**：浅层 pathway 最先共享化（普遍表示），深层保留更多灵活性（任务特化）
- **两个指标与泛化高度相关**：可作为零成本的泛化监控工具

## 亮点与洞察
- **"更聪明的记忆"**：loss 收敛不意味着学习停止——模型继续发现更高效的编码方式（shared pathways），解释了为什么持续训练能提升泛化
- **MoE 作为可解释性工具**：expert routing 的离散性天然提供了分析计算分配的窗口，这在 dense 模型中不可能做到
- **零成本泛化监控的实用价值**：对 LLM 训练者来说，不用做 instruction tuning + benchmark 就能判断何时停止预训练，极其有价值
- **局部 grokking 暗示数据课程设计**：不同数据的记忆→泛化延迟不同，暗示可以据此设计数据混合策略

## 局限性 / 可改进方向
- 仅在 OLMoE-7B 上分析，更大规模模型和 dense 架构的 grokking 行为未验证
- Pathway 指标依赖 MoE 架构，不能直接推广到 dense Transformer
- instruction tuning 的选择（LoRA vs full-finetune）可能影响泛化测量
- 因果关系未完全建立——pathway 共享化是泛化的原因还是结果？

## 相关工作与启发
- **vs Power et al. (原始 grokking)**: 小模型 + 算法数据 + 数千 epoch。本文首次在 7B MoE + 近单遍预训练中验证，发现是局部异步的
- **vs Nanda et al. (grokking 机制)**: 通过 weight 分析解释。本文通过 MoE pathway 分析，更适合大规模模型
- **vs Merrill et al. (子网络稀疏性)**: 在 ReLU 网络中关联 grokking 与稀疏性。本文的 pathway 结构化与此相呼应但在 MoE 框架下

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在实际规模 LLM 预训练中系统研究 grokking，发现局部异步模式
- 实验充分度: ⭐⭐⭐⭐ 4 域 × 多数据组 + 层级分析 + 理论支撑，但仅 1 个模型
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机推导严谨，发现的逐步揭示非常引人入胜
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 训练动态理解的根本性贡献+实用的泛化监控工具
