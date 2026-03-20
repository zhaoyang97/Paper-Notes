# State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models

**会议**: ACL 2025  
**arXiv**: [2503.03499](https://arxiv.org/abs/2503.03499)  
**代码**: [https://github.com/furiosa-ai/ssm-state-tuning](https://github.com/furiosa-ai/ssm-state-tuning)  
**领域**: LLM效率  
**关键词**: state space model, Mamba, PEFT, state-based tuning, parameter-efficient fine-tuning  

## 一句话总结
针对 SSM（如 Mamba）提出 State-offset Tuning，一种新的"状态基"PEFT 方法家族，通过在每个时间步直接注入可训练的状态偏移量 $h'$ 替代 Prefix-Tuning 的虚拟 token，解决了 prompt-based 方法在 SSM 上表达能力受限的问题，在更少参数量下持续优于 LoRA 和 Prefix-Tuning。

## 研究背景与动机
1. **领域现状**：SSM（Mamba）作为 Transformer 的亚二次替代正在崛起，但 PEFT 方法在 SSM 上的研究很少
2. **现有痛点**：
   - Prompt Tuning 和 Prefix-Tuning 在 Transformer 上有效，但在 SSM 上表现差——因为虚拟 token 只能影响初始状态 $h_0$，其效果随时间步 $t$ 的增大而指数衰减（$\bar{A}^t h_0$）
   - LoRA 虽然有效但不利用 SSM 的独特结构特性
3. **核心矛盾**：SSM 的循环结构使得 prompt-based 方法的影响随时间衰减，而 Transformer 的 Prefix-Tuning 在每层每步都有直接影响
4. **本文要解决什么？** 为 SSM 设计利用其架构特性的 PEFT 方法
5. **切入角度**：直接修改 SSM 的隐藏状态（state），而非通过外部 prompt 间接影响
6. **核心idea一句话**：在每个时间步直接给隐状态加一个可训练偏移 $h'$，消除了 Prefix-Tuning 的时间衰减问题

## 方法详解

### 整体框架
定义"状态基方法"（state-based methods）为一类直接修改 SSM 内部状态特征的 PEFT 方法。提出两种 State-offset Tuning 变体：(1) $h$-offset：加在隐状态上 $\hat{y}_t = y_t + C_t h'$；(2) $y$-offset：加在输出上 $\hat{y}_t = y_t + y'$。

### 关键设计

1. **Prefix-Tuning 在 SSM 上的局限性分析**:
   - 做什么：数学证明 Prefix-Tuning 等价于 Initial State Tuning
   - 核心推导：虚拟 token 的效果 = $\bar{A}^t h_{\text{prefix}}$，随 $t$ 指数衰减，只能影响初始状态
   - 设计动机：解释了为什么 prompt-based PEFT 在 SSM 上效果差——影响力衰减太快

2. **State-offset Tuning (h-offset)**:
   - 做什么：在每个时间步的 SSM 输出上加一个与位置无关的可训练偏移
   - 核心思路：$\hat{y}_t = y_t + C_t h'$，其中 $h' \in \mathbb{R}^H$ 是可训练参数（每通道共享）
   - 设计动机：消除了 $\bar{A}^t$ 的时间衰减——偏移量在每个时间步都有均匀影响。参数量仅 $D \cdot H$
   - 与 Initial State Tuning 的区别：Initial State 的效果 $= C_t (\prod \bar{A}_i) h'$，有衰减；State-offset 的效果 $= C_t h'$，无衰减

3. **State-offset Tuning (y-offset)**:
   - 做什么：直接在 SSM 的输出标量上加偏移
   - 核心思路：$\hat{y}_t = y_t + y'$，参数量仅 $D$（每通道一个标量）
   - 更极端地参数高效，但表达能力稍弱

## 实验关键数据

### 主实验

**Mamba-2.8B, 多种下游任务**:

| 方法 | 参数量 | 平均性能 |
|------|:-----:|:------:|
| Full FT | 100% | 基准上限 |
| LoRA | ~0.5% | 中等 |
| Prefix-Tuning | ~0.3% | 差 |
| Initial State Tuning | ~0.01% | 中等偏下 |
| **State-offset (h)** | **~0.01%** | **优于 LoRA** |
| **State-offset (y)** | **~0.001%** | **接近 LoRA** |

State-offset 用远少于 LoRA 的参数达到了更好的效果。

### 关键发现
- Prompt-based 方法在 SSM 上确实不如 Transformer——验证了时间衰减的理论分析
- State-offset 的均匀影响是关键——消除衰减后性能大幅提升
- h-offset 优于 y-offset——通过 $C_t$ 矩阵保留了与输入的交互

## 亮点与洞察
- **首次为 SSM 定义"状态基 PEFT"方法家族**：从 SSM 架构特性出发设计 PEFT，而非简单迁移 Transformer 的方法
- **理论分析清晰**：Prefix-Tuning 等价于 Initial State Tuning 的证明简洁有力，直接解释了为什么它在 SSM 上不行
- **极简设计**：State-offset 每层只增加 $D \times H$ 个参数（$h'$），可能是 SSM 上最参数高效的 PEFT 方法之一

## 局限性 / 可改进方向
- **仅在 Mamba 上验证**：其他 SSM 变体（RWKV、RetNet）的适用性未测试
- **偏移量与位置无关**：在所有时间步用相同 $h'$，可能限制了对位置敏感任务的适应
- **可与 LoRA 组合**：State-offset 用于 SSM 模块 + LoRA 用于线性层可能效果更好

## 相关工作与启发
- **vs LoRA on SSM**: LoRA 不利用 SSM 结构；State-offset 直接从 SSM 的状态方程出发
- **vs Prefix-Tuning**: Prefix-Tuning 在 SSM 上等价于 Initial State Tuning 且有衰减；State-offset 无衰减
- 随着 SSM 模型（Mamba-2、Jamba）的推广，State-based PEFT 将越来越重要

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次从 SSM 架构特性出发设计 PEFT，理论+方法都是原创
- 实验充分度: ⭐⭐⭐⭐ 多种 SSM 模型和下游任务验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论分析优雅，公式推导清晰
- 价值: ⭐⭐⭐⭐ SSM 的 PEFT 是新方向，State-offset 提供了简洁有效的起点
