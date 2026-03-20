# ASIDE: Architectural Separation of Instructions and Data in Language Models

**会议**: ICLR 2026  
**arXiv**: [2503.10566](https://arxiv.org/abs/2503.10566)  
**代码**: 无  
**领域**: AI安全 / Prompt Injection 防御  
**关键词**: instruction-data separation, prompt injection, orthogonal rotation, token embedding, architectural safety  

## 一句话总结
提出 ASIDE，一种在 token embedding 层面通过正交旋转区分指令和数据的架构级改造，仅需修改前向传播并在标准指令微调数据上训练，即可显著提升指令-数据分离度和 prompt injection 鲁棒性，无需任何安全专项训练。

## 研究背景与动机
1. **领域现状**：LLM 广泛集成在邮件客户端、Agent 流水线等软件系统中，这些场景天然存在指令（应执行）和数据（应处理、不执行）两类输入，但当前 LLM 架构对两者使用完全相同的 embedding，无法在模型内部区分。
2. **现有痛点**：缺少指令-数据分离是 prompt injection（间接注入、直接注入）攻击成功的根本原因。现有防御要么依赖 prompt engineering/特殊分隔符（易被绕过），要么依赖对抗训练（仅针对特定攻击模式），都不能从根本上解决问题。
3. **核心矛盾**：在传统 LLM 中，同一个 token 无论出现在指令还是数据中，其 embedding 完全相同——模型必须从上下文推断 token 的功能角色，这在深层网络中极难可靠实现。
4. **本文要解决什么？** 如何在不增加参数、不重新预训练的前提下，让模型从第一层就能区分指令和数据 token？
5. **切入角度**：token embedding 通常具有低秩结构，指令和数据可以共享同一个高维空间但驻留在不同的线性子空间中。正交旋转可以在不改变 embedding 范数和内积结构的情况下创建这种分离。
6. **核心idea一句话**：对数据 token 的 embedding 施加固定的 $\frac{\pi}{2}$ 正交旋转，让模型从第一层起就能通过 embedding 区分指令和数据。

## 方法详解

### 整体框架
ASIDE 仅修改 LLM 的 embedding 层前向传播：输入 token $x$ 如果是指令，embedding 为 $E[I_x, \cdot]$（原始 embedding）；如果是数据，embedding 为 $R(E[I_x, \cdot])$，其中 $R \in \mathbb{R}^{d \times d}$ 是一个固定的正交旋转矩阵。之后用标准 SFT 在 Alpaca-clean 数据集上微调即可。

### 关键设计

1. **Isoclinic 正交旋转**：
   - 做什么：将数据 token 的 embedding 旋转到正交子空间中。
   - 核心思路：将 embedding 维度按二维分组，每组施加 $\frac{\pi}{2}$ 旋转矩阵 $\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$。这是一个不可学习的固定变换。
   - 设计动机：正交旋转保持向量的范数和相对角度不变（不引入信息丢失），同时创造了两个完全正交的子空间。这比 ISE（Wu et al., 2024）的可学习偏移向量更有效，因为偏移在深层逐渐被模型消化（失去区分度），而旋转在几何上保持了永久的可分性。
   - 零额外参数：旋转矩阵是固定的，不增加任何可训练参数。

2. **功能角色注解**：
   - 做什么：在部署时标注每个 token 是指令还是数据。
   - 核心思路：利用系统设计中已有的角色信息（如邮件内容始终是数据，系统提示始终是指令），这不需要模型推断。
   - 限制：需要应用场景能提供 token 级角色标注，不适用于无法区分角色的通用聊天场景。

3. **后向兼容的集成流程**：
   - 做什么：将 ASIDE 集成到已预训练的模型中。
   - 步骤：(1) 修改前向传播加入旋转逻辑 (2) 在标准 SFT 数据（无安全数据）上微调 3 个 epoch。

### 训练策略
标准 SFT，无对抗训练、无安全目标函数。在 Alpaca-clean-gpt4-turbo 数据集（51.8k 样本）上训练，学习率 $[1 \times 10^{-6}, 2 \times 10^{-5}]$，batch size 64-256，warm-up ratio [0, 0.1]。

## 实验关键数据

### 主实验：指令-数据分离度 (SEP Score)
| 模型 | Vanilla | ISE | ASIDE | 提升 (vs Vanilla) |
|------|---------|-----|-------|-------------------|
| Llama 2 7B | ~55% | ~52% | ~67% | +12.3 pp |
| Llama 3.1 8B | ~50% | ~53% | ~70% | +20 pp |
| Qwen 2.5 7B | ~57% | ~57% | ~75% | +18 pp |
| Qwen 3 8B | ~31% | ~20% | ~65% | +34 pp |
| Mistral 7B | ~28% | ~50% | ~72% | +44.1 pp |

ASIDE 的 utility（AlpacaEval、SEP Utility）与 Vanilla 基本持平。

### Prompt Injection 鲁棒性 (ASR↓)
| 模型 | 攻击类型 | Vanilla | ASIDE | 降低 |
|------|---------|---------|-------|------|
| Llama 3.1 8B | BIPIA-text | 13.6% | 4.1% | -9.5 pp |
| Llama 3.1 8B | BIPIA-code | 22.8% | 9.2% | -13.6 pp |
| Llama 3.1 8B | StruQ-ID | 43.3% | 41.3% | -2.0 pp |
| Qwen 2.5 7B | BIPIA-text | 18.3% | 14.5% | -3.8 pp |
| Qwen 3 8B | BIPIA-text | 10.2% | 2.8% | -7.4 pp |
| Qwen 3 8B | StruQ-ID | 47.0% | 8.1% | -38.9 pp |
| Mistral 7B | BIPIA-text | 11.1% | 0.5% | -10.6 pp |
| Mistral 7B | StruQ-ID | 33.4% | 9.6% | -23.8 pp |

### 关键发现
- ASIDE 在所有模型上一致提升 SEP score（12-44 pp），同时保持 utility 基本不变。
- 间接 prompt injection 的 ASR 平均降低约 10-40 pp，效果在 Mistral 和 Qwen3 上尤为显著。
- ISE 方法在多数模型上与 Vanilla 无统计显著差异甚至更差，说明可学习偏移向量不足以维持深层的分离。
- 线性探测分析显示 ASIDE 从第 0 层（embedding 后）就达到 100% 线性可分性，而 Vanilla 需到 5-10 层才逐渐可分。
- 概念激活分析显示 ASIDE 有效抑制了数据区域中"指令概念"的虚假激活。

## 亮点与洞察
- **用架构解决安全问题**：类比计算机安全中数据执行保护（DEP），从架构层面区分可执行和不可执行内存。ASIDE 是首个将这一思想成功引入 LLM 的工作。
- **零成本安全收益**：不需要对抗训练、不需要安全数据集、不增加参数，仅靠一个固定旋转 + 标准 SFT 就获得显著的安全提升，这一结论非常有力。
- **旋转 vs 偏移的设计洞察**：ISE 用可学习偏移在 embedding 空间创建区分，但偏移随层加深会被模型"消化"；正交旋转在几何上更持久，因为它创造了正交子空间而非仅有偏移的同一子空间。

## 局限性 / 可改进方向
- 需要部署时知道每个 token 的功能角色（指令 vs 数据），限制了适用场景——通用聊天助手中用户输入既是"指令"也可能包含"数据"，角色边界模糊。
- 未在安全微调的 Instruct 模型上实验（有意为之，但实际部署会用 Instruct 模型），ASIDE + safety tuning 的叠加效果未知。
- StruQ-OOD 攻击上改善有限，说明 OOD 注入仍是挑战。
- 固定的 $\frac{\pi}{2}$ 旋转是否最优？论文未探索其他旋转角度或可学习旋转的效果。

## 相关工作与启发
- **vs ISE (Wu et al., 2024)**：ISE 用可学习偏移向量区分角色，但在深层失去分离效果，在安全指标上基本等同 Vanilla 甚至更差。ASIDE 用正交旋转实现更持久的分离。
- **vs Prompt Engineering**：分隔符/特殊 token 本质上是在输入层面做分离，容易被攻击者伪造；ASIDE 在 embedding 层面做分离，攻击者无法通过文本操控旋转。
- **vs 对抗训练**：对抗训练是"看过什么攻击就防什么"，ASIDE 提供一种攻击无关的结构性防御。
- 该方法可以直接迁移到多层级指令体系（如 system > user > tool），只需定义更多正交变换。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次从架构层面实现指令-数据分离，理念清晰优雅
- 实验充分度: ⭐⭐⭐⭐ 6 个模型 × 8 个安全基准 + 分离度评估 + 可解释性分析，但未测试 Instruct 模型
- 写作质量: ⭐⭐⭐⭐⭐ 动机来自计算机安全的经典原则，行文流畅，逻辑严密
- 价值: ⭐⭐⭐⭐⭐ 提供了一条全新的安全增强路径，实用且理论基础扎实
