# DeAL: Decoding-time Alignment for Large Language Models

**会议**: ACL 2025  
**arXiv**: [2402.06147](https://arxiv.org/abs/2402.06147)  
**代码**: 无  
**领域**: 对齐RLHF  
**关键词**: decoding-time alignment, reward model, A* search, harmlessness-helpfulness, jailbreak defense  

## 一句话总结
DeAL 将 LLM 对齐问题重新形式化为解码时的启发式搜索问题，在推理阶段利用可定制的奖励函数（包括程序化约束和参数化 reward model）引导 token 选择，实现了灵活的多目标对齐且可与 RLHF 互补叠加。

## 研究背景与动机
1. **领域现状**：当前 LLM 对齐主要依赖训练时方法（RLHF、DPO 等），通过人类偏好数据微调模型参数来嵌入对齐目标
2. **现有痛点**：
   - 对齐目标既不静态也不通用——不同用户、场景需要不同的对齐标准，但 RLHF 将单一对齐观"烧"进模型
   - 定制化对齐需要重新微调和维护多个模型，成本高昂
   - 训练时对齐的可靠性存疑——即使经过安全训练的模型仍可被 jailbreak 轻松绕过
3. **核心矛盾**：训练时对齐是一次性、静态的嵌入过程，无法适应动态变化的对齐需求，也无法在推理时提供强制性保证
4. **本文要解决什么？** 如何在解码阶段灵活、可靠地施加可定制的对齐约束
5. **切入角度**：将文本生成视为启发式搜索问题，alignment objectives 作为搜索启发式函数
6. **核心idea一句话**：把对齐从训练时移到解码时，通过 A* 搜索 + reward model 启发式引导每步 token 选择

## 方法详解

### 整体框架
DeAL 将对齐定义为搜索问题 $\langle S, V, T, R_a \rangle$：状态空间 $S$ 是 token 序列，动作集 $V$ 是词表，转移函数 $T$ 是自回归追加 token，$R_a$ 是对齐奖励函数。搜索代理基于 LLM 的 top-k 候选 token + lookahead 机制 + 启发式评分来选择每步最优 token，直至生成 EOS。

输入 prompt 分为三部分：任务指令 $p_t$、对齐指令 $p_a$（可空）、任务输入 $p_i$。对齐指令可以用自然语言表达可公开的对齐目标，作为搜索的"起始状态适应"。

### 关键设计

1. **起始状态适应（Start-state Adaptation）**:
   - 做什么：通过 alignment prompt $p_a$ 修改输入，改善生成的初始搜索方向
   - 核心思路：好的 $p_a$ 等效于好的搜索起点，减少找到满足对齐目标的终态的难度；实验中作为超参数手动设计
   - 设计动机：利用指令微调模型的指令跟随能力，通过 prompt 将对齐要求"软嵌入"搜索空间

2. **动作选择与 Lookahead（Action Selection）**:
   - 做什么：在每步解码时从 top-k 候选 token 中选最优
   - 核心思路：对每个候选 token，向前展望 $l$ 步（greedy lookahead）得到更完整的序列，然后用启发式函数 $h(\cdot)$ 评分。最终选择标准：$c(y_t) = \log P(y_{1:t}|p) + \lambda \cdot h(y_{1:t+l}, p)$，其中 $\lambda$ 控制对齐目标的权重
   - 设计动机：许多对齐指标（如"回复是否有害"）无法对部分生成的序列有效评分，lookahead 提供足够上下文使 $h(\cdot)$ 的评估更可靠

3. **模块化奖励组合（Modular Reward Ensembling）**:
   - 做什么：支持多个对齐目标的灵活组合
   - 核心思路：通过线性加权 $h = w_1 R_1 + w_2 R_2$ 组合不同 reward model（如 harmless + helpful），用户可调整权重实现细粒度校准
   - 设计动机：不同场景对 harmlessness 和 helpfulness 的权衡不同（如安全场景偏重 harmless），模块化设计避免了为每种组合训练专门模型
   - 与 RLHF 的区别：RLHF 在训练时用 reward model 优化策略，权衡是固定的；DeAL 在解码时用 reward model 引导搜索，权衡可运行时动态调整

4. **支持的启发式类型**:
   - 程序化约束：关键词覆盖、长度限制等可程序化验证的约束，$h(\cdot)$ 直接检查约束是否满足
   - 参数化奖励：使用训练好的 reward model（如 OPT-125M 在 HH-RLHF 上微调）作为 $h(\cdot)$，评估抽象对齐目标

### 训练策略
- DeAL 本身不需要训练——它是一个纯推理时框架
- 需要的 reward model 可以独立训练：本文在 HH-RLHF 数据上微调 OPT-125M 得到 $R_{harmless}$、$R_{helpful}$、$R_{hh}$ 三个 reward model
- 可与 RLHF 叠加使用：先 RLHF 微调模型，再在解码时施加 DeAL

## 实验关键数据

### 主实验

**关键词约束生成（CommonGen）**:

| 模型 | 方法 | Soft Coverage | Hard Coverage |
|------|------|:------:|:------:|
| Falcon-7B-Instruct | $p_a$ only | 0.88 | 0.62 |
| Falcon-7B-Instruct | $p_a$ + DeAL | **0.94** | **0.80** (+18%) |
| MPT-7B-Instruct | $p_a$ only | 0.91 | 0.71 |
| MPT-7B-Instruct | $p_a$ + DeAL | **0.96** | **0.85** (+14%) |
| Dolly-v2-3B | $p_a$ only | 0.65 | 0.30 |
| Dolly-v2-3B | $p_a$ + DeAL | **0.79** | **0.51** (+21%) |

**对齐目标（Harmlessness + Helpfulness）**:

| 方法 | HarmfulQ Harmless | HH-RLHF Harmless | HH-RLHF Helpful |
|------|:------:|:------:|:------:|
| Base (无对齐) | 0.43 | 0.40 | 0.33 |
| Safety prompt | 0.63 | 0.43 | 0.60 |
| Harmless rerank | 0.40 | 0.47 | 0.53 |
| DeAL w/ $R_{harmless}$ | **1.00** | 0.57 | 0.23 |
| DeAL w/ $R_{hh}$ | **1.00** | **0.67** | **0.67** |

### 消融实验

**多目标权重校准（$(w_{harmless}, w_{helpful})$）**:

| 权重配置 | HarmfulQ Harmless | HH-RLHF Helpful |
|---------|:------:|:------:|
| (1.0, 0) | 1.00 | 0.23 |
| (0.75, 0.25) | 1.00 | 0.34 |
| (0.50, 0.50) | 0.77 | 0.48 |
| (0.25, 0.50) | 0.43 | 0.67 |
| (0, 1.0) | 0.20 | 0.77 |

**与 RLHF 组合**:

| 方法 | HarmfulQ Harmless | HH-RLHF Helpful |
|------|:------:|:------:|
| No RLHF, No DeAL | 0.33 | 0.43 |
| RLHF w/ $R_{hh}$ | 0.80 | 0.70 |
| DeAL w/ $R_{hh}$ | 0.83 | 0.53 |
| RLHF + DeAL | **0.93** | **0.70** |

### 关键发现
- DeAL 对弱指令跟随模型增益更大（Dolly-v2-3B hard coverage +21% vs MPT +14%），说明解码时对齐对能力较弱的模型更有价值
- $R_{hh}$（联合 reward）比单独 $R_{harmless}$ 或 $R_{helpful}$ 表现更好，在 harmlessness 和 helpfulness 间实现较好平衡
- RLHF + DeAL 组合效果最佳——训练时和解码时对齐是互补的
- 面对 continuation attack（jailbreak），safety prompt 的 harmless 率仅 20%，而 DeAL 达到 73%

## 亮点与洞察
- **将对齐重构为搜索问题**，统一了 safety prompt、reranking、constrained decoding 等技术在同一框架下，展示了它们本质上都是搜索策略的不同特例。这个视角转换非常优雅
- **模块化 reward 组合**允许运行时调整对齐偏好——这对需要个性化对齐的场景（不同文化、企业政策）非常实用，避免了为每种对齐配置训练单独模型
- **anti-jailbreak 能力**：由于对齐检查在每步 token 生成时进行，比 prompt-based defense 更难绕过——对安全关键应用场景很有价值

## 局限性 / 可改进方向
- **推理延迟严重**：22-55x 减速（top-k lookahead + 参数化 reward model），限制了实际部署。可考虑 reward model 蒸馏、speculative decoding 结合、或预编译 grammar 加速
- **需要 logit 访问**：无法用于黑盒 API（如 GPT-4），限制了适用范围
- **reward model 质量上限**：DeAL 的对齐效果受限于 $h(\cdot)$ 的质量，OPT-125M 级别的 reward model 可能无法精确捕捉复杂对齐目标
- **实验规模偏小**：主要在 3B-7B 模型上验证，未在更大模型（70B+）上测试效果是否仍显著
- 可结合 speculative decoding 降低 lookahead 成本，用小模型快速生成候选再用大模型 + reward model 验证

## 相关工作与启发
- **vs RLHF/DPO**: RLHF 在训练时嵌入对齐，静态不可调；DeAL 在解码时施加，动态可定制，且两者可叠加
- **vs Reward-Augmented Decoding (RAD)**: RAD 只考虑单一参数化 reward，DeAL 支持多种类型（程序化+参数化）的模块化组合
- **vs Constrained Decoding (NeuroLogic, FUDGE)**: 这些工作是 DeAL 的特例——只考虑特定类型约束，未统一框架化
- 这篇论文对 alignment 的解码时方案提供了系统性的思考框架，可作为研究 inference-time alignment 的 baseline

## 评分
- 新颖性: ⭐⭐⭐⭐ 框架化统一了多种已有技术，但核心搜索思路并非全新
- 实验充分度: ⭐⭐⭐⭐ 覆盖了程序化约束和抽象对齐，包含 jailbreak 防御，但模型规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，框架图直观，实验设计系统化
- 价值: ⭐⭐⭐⭐ 解码时对齐是重要方向，框架有实际部署价值，但延迟问题限制了落地
