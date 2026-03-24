# AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search

**会议**: AAAI 2026  
**arXiv**: [2506.06017](https://arxiv.org/abs/2506.06017)  
**代码**: https://github.com/Ericccc02/AgentSwift  
**领域**: Agent  
**关键词**: 自动化Agent设计, MCTS, Value Model, 层次搜索空间, 功能组件

## 一句话总结
提出AgentSwift框架，通过层次化搜索空间（同时优化agentic workflow和功能组件）、轻量级value model预测agent性能、以及不确定性引导的MCTS搜索策略，自动发现高性能LLM agent设计，在7个基准上平均提升8.34%。

## 研究背景与动机
1. **领域现状**：LLM agent已在多种任务上展现强大能力，但agent的设计仍然高度依赖人工——从workflow组织到memory、planning、tool use等功能组件的选择与配置，都需要大量expert knowledge和反复调参。
2. **现有痛点**：已有的自动化agent设计方法存在三个主要问题：
   - **搜索空间受限**：AFlow、ADAS等方法只优化agentic workflow结构，不涉及memory/planning/tool use等功能组件，无法发现完整的agent架构
   - **评估成本过高**：每评估一个新agent都需要在benchmark上完整运行（如ALFWorld上一个CoT agent评估需~$60），大量低质量candidate浪费资源
   - **搜索效率低下**：面对巨大设计空间，现有搜索策略（如AFlow的局部优化）容易陷入局部最优
3. **核心矛盾**：设计空间的组合爆炸性（workflow × memory × tool × planning）与单次评估的高成本之间的根本矛盾，使得穷举式搜索不可行，而局部搜索又容易遗漏优质设计
4. **本文要解决什么？**
   - 如何构建一个同时包含workflow和功能组件的统一搜索空间？
   - 如何用低成本替代昂贵的真实评估？
   - 如何高效地在巨大搜索空间中导航？
5. **切入角度**：借鉴NAS（神经架构搜索）中performance predictor的思路——在NAS中，训练一个性能预测模型来替代完整训练评估已被证明非常有效，agent设计问题与NAS高度类似
6. **核心idea一句话**：将agent设计形式化为workflow+功能组件的层次搜索问题，用轻量value model做低成本评估，用不确定性引导的MCTS做高效搜索

## 方法详解

### 整体框架
AgentSwift的pipeline分三个核心模块：
- **输入**：任务描述 $d$ + 性能评估函数 $\text{Eval}_d(\cdot)$
- **搜索空间**：层次化定义 $\mathbf{A} = (\mathbf{W}, \mathbf{M}, \mathbf{T}, \mathbf{P})$，其中$\mathbf{W}$是agentic workflow、$\mathbf{M}$是memory、$\mathbf{T}$是tool use、$\mathbf{P}$是planning
- **搜索引擎**：不确定性引导的MCTS，每步通过recombination→mutation→refinement三级操作生成新candidate
- **评估器**：轻量级value model预测candidate性能得分，仅对top candidate做真实评估
- **输出**：最优agent设计 $\mathbf{A}^* = \arg\max_{\mathbf{A} \in \mathcal{S}_{\text{agent}}} \text{Eval}_d(\mathbf{A})$

### 关键设计

1. **层次化搜索空间**:
   - 做什么：统一定义workflow和功能组件的联合搜索空间
   - 核心思路：Agentic workflow $\mathbf{W}=(N,E)$ 由节点（LLM调用步骤，含model/prompt/temperature/format）和边（执行顺序）组成。在此基础上，三个功能组件以plug-and-play方式挂载——Memory $\mathbf{M}=(m,\tau,d)$ 负责检索/存储上下文，Tool Use $\mathbf{T}=(t,\tau,u)$ 连接外部API，Planning $\mathbf{P}=(p,\tau)$ 做子目标分解。完整agent定义为 $\mathbf{A}=(\mathbf{W},\mathbf{M},\mathbf{T},\mathbf{P})$
   - 设计动机：之前AFlow只搜索workflow，AgentSquare虽然引入组件但在固定workflow模板下搜索、且仍以prompt优化为主。本文的层次空间真正实现了workflow和组件的联合优化，搜索空间更具表达力

2. **Value Model（性能预测模型）**:
   - 做什么：给定candidate agent和任务描述，预测其性能得分 $\hat{v} = f_\theta(\mathbf{A}, d)$
   - 核心思路：基于7B预训练语言模型（Mistral-7B或Qwen2.5-7B）+ 轻量adapter，用MSE损失端到端微调。关键创新在数据集构建：先用 $t=2$ 的covering array确保所有成对组件交互至少出现一次（保证覆盖度），再用Balanced Bayesian Sampling同时探索高性能区域（UCB）和低性能区域（LCB），公式为 $a_{\text{UCB}}(\mathbf{A}) = \mu(\mathbf{A}) + \kappa \cdot \sigma(\mathbf{A})$ 和 $a_{\text{LCB}}(\mathbf{A}) = -\mu(\mathbf{A}) + \kappa \cdot \sigma(\mathbf{A})$，总共收集220个有标注样本
   - 设计动机：GPT-4o做in-context预测虽然可行但成本高（每次搜索都要调用大模型），且精度不够（Spearman仅0.77 vs 本方法0.90）。蒸馏到7B模型后推理快、成本低，且因为structured agent表示的组合性质，7B模型能学到好的component→performance映射

3. **不确定性引导的MCTS**:
   - 做什么：高效搜索最优agent设计
   - 核心思路：
     - **Selection**：采用soft mixed probability策略，结合实际性能 $s_i$ 和不确定性 $u_i$，公式 $P_{\text{mixed}}(i) = \lambda \cdot \frac{1}{n} + (1-\lambda) \cdot \frac{\exp(E(s_i, u_i))}{\sum_j \exp(E(s_j, u_j))}$，其中 $E(s_j, u_j) = \alpha((1-\beta) s_j + \beta u_j - s_{\max})$
     - **Expansion**：从parent agent出发，依次执行三步操作：(1) Recombination——从组件池中替换一个子系统；(2) Mutation——LLM根据任务和历史性能生成新组件实现；(3) Refinement——基于失败案例微调prompt/temperature/控制流。每步都用value model评分筛选最佳candidate
     - **Evaluation**：对最终candidate做真实评估，不确定性定义为 $u = |s_{\text{real}} - \hat{s}|$
     - **Backpropagation**：向上传播真实分数和不确定性，更新visit count
   - 设计动机：纯exploitation容易陷入局部最优，纯exploration浪费budget。不确定性机制让搜索既关注高性能区域，也关注预测不确定的区域（可能隐藏高性能candidate）

### 损失函数 / 训练策略
- Value model用MSE损失训练：$\mathcal{L} = \frac{1}{N}\sum_i (v_i - f_\theta(\mathbf{A}_i, d_i))^2$
- 数据集220样本，8:1:1划分训练/验证/测试
- 搜索budget上限60个agent（与baseline公平对比）
- 3张A100 GPU训练value model

## 实验关键数据

### 主实验
在GPT-4o-mini上的7个benchmark结果（3次独立运行取平均）：

| 方法 | ALFWorld | SciWorld | MATH | WebShop | M3Tool | Travel | PDDL | 类型 |
|------|----------|----------|------|---------|--------|--------|------|------|
| COT | 0.512 | 0.398 | 0.532 | 0.490 | 0.427 | 0.433 | 0.427 | 手工 |
| FoA | 0.587 | 0.427 | 0.556 | 0.509 | 0.488 | 0.474 | 0.472 | 手工 |
| AgentSquare | 0.701 | 0.475 | 0.556 | 0.520 | 0.561 | 0.553 | 0.577 | 搜索 |
| AFlow | 0.619 | 0.452 | 0.562 | 0.497 | 0.524 | 0.497 | 0.528 | 搜索 |
| **AgentSwift** | **0.806** | **0.509** | **0.628** | **0.562** | **0.634** | **0.573** | **0.614** | 搜索 |

AgentSwift在所有7个benchmark上全面超越手工设计和自动搜索baseline，平均提升8.34%。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| AgentSwift (full) | 0.806 (ALFWorld) | 完整模型 |
| w/o Uncertainty | 性能曲线明显变平 | 搜索倾向exploit已知区域 |
| w/o MCTS | 性能曲线最平 | 缺乏层次化探索，变成局部搜索 |
| Full evaluation (替代value model) | 收敛最慢 | 预算大量浪费在低质量candidate上 |
| GPT-4o few-shot预测 | 中间 | 比value model精度低 |

Value model预测质量对比（Spearman相关系数）：

| 方法 | MSE | MAE | R² | Spearman |
|------|-----|-----|----|----|
| AgentSwift (Mistral) | 0.006 | 0.053 | 0.807 | **0.903** |
| AgentSwift (Qwen) | 0.005 | 0.055 | 0.828 | 0.899 |
| GPT-4o few-shot | 0.016 | 0.089 | 0.479 | 0.765 |
| GPT-4o zero-shot | 0.068 | 0.207 | -1.17 | 0.056 |

### 关键发现
- **MCTS + 不确定性引导**是搜索效率的核心——去掉任一都导致搜索曲线显著变平
- Value model用30个标注样本few-shot adaptation到新任务，MSE就接近oracle（全量训练）性能，说明agent设计的结构化表示具有很强的迁移性
- 发现的agent架构具有跨模型泛化性——用GPT-4o-mini搜到的agent直接在其他LLM上也能表现良好
- 超参数 $\alpha$, $\lambda$, $\beta$ 的敏感性分析显示方法很鲁棒（ALFWorld上变化范围0.768-0.813）

## 亮点与洞察
- **NAS→Agent Search的范式迁移**非常巧妙：将NAS中的performance predictor思想迁移到agent设计搜索中，7B模型做surrogate evaluation比直接调GPT-4o预测更准且更便宜。这个思路可以推广到任何涉及expensive evaluation的搜索问题
- **Balanced Bayesian Sampling数据集构建**：同时采样高性能和低性能区域的策略很聪明，保证value model既能识别好agent也能区分坏agent，比只采样高性能区域的数据集更有判别力
- **三级expansion操作（recombination→mutation→refinement）**形成了从粗到细的搜索策略，recombination做大步跳跃、mutation做中等创新、refinement做细粒度调优，层次分明

## 局限性 / 可改进方向
- **搜索空间仍然有限**：memory/tool/planning三个功能组件的定义比较固定，没有涵盖RAG、multi-agent协作、self-play等更复杂的agent capability
- **评估仅用GPT-4o-mini**：主实验基于一个模型，虽然做了跨模型迁移实验，但搜索过程本身是否在不同backbone上都有效未充分验证
- **value model的220样本训练数据**：对新任务需要few-shot adaptation，如果目标任务与训练任务差异很大（如跨模态），迁移效果可能受限
- **只考虑单agent设计**：未涉及multi-agent系统的自动化设计，这是一个重要的扩展方向

## 相关工作与启发
- **vs AFlow**: AFlow只搜索workflow结构，不涉及功能组件。AgentSwift在AFlow基础上扩展了搜索空间，且引入value model降低评估成本
- **vs AgentSquare**: AgentSquare虽引入memory/tool/planning，但在固定workflow模板下搜索，且用GPT-4o做in-context预测（精度低、成本高）。AgentSwift用层次搜索空间+专用value model，两方面都更优
- **vs ADAS**: ADAS做端到端workflow搜索但没有value model指导，搜索效率低。AgentSwift的MCTS+uncertainty策略明显更高效
- **NAS启发**：这篇论文建立了NAS→Agent Design的清晰类比，为后续agent自动化设计研究提供了一个很好的框架参考

## 评分
- 新颖性: ⭐⭐⭐⭐ 层次搜索空间+value model+MCTS的组合很完整，但各个组件都有先例（NAS predictor、MCTS in LLM reasoning等）
- 实验充分度: ⭐⭐⭐⭐⭐ 7个benchmark、多种baseline对比、详细消融、泛化分析、超参敏感性，非常完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，形式化定义完备，但部分细节（如组件池的具体实现）在附录中
- 价值: ⭐⭐⭐⭐ 为自动化agent设计提供了一个系统性框架，实际应用中能帮助快速发现好的agent architecture
