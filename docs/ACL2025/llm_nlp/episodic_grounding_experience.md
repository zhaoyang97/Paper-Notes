# Growing Through Experience: Scaling Episodic Grounding in Language Models

**会议**: ACL 2025  
**arXiv**: [2506.01312](https://arxiv.org/abs/2506.01312)  
**代码**: 无  
**领域**: LLM Agent  
**关键词**: Episodic Grounding, Weak-to-Strong Distillation, MCTS, Physical Planning, Preference Optimization  

## 一句话总结

提出一个 weak-to-strong episodic grounding 框架，利用 MCTS 收集结构化经验数据，通过行为比率蒸馏将小模型的 episodic grounding 能力迁移到大模型，结合 DPO 优化实现从成功和失败经验中学习，在物理规划任务上超越 GPT-4o 等 SOTA 模型 3.45%。

## 研究背景与动机

语言模型在各种生成任务中表现出色，但在物理规划任务中仍然力不从心，核心原因是缺乏 **episodic grounding**（情景记忆基础）——即从过去经验中学习并应用到新情境的能力。这一挑战与脑科学中 episodic memory 在认知中的作用密切相关。

当前方法面临一个根本性的 **scale paradox（规模悖论）**：

- **小模型（1.3B-7B）**：可以方便地在 episodic 数据上微调，但其层次化表示和长期上下文回忆能力不足，在规划任务上仅达 54.76% 准确率，远低于大模型（405B）的 74.34%。
- **大模型（70B-405B）**：拥有深层架构和丰富预训练知识，但缺乏高效的经验整合路径——参数规模使得直接微调代价极高。

这种架构不对称性意味着：最有能力利用 episodic 经验的大模型，恰恰最难被训练来整合这些经验。本文旨在打破这一悖论。

## 方法详解

### 整体框架

框架分为两个阶段：

1. **Experience Collection**：使用 MCTS 从物理模拟器（VirtualHome）中收集结构化的 episodic 经验（包括成功和失败的探索轨迹）
2. **Weak-to-Strong Distillation**：将小模型学到的 episodic 行为迁移到大模型，同时利用 DPO 从正负经验中学习偏好

### 关键设计

#### 1. MCTS 经验收集

利用蒙特卡洛树搜索从物理模拟器中收集 episodic 数据，MCTS 包含四个标准步骤：

- **Selection**：使用 UCT 公式选择有前景的节点：$UCT = Q(s,a) + C \cdot \sqrt{\frac{\log(N(s))}{N(s,a)}}$
- **Expansion**：扩展叶节点，添加未探索动作的子节点
- **Rollout**：在模拟器中执行动作序列，奖励函数为满足目标谓词 +2，无关动作 -0.1/步
- **Backpropagation**：将奖励回传更新 Q 值和访问计数

成功探索（满足所有目标谓词）标记为正样本 $y^+$，失败探索标记为负样本 $y^-$。此外还引入冗余的啰嗦计划作为额外负样本，训练模型避免生成低效的行动序列。

#### 2. 小模型训练

将物理规划任务形式化为序列预测问题，小模型（< 8B 参数）作为策略函数 $\pi$ 将输入 $\mathbf{x}$ 映射到动作序列 $\mathbf{y}$。训练目标：

$$\mathcal{L}_V = \sum_{v \in V} \alpha_v \sum_{m=1}^{M} \log \pi(y_m | \mathbf{y}_{<m}, \mathbf{x})$$

#### 3. 行为比率蒸馏（Episodic Distillation）

核心创新在于利用小模型训练前后的行为变化来指导大模型。设 $\pi^{\mathcal{E}}$ 为训练后的小模型策略，$\pi^{\mathcal{N}}$ 为原始小模型策略，其行为比率 $\frac{\pi^{\mathcal{E}}(y_m|\mathbf{y}_{<m},\mathbf{x})}{\pi^{\mathcal{N}}(y_m|\mathbf{y}_{<m},\mathbf{x})}$ 捕捉了 episodic grounding 的效果。

大模型的调整策略分布为：

$$\bar{\pi}(y_m|\mathbf{y}_{<m},\mathbf{x}) = \frac{1}{\bar{Z}} \pi^{\mathcal{L}}(y_m|\mathbf{y}_{<m},\mathbf{x}) \times \frac{\pi^{\mathcal{E}}(y_m|\mathbf{y}_{<m},\mathbf{x})}{\pi^{\mathcal{N}}(y_m|\mathbf{y}_{<m},\mathbf{x})}$$

然后通过最小化 reverse KL-divergence 进行对齐：

$$\mathcal{L}_{\text{RKL}} = \mathbb{E}_{\mathbf{x},\mathbf{y} \sim \bar{\pi}} \left[\sum_{m=1}^{M} \log \frac{\bar{\pi}(y_m|\mathbf{y}_{<m},\mathbf{x})}{\pi^{\mathcal{L}}(y_m|\mathbf{y}_{<m},\mathbf{x})}\right]$$

选择 reverse KL 而非 forward KL 的原因是 reverse KL 具有 mode-seeking 特性，确保大模型产生更精确、自信和目标对齐的动作序列。

#### 4. DPO 偏好优化

为同时从成功和失败经验中学习，引入修改版 DPO 损失：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(\mathbf{x},y^+,y^-) \sim \mathcal{D}} \left[\log \sigma\left(\beta \cdot (\log \pi(y^+|\mathbf{x}) - \log \pi(y^-|\mathbf{x}))\right)\right] + \lambda \cdot \mathbb{E}_{\mathbf{x},y \sim \pi}\left[\log \frac{\pi(y|\mathbf{x})}{\pi_0(y|\mathbf{x})}\right]$$

其中 $\beta$ 控制偏好学习的锐度，$\lambda$ 权重 reverse KL 正则项以保持与初始策略的接近度。

### 损失函数 / 训练策略

整个训练分为三阶段流水线：

1. **阶段一**：在 MCTS 收集的指令数据上训练小模型（SFT），使其学会 episodic grounding
2. **阶段二**：利用行为比率将 episodic 行为蒸馏到大模型（reverse KL 优化）
3. **阶段三**：对大模型用 MCTS 的正负样本进行 DPO 偏好优化

## 实验关键数据

### 主实验

**评估任务**：物理规划（VirtualHome）和问答（VirtualHome QA）

**Plan Generation 结果（Accuracy %）**：

| 模型 | VS | VU | CS | CU | Path | Avg. |
|------|-----|-----|-----|-----|------|------|
| GPT-4o (base) | 52.67 | 49.35 | 47.54 | 46.22 | 81.23 | 55.40 |
| GPT-Neo 1.3B-ewc | 49.70 | 49.27 | 46.88 | 42.34 | 85.91 | 54.82 |
| GPT-J 6B-ewc | 51.23 | 49.58 | 48.94 | 45.60 | 98.67 | 58.80 |

**Question Answering 结果（Accuracy %）**：

| 模型 | HW | Neg. | Recog. | Inf. | Count. | Loc. | Avg. |
|------|------|------|--------|------|--------|------|------|
| GPT-4o (base) | 85.37 | 84.31 | 95.60 | 84.85 | 78.43 | 74.21 | 83.80 |
| GPT-J 6B-ewc | 85.44 | 39.51 | 88.52 | 74.43 | 67.01 | 34.50 | 64.90 |

**总体平均**：本文方法在所有任务上平均超越 GPT-4o 3.45%。

**关键对比**：
- GPT-Neo 1.3B base → ewc：34.81% → 54.82%（+20%）
- GPT-J 6B base → ewc：45.51% → 61.29%（+15.78%）
- 小模型通过 episodic grounding 训练后显著提升

### 关键发现

1. **Weak-to-strong 有效性**：通过行为比率蒸馏，70B 和 405B 模型能在复杂长步骤规划中保持稳定准确率，而基线方法在超过 4 步后性能严重退化。

2. **Scale paradox 的解决**：小模型直接训练虽有效但天花板低，大模型通过行为比率蒸馏继承了任务特定的 grounding 能力的同时保留了通用能力。

3. **Layer-wise probing 分析**：模型的后层（deeper layers）在 episodic 推理任务上达到 90% 准确率，提供了类似人类新皮层层次化处理的经验证据。浅层编码基本感知信息，深层发展出复杂的 episodic 推理能力。

4. **失败经验的价值**：DPO 偏好优化利用失败探索作为负例，显著提升了泛化能力，避免了仅从正例学习带来的过拟合。

## 亮点与洞察

1. **Scale Paradox 的精准刻画**：清晰地诊断了 episodic grounding 中"最需要这一能力的模型最难获得它"的悖论，并提出了行为比率蒸馏这一巧妙的解决方案。
2. **认知科学的类比**：从 episodic memory 和 neocortex 层次化处理中获得灵感，layer-wise probing 分析提供了 LLM 内部表示与认知科学的有趣对应。
3. **正负经验的共同利用**：不仅学习成功轨迹，还从失败中获取有价值的信号，并引入冗余文本作为额外负例，全方位提升模型效率。
4. **不需要直接微调大模型**：通过行为比率调整推理时的分布，避免了大模型昂贵的微调成本。

## 局限性

1. **环境依赖**：经验收集依赖于 VirtualHome 模拟器，迁移到真实物理环境的效果未知。
2. **蒸馏假设**：行为比率蒸馏假设小模型和大模型的 token 空间对齐，跨模型族的迁移效果可能受限。
3. **计算开销**：MCTS 探索在模拟器中的计算成本较高，且需要为每个新环境重新收集经验。
4. **QA 任务退化**：ewc 模型在某些 QA 子任务（如 Negation、Location）上相比 base 有所退化，说明 episodic grounding 训练可能影响部分原有能力。

## 相关工作与启发

- **Embodied AI**：VirtualHome (Puig et al., 2018)、ProcTHOR (Deitke et al., 2022) 等模拟器为 episodic 数据收集提供了基础。
- **LM Grounding**：SayCan (Ichter et al., 2022)、DEPS (Wang et al., 2023) 等方法探索了 LLM 在物理环境中的 grounding，但多局限于特定任务。
- **Weak-to-strong learning**：本文的行为比率蒸馏可视为 speculative decoding 的思想延伸，用于知识迁移而非加速推理。
- **启发**：该框架的 weak-to-strong 蒸馏思路可推广到其他需要大规模微调的场景，如 RLHF 中用小模型的奖励信号引导大模型。

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4.5 |
| 技术深度 | 4.5 |
| 实验充分性 | 4 |
| 实用价值 | 3.5 |
| 写作质量 | 4 |
| 总体评分 | 4.1 |
