# Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers

**会议**: NeurIPS 2025  
**arXiv**: [2505.13737](https://arxiv.org/abs/2505.13737)  
**代码**: https://github.com/andrewnam/causal_head_gating  
**领域**: LLM推理  
**关键词**: mechanistic interpretability, attention head, causal taxonomy, circuit discovery, Llama

## 一句话总结
提出 Causal Head Gating (CHG)，通过对 Transformer 的每个 attention head 学习一个可微门控系数并结合正/负正则化，将 head 分为促进（facilitating）、干扰（interfering）、无关（irrelevant）三类，无需人工标签或 prompt 模板即可发现因果子电路，并扩展为对比 CHG 以分离 ICL 和指令遵循的独立电路。

## 研究背景与动机

1. **领域现状**：机械可解释性（mechanistic interpretability）试图理解 LLM 内部计算。现有方法分两类：(a) 训练探针解码器从隐藏状态映射到可解释概念（相关性而非因果性）；(b) 因果中介分析（CMA）通过精心设计的 prompt 对定位特定行为的head（因果性强但规模受限）。
2. **现有痛点**：(1) 探针方法是相关性的，不能证明模型确实使用了检测到的特征；(2) CMA 需要手工设计 prompt 模板和明确的机制假设，难以扩展到复杂任务（如数学推理，不同题目结构差异巨大）；(3) 现有 head 剪枝工作主要在小模型/BERT 上，对现代 LLM 的适用性有限；(4) Gumbel-based 硬门控方法假设 head 独立，无法捕捉 head 间的交互依赖。
3. **核心矛盾**：如何在不依赖人工模板和标签的前提下，大规模地发现 LLM 中具有因果效应的注意力头？
4. **本文要解决什么？** 设计一种可扩展的方法来识别和分类 attention head 的因果角色，适用于任意任务和数据集。
5. **切入角度**：利用 next-token prediction 目标训练门控参数（每个 head 仅 1 个参数），通过正/负正则化产生变异，从而区分促进、干扰和无关 head。
6. **核心 idea 一句话**：用正/负 L1 正则化各拟合一次 head 门控值，促进 head 两次都保持高值，干扰 head 两次都被压低，无关 head 在两次间分化——由此建立三元因果分类。

## 方法详解

### 整体框架
对 Transformer 的 $L \times H$ 个 attention head 定义门控矩阵 $G \in [0,1]^{L \times H}$，$G_{\ell,h}$ 缩放 head $(\ell,h)$ 的输出。冻结模型参数，仅优化 $G$ 在 NLL + 正则化目标下。分别用 $\lambda > 0$（鼓励保留 -> 得到 $G^+$）和 $\lambda < 0$（鼓励移除 -> 得到 $G^-$）拟合两次，再根据两次结果的模式分类 head。

### 关键设计

1. **门控机制**:
   - 做什么：对每个 attention head 的输出施加可学习的缩放系数。
   - 核心思路：$Z_{\ell,h} = G_{\ell,h} \cdot (A_{\ell,h} V_{\ell,h})$，其中门控系数 $G_{\ell,h}$ 在 attention 计算后、output projection 前施加。冻结所有模型参数 $\theta$，仅优化 $G$。
   - 设计动机：每个 head 只增加 1 个参数，极低开销。直接在模型的计算图中施加干预，保证结果是因果性的而非相关性的。

2. **正则化分离**:
   - 做什么：通过正/负正则化将无关 head 从促进/干扰 head 中分离出来。
   - 核心思路：目标函数 $\mathcal{L} = \text{NLL} - \lambda \sum \sigma^{-1}(G_{\ell,h})$。$\lambda > 0$ 时，正则化鼓励门控趋向 1（保留所有 head），但 NLL 梯度会压低干扰 head -> 得到 $G^+$。$\lambda < 0$ 时，鼓励门控趋向 0（移除所有 head），但 NLL 梯度会保留促进 head -> 得到 $G^-$。关键洞察：如果 head 无关，其 NLL 梯度期望为 0，因此它的门控值完全由正则化决定——在 $G^+$ 中趋向 1，在 $G^-$ 中趋向 0，形成分化。
   - 设计动机：纯 NLL 优化无法区分促进 head（门控保持高）和无关 head（门控碰巧高），正则化打破这个混淆。

3. **三元因果分类**:
   - 做什么：根据 $G^+$ 和 $G^-$ 的模式将 head 分为三类。
   - 核心思路：**促进**：$G^+ \approx 1, G^- \approx 1$（两次都被保留）；**干扰**：$G^+ \approx 0, G^- \approx 0$（两次都被移除）；**无关**：$G^+ \approx 1, G^- \approx 0$（正则化主导）。促进分数 = $G^-$，干扰分数 = $1 - G^+$，无关分数 = $G^+ \times (1 - G^-)$。
   - 设计动机：避免了 Gumbel-based 方法的独立性假设。CHG 联合优化所有门控，捕捉 head 间的交互。

4. **对比 CHG（Contrastive CHG）**:
   - 做什么：分离执行同一任务的不同子电路（如 ICL vs 指令遵循）。
   - 核心思路：构造同一任务的两种变体（ICL 格式 vs 指令格式），拟合单个门控矩阵来"遗忘"一种格式同时保留另一种。目标函数包含两项：最大化遗忘变体的 NLL（忘记）+ 最小化保留变体的 NLL（记住）。
   - 设计动机：标准 CHG 只能发现"做任务需要哪些 head"，Contrastive CHG 进一步区分"理解任务"和"执行任务"用的是不同 head。

### 损失函数 / 训练策略
NLL + L1 正则化。先用 $\lambda=0$ 拟合初始化 $G$，再分别用 $\lambda > 0$ 和 $\lambda < 0$ 拟合。梯度裁剪确保 NLL 是主导项。每种设置用 10 个随机种子拟合。在 Llama-3 系列（1B/3B/8B）上实验。

## 实验关键数据

### 主实验：因果分类验证

通过 targeted ablation 验证 CHG 分类——按 facilitation/irrelevance/interference 分数排序 head 后逐个消融：
- 消融促进 head -> 性能下降（负 delta log-prob）
- 消融无关 head -> 性能不变（delta 约 0）
- 消融干扰 head -> 性能上升（正 delta log-prob）

三种模式在 4 个模型 x 3 个任务上全部一致验证成功。

### head 分布分析

| 任务 | Always Facilitating | Always Interfering | 说明 |
|------|--------------------|--------------------|------|
| Syntax | <5% | ~0% | 紧凑稀疏电路 |
| Common Sense | <5% | ~0% | 紧凑稀疏电路 |
| Math | 38.3% (3B) | 1.3% (3B) | 更大更刚性的电路 |

### 消融实验：Contrastive CHG

| 任务 | 保留格式 | 遗忘格式准确率 | 保留格式准确率 |
|------|---------|-------------|-------------|
| antonym (ICL) | ICL | 0% (遗忘成功) | 接近 baseline |
| singular-plural (Inst) | Instruction | 0% (遗忘成功) | 21% (有交叉) |

### 关键发现
- **CHG 分类与消融结果高度一致**：验证了因果性而非相关性。
- **数学推理需要更大、更刚性的电路**：52.6% 的 head 是 facilitating（vs syntax 25.6%），且跨种子一致性更高。
- **Head 角色不是模块化的**：同一个 head 在不同种子下可能是 facilitating 或 irrelevant，取决于其他 head 的配置。模型包含多个冗余的 sufficient sub-circuits。
- **ICL 和指令遵循使用可分离的电路**：Contrastive CHG 成功地遗忘一种而保留另一种，泛化到未见过的任务。
- **跨模型高度一致**：1B-8B 模型的 CHG 分布 Pearson 相关达 99.2%。
- **与 CMA 的互补性已验证**：CMA 识别的 head 在 CHG 中也有高 facilitation 分数（$t(53.77)=11.18$, $p<10^{-15}$）。

## 亮点与洞察
- **正/负正则化的对称设计**极其巧妙：一个简单的想法（用两次不同方向的正则化区分三类 head）产生了强大的分析工具。每个 head 仅 1 个参数，几分钟即可拟合，扩展性极好。
- **"Multiple sufficient sub-circuits"** 的发现很深刻：模型不是用固定电路做任务，而是有多个功能等价的子电路。这解释了为什么 head 剪枝通常损失不大——剩余子电路可以补偿。
- **Contrastive CHG 分离 ICL vs 指令遵循**是新颖的应用：第一次证明这两种能力在 head 层面是可分离的。

## 局限性 / 可改进方向
- **只分析 attention head，未涉及 MLP 层**：MLP 也存储和处理重要信息，未来可扩展到 MLP 神经元。
- **不能解释 head 具体做什么**：CHG 发现哪些 head 重要，但不说明它们执行什么计算。需要与 CMA 等方法结合使用。
- **依赖 NTP 目标**：如果任务不能很好地用 next-token prediction 衡量，CHG 可能不适用。
- **建议方向**：将 CHG 扩展到 MLP 层 / residual stream 的 feature 级别分析。

## 相关工作与启发
- **vs CMA**: CMA 高精度低灵敏度（假设驱动），CHG 高灵敏度可中等精度（数据驱动）。两者互补。
- **vs Sparse Autoencoders**: SAE 发现可解释特征但是相关性的，CHG 建立因果链接。
- **vs Gumbel-based gating**: Gumbel 方法假设 head 独立（factorized），CHG 联合优化捕捉交互。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 正/负正则化三元分类设计优美，Contrastive CHG 新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型多任务，与 CMA 交叉验证，消融完整
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，图表精美
- 价值: ⭐⭐⭐⭐⭐ 提供了一个轻量、可扩展、因果性的 head 分析工具
