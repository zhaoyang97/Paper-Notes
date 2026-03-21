# MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization

**会议**: CVPR2026
**arXiv**: [2603.03192](https://arxiv.org/abs/2603.03192)
**代码**: 待确认
**领域**: llm_alignment
**关键词**: omni LLM, cross-modal hallucination, DPO, modality decoupling, audio-visual, preference optimization

## 一句话总结
提出 MoD-DPO（Modality-Decoupled DPO），通过不变性正则化、敏感性正则化和语言先验去偏三个机制解耦多模态 LLM 中各模态的贡献，有效缓解跨模态幻觉（如用听觉信息回答视觉问题），并推导出闭式最优策略。

## 背景与动机
全模态 LLM（Omni LLM）同时处理文本、视觉、音频等多种模态输入，是当前多模态智能的前沿方向。然而，这类模型面临一个独特且严重的问题——**跨模态幻觉（cross-modal hallucination）**：

1. **虚假相关性（Spurious Correlations）**：训练数据中不同模态的信息经常共现（如看到"狗"的画面时通常伴随"吠叫"的声音），模型学会利用这种统计相关性"走捷径"。当测试时这种相关性不成立时（如看到狗但没有声音），模型仍然会虚构另一个模态的信息
2. **语言先验支配（Dominant Language Priors）**：Omni LLM 的骨干通常是预训练的 LLM，其强大的语言先验会覆盖真实的多模态感知。例如，模型可能忽略实际音频内容，仅根据文本 prompt 中的线索（如"What sound..."）胡编一个"合理"的声音描述
3. **模态间干扰**：当一个模态的输入质量差或不相关时，模型不能正确忽略它，反而会被干扰

具体例子：给 Omni LLM 一个视频及问题"视频中的人在说什么？"，即使音频轨道被完全静音，模型也可能根据视觉中人的嘴型/场景"脑补"一段对话内容——这就是跨模态幻觉。

现有方法（如 vanilla DPO、mDPO）将多模态输入视为整体进行偏好优化，没有区分各模态的独立贡献，因此无法精准解决跨模态幻觉问题。

## 核心问题
如何让 Omni LLM 正确区分各模态的贡献——对相关模态敏感、对无关模态不敏感——从而消除跨模态幻觉？

## 方法详解

### 问题形式化
设 Omni LLM 的输入包含 $M$ 个模态 $\{x^1, x^2, \ldots, x^M\}$ 和文本 prompt $q$，输出为文本回答 $y$。对于一个特定的问题 $q$，只有部分模态是**相关的**（记为 $x^{rel}$），其他是**无关的**（记为 $x^{irr}$）。

跨模态幻觉的本质是：模型对 $x^{irr}$ 的变化过度敏感，或对 $x^{rel}$ 的变化不够敏感。

### 模态解耦策略

#### 1. 不变性正则化（Invariance Regularization）
核心思想：当一个无关模态被替换为随机噪声或其他样本时，模型的输出不应该变化。

给定原始输入 $(x^{rel}, x^{irr}, q)$，构造"损坏"输入 $(x^{rel}, \tilde{x}^{irr}, q)$，其中 $\tilde{x}^{irr}$ 是替换后的无关模态。不变性目标为：

$$\mathcal{L}_{\text{inv}} = D_{\text{KL}}\big(\pi_\theta(\cdot | x^{rel}, x^{irr}, q) \| \pi_\theta(\cdot | x^{rel}, \tilde{x}^{irr}, q)\big)$$

最小化这个 KL 散度，使模型在无关模态变化时输出保持稳定。

#### 2. 敏感性正则化（Sensitivity Regularization）
核心思想：当相关模态被损坏时，模型应该"意识到"输入已改变，输出应发生变化。

构造 $(x^{rel} \to \tilde{x}^{rel}, x^{irr}, q)$，敏感性目标为：

$$\mathcal{L}_{\text{sen}} = -D_{\text{KL}}\big(\pi_\theta(\cdot | x^{rel}, x^{irr}, q) \| \pi_\theta(\cdot | \tilde{x}^{rel}, x^{irr}, q)\big)$$

注意这里是**最大化** KL 散度（loss 取负），鼓励模型对相关模态的变化敏感。

#### 3. 语言先验去偏（Language-Prior Debiasing, LPD）
针对语言先验支配问题，引入惩罚项：

$$\mathcal{L}_{\text{LPD}} = \log \pi_\theta(y_w | q) - \log \pi_\theta(y_l | q)$$

其中 $y_w$ 和 $y_l$ 分别是 preferred 和 rejected 回答，$\pi_\theta(\cdot | q)$ 是仅给文本 prompt（不给任何多模态输入）时的输出概率。这个惩罚项抑制模型在没有模态输入时也能"自信"地给出回答的倾向——如果模型仅靠文本 prompt 就能区分 $y_w$ 和 $y_l$，说明它在依赖语言先验而非真实感知。

### MoD-DPO 总损失

$$\mathcal{L}_{\text{MoD-DPO}} = \mathcal{L}_{\text{DPO}} + \alpha \cdot \mathcal{L}_{\text{inv}} + \gamma \cdot \mathcal{L}_{\text{sen}} + \lambda \cdot \mathcal{L}_{\text{LPD}}$$

其中 $\mathcal{L}_{\text{DPO}}$ 是标准的 DPO 损失。作者进一步推导出 MoD-DPO 的**闭式最优策略**（类似原始 DPO 的 Bradley-Terry 推导），证明最优策略自然地对相关模态敏感、对无关模态不变。

### 偏好数据自动构建
从 10.8k 个视频出发，自动构建 18.1k 个偏好训练样本：
- 使用 GPT-4o 生成针对不同模态的问题（视觉问题、音频问题、联合问题）
- 对于每个问题，通过损坏相关/无关模态获取模型在不同条件下的回答
- 基于回答的模态一致性自动判定 preferred/rejected

### 训练效率优化
损坏输入的 forward pass 不需要计算梯度（仅用于计算 KL 目标），因此可以用 `torch.no_grad()` 高效执行。整体只需标准 DPO 的约 **1/4 训练 epoch** 即可收敛。

## 实验关键数据

### AVHBench（Audio-Visual Hallucination Benchmark）

| 方法 | Visual Acc ↑ | Audio Acc ↑ | Joint Acc ↑ | Avg ↑ |
|------|-------------|-------------|-------------|-------|
| Vanilla SFT | 61.2 | 58.7 | 55.3 | 58.4 |
| Vanilla DPO | 65.8 | 62.1 | 59.4 | 62.4 |
| mDPO | 67.3 | 63.5 | 60.1 | 63.6 |
| OmniDPO | 68.1 | 64.2 | 61.8 | 64.7 |
| **MoD-DPO** | **73.5** | **70.8** | **68.2** | **70.8** |

### CMM Benchmark（Cross-Modal Mismatch）

| 方法 | Audio→Visual ↓ | Visual→Audio ↓ | Language Prior ↓ | Overall Score ↑ |
|------|----------------|----------------|------------------|----------------|
| Vanilla DPO | 28.3 | 31.5 | 35.2 | 62.4 |
| mDPO | 25.1 | 28.7 | 32.8 | 63.6 |
| OmniDPO | 23.4 | 26.3 | 30.1 | 64.7 |
| **MoD-DPO** | **15.2** | **17.6** | **19.8** | **70.8** |

（↓ 表示跨模态幻觉率越低越好）

### 消融实验
- **不变性正则化**：去掉后 Audio→Visual 幻觉率增加 5.3%，说明不变性是抑制无关模态干扰的关键
- **敏感性正则化**：去掉后 Visual Acc 下降 3.2%，说明敏感性帮助模型更好地利用相关模态
- **LPD**：去掉后 Language Prior 幻觉率增加 8.1%，证实了语言先验去偏的必要性
- **训练效率**：MoD-DPO 在 1/4 epoch 后性能即超过 vanilla DPO 的完整训练，4 epoch 后进一步提升

## 亮点
- **问题定义精准**：跨模态幻觉是 Omni LLM 的核心问题，本文是首批系统性研究并提出解决方案的工作之一
- **模态解耦的三重机制**：不变性（对无关模态稳定）+ 敏感性（对相关模态响应）+ 语言去偏（抑制文本支配）三管齐下，设计逻辑完整
- **理论支撑**：推导出闭式最优策略，不是纯经验性的 loss 设计
- **自动数据构建**：从 10.8k 视频自动生成 18.1k 偏好样本，无需人工标注，可扩展性强
- **训练效率高**：无梯度 forward pass + 快速收敛，仅需 1/4 epoch 即可超越完整训练的 baseline

## 局限性 / 可改进方向
1. **模态数量限制**：当前主要在音视频两个模态上验证，扩展到更多模态（如触觉、深度、点云）时，损坏策略和正则化的设计可能需要调整
2. **相关/无关模态的判定**：自动数据生成依赖于预定义的"哪个模态与问题相关"规则，对于模态边界模糊的问题（如"描述整个场景"涉及所有模态）不完全适用
3. **损坏方式的影响**：不变性/敏感性正则化的效果可能依赖于损坏操作的具体方式（随机噪声 vs 替换为其他样本 vs 完全移除），不同损坏方式的比较不充分
4. **闭式解的实际差距**：虽然推导了闭式最优策略，但实际训练是近似优化，两者之间的差距未被量化
5. **只关注幻觉缓解**：对 Omni LLM 的整体能力（如多模态理解、生成质量）是否有退化未充分评估

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性解决 Omni LLM 跨模态幻觉，模态解耦思路新颖
- 实验充分度: ⭐⭐⭐⭐ 在两个 benchmark 上验证，消融完整，但仅限音视频两个模态
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法推导严谨
- 价值: ⭐⭐⭐⭐ 切中 Omni LLM 的核心痛点，方法可推广到更多多模态场景

