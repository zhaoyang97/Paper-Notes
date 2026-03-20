# When to Speak, When to Abstain: Contrastive Decoding with Abstention

**会议**: ACL 2025  
**arXiv**: [2412.12527](https://arxiv.org/abs/2412.12527)  
**代码**: 无  
**领域**: 其他  
**关键词**: contrastive decoding, abstention, knowledge conflict, parametric vs contextual, training-free

## 一句话总结
提出 CDA（Contrastive Decoding with Abstention），一种免训练解码方法，通过熵校准的不确定性估计让 LLM 在参数/上下文知识可用时生成正确回答、在两者都不可靠时主动弃权，覆盖全部四种知识可用性场景。

## 研究背景与动机

1. **领域现状**：LLM 同时拥有参数知识 $\mathcal{P}$（预训练获得）和上下文知识 $\mathcal{C}$（推理时提供），现有对比解码（CCD）方法通过对比两种知识的输出分布来选择更可靠的知识源。
2. **现有痛点**：CCD 假设至少一种知识可用，但现实中常遇到两者都不可靠的情况——此时 LLM 应弃权而非强行回答（会产生幻觉）。
3. **核心矛盾**：弃权需要精确评估知识相关性并将其整合到生成过程——两者都具有挑战性。
4. **本文要解决什么？** 设计一个覆盖所有四种场景的免训练解码方法：$\mathcal{P}$=1,$\mathcal{C}$=1（回答）；$\mathcal{P}$=1,$\mathcal{C}$=0（依赖参数）；$\mathcal{P}$=0,$\mathcal{C}$=1（依赖上下文）；$\mathcal{P}$=0,$\mathcal{C}$=0（弃权）。

## 方法详解

### 整体框架
三个输出分布的加权融合：$d^o_t = w^p_t \cdot d^p_t + w^c_t \cdot d^c_t + (1-w^p_t-w^c_t) \cdot d^a_t$，其中 $d^p_t$ 来自参数模板，$d^c_t$ 来自上下文模板，$d^a_t$ 来自显式弃权指令模板。

### 关键设计

1. **三分布融合**:
   - $d^p_t = \text{logit}_\theta(y_t | \mathcal{T}_p(x, y_{<t}))$（无上下文的参数分布）
   - $d^c_t = \text{logit}_\theta(y_t | \mathcal{T}_c(c, x, y_{<t}))$（有上下文的分布）
   - $d^a_t = \text{logit}_\theta(y_t | \mathcal{T}_a(c, x, y_{<t}))$（弃权指令分布）
   - 弃权权重 $w^a_t = 1 - w^p_t - w^c_t$：当两种知识都不确定时自动增大。

2. **熵校准的知识相关性估计**:
   - 计算参数/上下文分布的熵 $\mathcal{H}^p_t$, $\mathcal{H}^c_t$。
   - 用"内容无关"空提示（占位符替换具体输入）计算 null 分布的基准熵 $\bar{\mathcal{H}}^p_t$, $\bar{\mathcal{H}}^c_t$。
   - 相关性 = 知识提供的额外信息 = $r^p_t = \frac{\max(\mathcal{H}^p_t - \bar{\mathcal{H}}^p_t, 0)}{\bar{\mathcal{H}}^p_t}$。
   - 归一化得到最终权重：$w^p_t = \frac{r^p_t}{r^p_t + r^c_t} \cdot r^p_t$。
   - 设计动机：直接比较 $\mathcal{H}^p$ 和 $\mathcal{H}^c$ 不公平（条件不同），校准消除了模型内在偏差。

3. **动量平滑 (CDA-m)**:
   - $w_t \leftarrow \alpha \cdot w_{t-1} + (1-\alpha) \cdot w_t$，防止权重在相邻 token 间剧烈跳变。

### 受控测试台
精心构造四场景数据集：用生成一致性率估计参数知识（$r=0$→无，$r>\eta$→有），用 SBERT 相似度选择不相关上下文，确保每个场景的知识状态可控。

## 实验关键数据

### 主实验（4 LLM × 3 QA 数据集）

| 场景 | 期望行为 | CDA 表现 |
|------|---------|----------|
| P=1, C=1 | 正确回答 | ✓ 准确率高 |
| P=1, C=0 | 依赖参数 | ✓ 不被噪声误导 |
| P=0, C=1 | 依赖上下文 | ✓ 准确跟随 |
| P=0, C=0 | **弃权** | ✓ **有效弃权** |

CDA 在所有场景下均超越现有 CCD 方法（Zhao et al., Shi et al.等），且超越需要训练的弃权方法（Zhang et al.等）在泛化性上表现更优。RAG 场景同样有效。

## 亮点与洞察
- **"内容无关"空提示做熵校准**非常巧妙：用占位符替换输入消除语义信息，仅保留模板偏差，作为校准基准。
- **三分布融合的自动弃权机制**：不需要显式判断"是否弃权"，而是通过权重自然衰减到弃权分布——优雅且无需阈值设置。
- **免训练**：直接应用于任何现成 LLM，不改参数不改架构——实用性极强。

## 局限性 / 可改进方向
- 弃权指令模板的设计可能影响效果，需要良好的 prompt engineering。
- 每步解码需要 3 次前向传播（参数/上下文/弃权），推理成本约 3x。
- 熵校准中"内容无关"提示的选择可能不够鲁棒。

## 相关工作与启发
- **vs 现有 CCD (Zhao et al. 2024)**: 仅对比参数和上下文，不处理两者都不可靠的场景；CDA 扩展到含弃权的三分布。
- **vs 训练型弃权 (Zhang et al. 2024)**: 需微调 LLM 学习弃权能力；CDA 免训练且泛化更好。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将弃权自然整合到对比解码框架中
- 实验充分度: ⭐⭐⭐⭐⭐ 4 场景精心控制 + 4 LLM + RAG 验证
- 写作质量: ⭐⭐⭐⭐ 测试台设计严谨，公式推导清晰
- 价值: ⭐⭐⭐⭐⭐ 免训练提升 LLM 可靠性，对部署极具实用价值
