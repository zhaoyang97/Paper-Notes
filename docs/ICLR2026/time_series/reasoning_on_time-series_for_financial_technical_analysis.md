# Reasoning on Time-Series for Financial Technical Analysis

**会议**: ICLR2026  
**arXiv**: [2511.08616](https://arxiv.org/abs/2511.08616)  
**代码**: [chen-jan/VTA](https://github.com/chen-jan/VTA)  
**领域**: llm_nlp  
**关键词**: 时间序列推理, 金融技术分析, 强化学习, LLM微调, 可解释预测  
**作者**: Kelvin J.L. Koa, Jan Chen, Yunshan Ma, Huanhuan Zheng, Tat-Seng Chua (NUS, TUM, SMU, CityU HK)

---

## 一句话总结

提出 Verbal Technical Analysis (VTA) 框架，结合 LLM 的语言推理能力与时间序列模型的模式捕捉能力，通过 Time-GRPO 强化学习优化推理链，并以推理属性条件化时序预测，实现了兼具准确性和可解释性的金融时间序列预测。

---

## 研究背景与动机

1. **LLM 在金融中的局限**：现有金融 LLM 主要分析文本报告（财报问答、情感分析），但忽略了对历史价格数据的可解释分析，即技术分析（Technical Analysis），而这对交易从业者极为重要。
2. **LLM 不擅长时序推理**：已有研究 (Merrill et al., 2024) 表明 LLM 在零样本时序推理上表现"remarkably bad"，直接输入原始时序数据效果很差。
3. **时序 LLM 牺牲可解释性**：Time-LLM、CALF 等方法通过修改嵌入空间来输出时序预测，但 LLM 丧失了自然语言推理能力，无法提供可解释的分析。
4. **现有可解释方案不足**：最接近的 TimeCAP 仅产出分类标签预测而非完整时序轨迹，且其推理依赖外部辅助数据而非内生信号。
5. **跨域挑战**：任务需要在两个域之间切换——输入/输出为时序域（股价），推理过程为自然语言域，这增加了建模难度。
6. **金融时序的内在可解释信号**：与一般时序不同，金融数据包含大量经专家研究的技术指标（MACD、RSI、布林带等），为语言化推理提供了天然抓手。

---

## 方法详解

### 整体框架

VTA 框架包含三个核心组件：

- **Time-Series Reasoning（时序推理）**：教 LLM 对时序输入进行语言推理
- **Time-Series Forecasting（时序预测）**：用 backbone 时序模型捕捉底层复杂模式
- **Joint Conditional Training（联合条件训练）**：将推理属性条件化注入时序预测

### 问题形式化

给定历史 $T$ 个交易日的输入 $\mathbf{X} = \{\mathbf{x}_{t-T+1}, \ldots, \mathbf{x}_t\}$，其中 $\mathbf{x}_t = [o_t, h_t, l_t, v_t, c_t, p_t]$（开高低量收+调整收盘价），目标生成：

- 语言推理轨迹 $\mathbf{v}$
- 未来 $T'$ 交易日的价格预测 $\mathbf{y} = \{p_{t+1}, \ldots, p_{t+T'}\}$

实验中 $T = T' = 10$（短期预测）。

### 时序推理 (Time-GRPO)

**文本标注化**：将原始时序数据转换为文本标注 $\mathbf{X'} = \mathbf{f}(\mathbf{X})$，包括统计信息（均值/最小值/最大值）和金融技术指标（均线、动量、MACD、RSI、布林带等）。

**Time-GRPO 目标**：基于 GRPO (Shao et al., 2024) 修改设计，核心公式：

$$\mathcal{L}_{\text{time-grpo}}(\theta) = \mathbb{E}_{\mathbf{q} \sim \mathcal{Q}} \frac{1}{G} \sum_{i=1}^{G} \left( \min\left(\frac{\pi_\theta(\mathbf{o_i}|\mathbf{q})}{\pi_{\theta_{\text{old}}}(\mathbf{o_i}|\mathbf{q})} A_i, \text{clip}(\cdot, 1{-}\epsilon, 1{+}\epsilon) A_i \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right)$$

**逆 MSE 奖励**：使推理链最大化预测准确性：

$$r_{\text{MSE}}(\theta) = \frac{1}{\lambda \cdot \|\hat{\mathbf{y}}_\theta - \mathbf{y}\|_2^2}$$

使用逆 MSE 是因为奖励需要最大化（MSE 越小则奖励越大）。

**多阶段训练流水线**：

1. **Cold-Start 阶段**：用 Time-GRPO 生成初始训练样本，性能提升有限但为后续提供数据
2. **拒绝采样 + SFT 阶段**：保留 MSE 在各 bucket 中前 10% 的推理链，进行监督微调
3. **RL 优化阶段**：在已学会推理的基础上再用 Time-GRPO 搜索最优推理策略

### 时序预测 Backbone

基于 GPT-2，通过跨模态微调：

- 时序输入经 Embedding + Multi-head Attention → 投影时间 token $\mathbf{X}_{\text{time}}$
- 对 LLM 词嵌入做 PCA 得到主成分词嵌入 $\hat{\mathbf{D}}$
- 通过 Multi-head Cross-Attention 对齐时间 token 与词嵌入：

$$\mathbf{X}_{\text{text}} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{C}}\right)\mathbf{V}$$

- 逐层特征正则化对齐 temporal 和 text 分支：

$$\mathcal{L}_{\text{feature}} = \sum_{n=1}^{N} \gamma^{(N-n)} \text{sim}\left(\phi_{\text{text}}^n(\mathbf{F}_{\text{text}}^n), \phi_{\text{time}}^n(\mathbf{F}_{\text{time}}^n)\right)$$

### 联合条件训练

从推理输出提取描述性属性类 $\mathbf{c}$（最大值/最小值/均值），条件化时序预测：

$$\mathcal{L}_{\text{forecast}}(\phi) = \mathbb{E}_{\mathbf{X}, \mathbf{y}, \mathbf{c}} \left[\|\hat{\mathbf{y}}_\psi(\mathbf{X}, \tilde{\mathbf{c}}) - \mathbf{y}\|^2\right]$$

以概率 $p_{\text{uncond}}=0.3$ 随机将 $\mathbf{c}$ 置为空（类似 Classifier-Free Guidance 思想），同时训练条件/无条件路径。推理时：

$$\hat{\mathbf{y}} = s \cdot \hat{\mathbf{y}}_\psi(\mathbf{X}, \mathbf{c}) + (1-s) \cdot \hat{\mathbf{y}}_\theta(\mathbf{X})$$

其中引导尺度 $s=0.1$。

---

## 实验

### 主实验：预测性能对比

**数据集**：ACL18 StockNet（88 只美股，2012-2017）+ Dow Jones/China A50/EURO STOXX 50 (2024)

| 模型 | StockNet MSE | StockNet MAE | All MSE | All MAE |
|------|-------------|-------------|---------|---------|
| GPT-4.1 mini | 0.0846 | 0.1827 | 0.2014 | 0.2376 |
| DeepSeek-R1 | 0.0788 | 0.1853 | 0.1428 | 0.2323 |
| TimesNet | 0.0708 | 0.1789 | 0.1286 | 0.2229 |
| TimeLLM | 0.0704 | 0.1780 | 0.1262 | 0.2210 |
| CALF | 0.0674 | 0.1738 | 0.1235 | 0.2180 |
| **VTA (Ours)** | **0.0659** | **0.1701** | **0.1178** | **0.2122** |

VTA 在全部 4 个数据集上取得最优 MSE 和 MAE，整体 MSE 提升 4.6%，MAE 提升 2.7%。

### 消融实验：多阶段训练的贡献

| 阶段 | Llama-3.1-8B MSE | Qwen-2.5-3B MSE | Qwen-2.5-7B MSE |
|------|-----------------|-----------------|-----------------|
| Base Model | 0.1482 | 0.1707 | 0.0949 |
| Cold Start (RL) | 0.1475 | 0.1648 | 0.0941 |
| SFT for Reasoning | 0.1168 | 0.1032 | 0.0893 |
| RL for Reasoning | 0.0955 | 0.0832 | 0.0686 |
| + Conditioning (VTA) | 0.0667 | 0.0672 | **0.0659** |

**关键发现**：
- Cold-Start RL 仅提升 1.6%（平均），但其生成的数据是后续阶段的基础
- 拒绝采样 + SFT 后再做 RL，提升幅度达 20.3%，证明多阶段流水线的有效性
- 条件化 backbone 模型进一步降低误差，说明"外部推理 + 内部模式"互补有益
- Qwen-2.5-7B 作为推理模型效果最佳，但加上条件训练后 3B 模型也能达到接近水平

### 推理质量评估

25 位金融行业专家（来自 JPMorgan、UBS、Evercore、Allianz 等）对 VTA、GPT-4.1 mini、DeepSeek-R1 的推理链进行 1-5 分盲评：

- **Depth（深度）、Accuracy（准确性）、Relevance（相关性）**：VTA 显著领先，反映其技术指标使用和推理能力
- **Coherence（连贯性）、Clarity（清晰度）**：差距较小，通用 LLM 本身在文本流畅度上有优势

### 投资组合评估

| 模型 | Returns | Volatility | Max Drawdown | Sharpe Ratio |
|------|---------|-----------|-------------|-------------|
| TimeLLM | 0.2185 | 0.1193 | -0.1040 | 1.5230 |
| CALF | 0.2019 | 0.1247 | -0.0981 | 1.4566 |
| **VTA (Ours)** | **0.2409** | **0.1185** | **-0.0883** | **1.7190** |

VTA 在 Sharpe Ratio 上大幅领先（1.7190 vs 次优 1.5230），证明在真实投资场景中的实用价值。

### 推理扰动实验

- 移除技术指标 → 预测性能明显下降，说明推理链确实提供了有用的指导信号
- 添加对抗噪声 → 性能下降但趋势不一致，可能因模型在联合训练中学会了在推理不可靠时更多依赖时序 backbone

---

## 亮点

1. **巧妙的跨域桥接**：利用金融技术指标作为时序与语言域之间的天然桥梁，解决了 LLM 不擅长直接处理原始时序的问题
2. **Time-GRPO 设计**：用逆 MSE 作为 RL 奖励，直接以预测精度驱动推理链优化，无需人工标注推理数据
3. **多阶段训练流水线**：Cold-Start → 拒绝采样 SFT → RL 的渐进设计使训练更稳定高效
4. **Classifier-Free Guidance 思想迁移**：将扩散模型中的条件引导技术应用于时序预测，同时训练条件/无条件路径
5. **全面的评估体系**：不仅评预测精度，还包括行业专家推理质量评分和 Markowitz 投资组合验证

---

## 局限性 / 可改进方向

1. **仅限金融时序**：跨域实验（医疗/能源）表明，VTA 的推理优势依赖金融技术指标的内在可解释信号，对一般时序数据退化为简单趋势外推
2. **短期预测**：$T=T'=10$ 仅覆盖短期交易场景，长期预测有效性未验证
3. **推理与预测的对齐**：条件化仅用了最大/最小/均值等简单属性，推理链中更丰富的信息（趋势方向、指标信号）未充分利用
4. **引导尺度固定**：$s=0.1$ 说明模型实际上主要依赖 backbone，推理引导的实际贡献比例较低
5. **计算成本**：多阶段 RL + LLM 推理 + 时序 backbone 联合训练，资源消耗较大
6. **基座模型选择**：仅测试了 3 个 LLM 基座（Llama-3.1-8B、Qwen-2.5-3B/7B），未探索更大规模模型

---

## 相关工作对比

| 方向 | 代表工作 | 与 VTA 的区别 |
|------|---------|-------------|
| 金融 LLM | Fin-R1, FinMem, SEP | 分析文本报告/新闻，不处理价格时序 |
| 时序 LLM | Time-LLM, CALF | 修改嵌入空间，失去语言推理能力 |
| 时序推理 | TimeCAP | 依赖外部辅助数据，仅产出分类标签 |
| LLM 时序推理 | Merrill et al. | 发现 LLM 零样本时序推理差，VTA 通过 RL 微调解决 |
| 推理优化 | DeepSeek-R1, GRPO | VTA 将 GRPO 适配为 Time-GRPO，用逆 MSE 奖励 |

---

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将 RL 推理优化（GRPO）与时序预测结合，Classifier-Free Guidance 迁移到时序条件化，思路新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4 个数据集、14+ 基线、消融实验、专家评估、投资组合验证、跨域泛化分析，非常全面
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机充分，图表丰富
- **实用价值**: ⭐⭐⭐⭐ — 可解释的金融预测对从业者有直接价值，Sharpe Ratio 验证了实际投资潜力
