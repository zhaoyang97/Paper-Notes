# Don't Let It Fade: Preserving Edits in Diffusion Language Models via Token Timestep Allocation

## 基本信息
- **arXiv**: 2510.26200
- **会议**: NeurIPS 2025
- **作者**: Woojin Kim, Jaeyoung Do (Seoul National University)
- **机构**: AIDAS Laboratory, Seoul National University
- **代码**: 计划开源 (Apache 2.0)

## 一句话总结
提出 Token Timestep Allocation (TTA-Diffusion)，通过为每个 token 分配独立的去噪时间步来解决扩散语言模型中 classifier guidance 导致的 update-forgetting 问题，实现可控文本生成的稳定性和效率大幅提升。

## 背景与动机
扩散语言模型 (DLM) 通过迭代去噪实现文本生成，classifier guidance 可注入外部梯度来引导生成方向（如情感控制、去毒化）。然而作者发现一个核心失败模式——**update-forgetting**：由于所有 token 共享统一的、上下文无关的噪声更新，前面时间步通过 classifier guidance 修改的语义编辑（如将"hate"改为"love"）会在后续去噪步骤中被覆盖或撤销。这导致三个问题：
1. **流畅性下降**：过度 fluctuation 破坏 token 间的连贯性
2. **可控性削弱**：关键语义编辑被遗忘，classifier 的引导效果无法持续
3. **效率低下**：需要大量步骤（>200）才能逐步强制控制，计算开销巨大

## 核心问题
如何在扩散语言模型的推理过程中保持 classifier-guided 的语义编辑不被后续去噪步骤覆盖？

## 方法详解

### 1. 问题形式化
定义两个关键概念：
- **Diffusion Fluctuation** $R_t = \text{dist}(x_{t+1}, x_{t+1}^{in})$：单步去噪的输入输出偏差
- **Update-Forgetting** $F_t = \text{dist}(x_t^{guided}, x_{t+1})$：guidance 效果在下一步的语义漂移

实验验证：fluctuation 与 perplexity 强正相关 ($r = 0.86$)；关键 token 被修改时 classifier 置信度下降超过 10%。

### 2. Token Timestep Allocation (TTA)
核心思想：不再对所有 token 施加统一时间步，而是为每个 token 分配独立的局部时间步 $t_i = f(i, t)$。时间步越大 → 噪声越强 → 去噪修改越大；时间步越小 → token 越"冻结"。

**固定策略**：线性 schedule $f_{linear}(i,t) = \lfloor \frac{i}{N-1} t \rfloor$，让序列前面的 token 更早稳定。

**自适应策略**：利用 classifier 梯度作为 token 重要性指标：
$$t_i^{adaptive} = \alpha_{smooth} \cdot t + (1 - \alpha_{smooth}) \cdot (1 - \hat{g}_i) \cdot t$$
其中 $\hat{g}_i$ 是归一化梯度幅值。梯度大的 token 说明已被 classifier 充分引导 → 分配更小时间步 → 减少后续噪声扰动 → 保留编辑效果。

### 3. Progressive Step Reduction
从 $T=5000$ 步模型逐步微调到 $T \in \{1000, 200, 50\}$，使用交叉熵损失直接优化（不需要蒸馏），实现推理加速。

### 4. 理论支撑
- 过度 fluctuation 下界了每步去噪 KL 散度，进而提升 perplexity 上界
- TTA 的自适应分配等价于在固定噪声预算下最小化交叉熵上界和 margin-drop 上界的 KKT 解
- $\sigma_i^2 \propto (1 - \hat{g}_i)$ 是 Pareto 最优解

## 实验关键数据

### 去毒化 (RealToxicityPrompts)
| 方法 | Avg. Tox↓ | Max. Tox↓ | PPL↓ |
|---|---|---|---|
| DExperts | 15.1 | 32.0 | 48.0 |
| SSD-LM (T=1000) | 24.6 | 50.3 | 58.3 |
| **TTA (T=200)** | **12.2** | **26.0** | **—** |
| **TTA (T=50)** | **12.5** | **—** | **59.5** |

### 情感控制 (PPLM prompts)
| 方法 | Acc↑ | PPL↓ |
|---|---|---|
| LM-Steer | 85.4 | 78.8 |
| TESS (T=1000) | 82.6 | 42.8 |
| **TTA (T=200)** | **92.1** | **23.2** |
| **TTA (T=50)** | **85.9** | **40.2** |

准确率比最强基线提升 >20%，PPL 几乎减半，且仅需 1/5 的步数。

### 词汇约束生成
| 方法 | Syntax Tree Acc | Mean PPL |
|---|---|---|
| Diffusion-LM | 86.0 | 248.6 |
| **TTA** | **93.1** | **111.4** |

### 跨扩散框架泛化
- 连续扩散：Diffusion-LM 上准确率 72.8% → 75.6%，PPL 89.3 → 77.9
- 离散扩散：D-CBG validity 98% → 99%，mean property 0.474 → 0.494

## 亮点
1. **问题定义精准**：update-forgetting 的形式化定义和实验验证非常扎实
2. **推理时方法，无需训练**：TTA 纯推理时操作，可直接应用于已有 DLM
3. **理论与实践统一**：从 KKT 最优解推导出自适应分配规则，理论严谨
4. **速度效率飞跃**：50 步就能超越 200+ 步基线，速度提升 5-10 倍
5. **通用性强**：适用于 simplex、连续、离散三种扩散框架

## 局限性
1. 仅在 330M 级别模型验证，未扩展到大规模 DLM (如 LLaDA, MDLM 等)
2. 依赖外部 classifier 的梯度信号，classifier 质量直接影响效果
3. 单属性控制为主，多属性联合控制未深入探索
4. RoBERTa-large 作为 backbone 的生成能力本身有限

## 与相关工作的对比
- **vs. Diffusion-LM/SSD-LM**：同为扩散文本生成，但 TTA 解决了它们的 update-forgetting 问题
- **vs. AR-Diffusion**：AR-Diffusion 在训练时分配时间步（基于位置），TTA 在推理时分配（基于语义重要性）
- **vs. MDLM/Simple Diffusion**：离散扩散通过 unmasking 排序实现"硬"排序，TTA 是"软"排序更灵活
- **vs. PPLM/DExperts**：AR 方法受限于序列依赖无法修改已生成 token，DLM 天然支持修改但有 forgetting 问题
- **vs. Token ordering (Kim et al., ICML 2025)**：互补工作，理论分析排序重要性，TTA 给出实际推理方案

## 启发与关联
- **与 SANA-Sprint/DiCo 的关联**：都关注推理效率，但在文本 vs 图像领域。TTA 的 progressive step reduction 与图像扩散的步数压缩异曲同工
- **对 LLaDA 等大规模 DLM 的启示**：随着 DLM 扩展到 LLM 规模，update-forgetting 可能更严重，TTA 有直接应用价值
- **可控生成的范式转变**：从"加更多步来强制控制"到"用更少步但更聪明地分配"——效率优先的控制思路

## 评分
- 新颖性：★★★★☆ — update-forgetting 的发现和 TTA 的设计独特
- 技术深度：★★★★★ — 理论推导完整，从问题定义到 KKT 解
- 实验完整度：★★★★☆ — 多任务多框架验证充分，缺少大规模 DLM 验证
- 写作质量：★★★★★ — 逻辑清晰，从现象到原因到解决方案环环相扣
