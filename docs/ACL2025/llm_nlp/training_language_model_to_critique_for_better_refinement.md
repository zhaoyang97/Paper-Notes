# Training Language Model to Critique for Better Refinement

**会议**: ACL 2025  
**arXiv**: [2506.22157](https://arxiv.org/abs/2506.22157)  
**代码**: https://github.com/publicstaticvo/critique  
**领域**: LLM NLP  
**关键词**: Critique-Refinement Loop, Critique Utility, DPO变体, 自动偏好学习, 多任务评估

## 一句话总结
提出 Refinement-oriented Critique Optimization（RCO），以"批判效用"（Critique Utility, CU）——即批判导致的精炼改善比例——作为奖励信号训练 critic 模型，通过 DPO 变体的 MSE 目标函数优化，无需直接评估批判质量；在对话生成、摘要、问答、数学推理、代码生成 5 个任务上，RCO 训练的 7B/13B critic 模型在 CU 和 RQS 指标上显著超过 70B 基线模型和 DPCO 方法。

## 研究背景与动机

1. **领域现状**：LLM 的批判能力（critique ability）是自动评估和自我改进的关键。近期工作通过人工标注批判数据 + SFT/RLHF 训练 critic 模型，取得一定进展。

2. **现有痛点**：(a) 现有方法训练 critic 模型是为了评估，而非为了驱动精炼改善——批判和精炼是割裂的；(b) 直接评估批判质量困难且主观——什么算"好的"批判缺乏客观标准；(c) 人工标注批判偏好成本高且质量不稳定。

3. **核心矛盾**：好的批判应该是能带来好的精炼结果的批判，但现有方法没有建立批判质量与精炼效果之间的因果链。

4. **本文要解决什么？** 设计以精炼效果为导向的 critic 训练方法——让批判质量直接由其带来的改善程度定义。

5. **切入角度**：构建一个闭环——critic 生成批判 → actor 基于批判精炼 → 评估精炼 vs 原始回答的偏好 → 偏好作为 critic 的奖励。

6. **核心idea一句话**：用精炼改善率（CU = 精炼优于原始的概率）作为 critic 模型的奖励信号，通过 DRO-style MSE 目标训练。

## 方法详解

### 整体框架
数据集 $\mathcal{D}$：prompt $x$ + 初始回答 $y_0$（actor 生成）→ critic 生成 $N$ 个批判 $c_1,...,c_N$ → actor 基于每个 $c_i$ 精炼 $M$ 个回答 $y_{i1},...,y_{iM}$ → 评估偏好计算 CU → 用 CU 作为奖励训练 critic。

### 关键设计

1. **Critique Utility（CU）定义**:
   - $CU(c_i | y_0, x) = P(y \succ y_0 | y \sim \pi_{c_i})$
   - 近似计算：$CU \approx \frac{1}{M}\sum_{j=1}^{M} PS(y_{ij}, y_0)$
   - $PS = 1$（精炼优于原始）、$0.5$（持平）、$0$（原始更好）
   - 由 Qwen-2.5-72B-Instruct 做偏好判断，交换位置避免位置偏差，每个批判共 10 次判断（$2M=10$）

2. **训练目标推导（DRO-style MSE Loss）**:
   - 起点：最大化 $\mathbb{E}_{c \sim p_\theta}[CU(c)] - \beta D_{KL}[p_\theta \| p]$
   - 最优解：$p^*(c) = \frac{p(c) \exp(\frac{1}{\beta}CU(c))}{Z_\beta}$
   - 关键：归一化常数 $Z_\beta$ 可通过 $N$ 个采样批判近似计算
   - 最终损失（Eq. 7）：$\mathcal{L}_{RCO} = \frac{1}{2N}\sum_i (\log\frac{p_\theta(c_i)}{p(c_i)} + \log Z_\beta - \frac{1}{\beta}CU(c_i))^2$
   - 优势：相比 DPO 只用二元偏好，RCO 利用连续标量 CU 值，更细粒度

3. **数据收集流程**:
   - **初始回答**：4 个 actor 模型（LLaMA-2-7B/13B/70B-Chat, LLaMA-3-8B-Instruct）× 10,000 prompts = 40,000 回答
   - **批判生成**：5 个 base critic 模型（LLaMA-2-7B/13B-Chat, LLaMA-3-8B, Auto-J-13B, UltraCM-13B），每个回答生成 $N=4$ 个批判
   - **精炼生成**：每个批判由生成初始回答的 actor 精炼 $M=5$ 个回答
   - **5 个任务**：对话生成、摘要、问答、数学推理、代码生成，来自 14 个数据集

## 实验关键数据

### 主实验表格（CU 和 RQS 评分）

| 模型 | 方法 | Dialog CU | Summ. CU | QA CU | Math CU | Code CU | Overall CU |
|------|------|-----------|----------|-------|---------|---------|------------|
| — | Initial Answer | — | — | — | — | — | 46.9 |
| LLaMA-2-70B-Chat | 基线 | 82.7 | 68.3 | 88.8 | 62.7 | 59.7 | 72.4 |
| LLaMA-3-70B-Inst | 基线 | 82.6 | 87.8 | 86.3 | 76.2 | 78.1 | 82.2 |
| Self-refinement | — | 75.2 | 77.6 | 79.9 | 64.5 | 65.8 | 72.6 |
| LLaMA-2-7B-Chat | Base | 83.5 | 63.4 | 87.1 | 60.1 | 59.7 | 70.8 |
| LLaMA-2-7B-Chat | +DPCO | 79.2 | 70.2 | 91.2 | 58.7 | 62.1 | 72.3 |
| LLaMA-2-7B-Chat | +**RCO** | **90.4** | **77.4** | **94.3** | **70.7** | — | — |

- RCO 训练的 7B 模型在 Overall CU 上超过 LLaMA-2-70B（72.4）和 Self-refinement（72.6）
- DPCO（直接批判偏好优化）提升有限，说明直接评估批判偏好不如通过精炼信号

### Refinement Accuracy 评估
- 在 GSM8K 上，RCO critic 辅助的精炼比 base critic 提升 3-5% accuracy
- 在 MBPP 代码生成上，RCO 精炼通过率比 self-refinement 提升 4-7%
- 跨模型泛化：用 LLaMA-3-8B 为 actor 时，RCO critic 也优于基线

### RewardBench 评估
- RCO 训练的模型在作为判别式 judge 时也有提升，说明 critique 训练同时增强了偏好判断能力

### 人工评估
- 200 样本人工评估：RCO 批判优于基线的比例显著更高
- 精炼偏好：基于 RCO 批判的精炼优于基于基线批判的精炼

### 关键发现
- CU 作为奖励信号比直接批判偏好更有效——好的批判 = 能带来好精炼的批判
- 连续标量 CU 值比二元偏好提供更丰富的训练信号
- 小模型（7B）经 RCO 训练可超过大模型（70B）的批判能力

## 亮点与洞察
- **闭环设计**：critique → refine → evaluate → reward 形成完整反馈循环，无需人工标注
- **CU 定义巧妙**：将难以定义的"批判质量"转化为可量化的"精炼改善率"，简洁有效
- **DRO-style 目标函数**：利用连续标量 CU 而非二元偏好，比标准 DPO 信息量更大
- **多任务泛化**：5 个任务 × 14 个数据集的覆盖面很广

## 局限性
1. CU 的计算依赖 Qwen-2.5-72B-Instruct 做偏好判断——judge 模型的bias会传递到训练
2. 每个批判需要精炼 $M=5$ 次、交换位置做 $2M=10$ 次偏好判断，数据收集成本高
3. Actor 和 Critic 使用不同模型，实际部署中可能希望统一模型同时具备两种能力
4. 未探讨 CU 信号随训练轮数的衰减——多轮迭代训练是否持续有效未验证

## 相关工作与启发
- 与 CriticGPT (McAleese et al., 2024) 的区别：后者用 RLHF + 人工标注训练 critic，RCO 用精炼信号自动获取奖励
- 与 Critic-CoT (Zheng et al., 2024) 的互补：Critic-CoT 关注逐步批判格式，RCO 关注批判的训练信号——两者可结合
- 启发：CU 的思想可推广——不只是 critique，任何中间输出的质量都可以通过其下游效果来度量

## 评分
- 新颖性: ⭐⭐⭐⭐ (CU 作为奖励信号的思路新颖)
- 理论深度: ⭐⭐⭐⭐ (DRO 推导完整，目标函数有理论基础)
- 实验充分性: ⭐⭐⭐⭐⭐ (5 任务 + CU/RQS/Accuracy/RewardBench + 人工评估)
- 实用价值: ⭐⭐⭐⭐ (可直接用于训练开源 critic 模型)
- 总体推荐: ⭐⭐⭐⭐ (critique-refinement 方向的扎实推进)
