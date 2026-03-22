# $\varphi$-DPO: Fairness Direct Preference Optimization Approach to Continual Learning in Large Multimodal Models

**会议**: CVPR2026  
**arXiv**: [2602.22601](https://arxiv.org/abs/2602.22601)  
**代码**: 待确认  
**领域**: llm_alignment  
**关键词**: continual learning, DPO, fairness, catastrophic forgetting, large multimodal model, focal loss

## 一句话总结
提出 $\varphi$-DPO，将 DPO 作为持续学习范式（以前一步模型为参考策略），并引入受 focal loss 启发的公平性调制因子 $(1-p)^\gamma$ 来平衡不同数据组间的梯度贡献，在理论上证明 $\gamma \to \infty$ 时梯度偏差趋于零，在 CoIN 和 MLLM-CL 基准上达到 SOTA。

## 背景与动机
大型多模态模型（LMM）在实际部署中需要不断学习新任务，持续学习（Continual Learning, CL）是实现这一目标的关键能力。然而，LMM 的持续学习面临双重挑战：

### 挑战一：灾难性遗忘
这是持续学习的经典问题——学习新任务时旧任务性能退化。现有缓解方法包括：
- **经验回放（Experience Replay）**：存储旧任务数据用于复习，但存储开销大，且可能违反隐私约束
- **正则化方法（EWC, LwF 等）**：通过参数约束限制旧知识的覆写，但约束过强会限制新任务学习
- **知识蒸馏**：用旧模型的输出作为软标签指导新模型，但需要额外的前向传播开销

### 挑战二：公平性问题
这是本文新发现的一个被忽视的问题——**持续学习中的数据不平衡导致的公平性退化**：

1. **不同数据组大小差异大**：持续学习的不同阶段数据量差异悬殊（如第一阶段 10 万样本，第二阶段仅 1 万样本），经验回放时旧数据远多于新数据
2. **梯度被支配**：数据量大的组贡献更多梯度，数据量小的组被"淹没"，导致模型在小组上表现差
3. **群体公平性**：对于不同的用户群体或数据来源，模型性能的差异构成潜在的公平性风险

传统 CL 方法几乎不考虑公平性，而公平性方法（如 DRO、FairBatch）不考虑遗忘。$\varphi$-DPO 的动机正是**同时解决这两个问题**。

### 核心洞察：DPO 天然适合持续学习
标准 DPO 的损失函数依赖一个**参考策略** $\pi_{\text{ref}}$，其作用是防止优化后的策略偏离参考太远。作者发现，如果将 $\pi_{\text{ref}}$ 设定为**上一持续学习步骤的模型** $\pi_{t-1}$，那么 DPO 本身就隐式地实现了知识蒸馏效果——KL 散度约束自然地限制了新模型与旧模型的偏差，从而缓解遗忘。

## 核心问题
如何将DPO 改造为同时解决持续学习中灾难性遗忘和公平性退化的统一框架？

## 方法详解

### DPO 作为持续学习范式

在持续学习的第 $t$ 步，模型从 $\pi_{t-1}$ 更新到 $\pi_t$。标准 DPO 损失为：

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{t-1}) = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{t-1}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{t-1}(y_l|x)}\right)\right]$$

其中 $y_w$ 是 preferred 回答，$y_l$ 是 rejected 回答，$\beta$ 为温度参数。参考策略 $\pi_{t-1}$ 是上一步的模型，这意味着 DPO 隐式地惩罚新策略偏离旧策略太远。

#### 理论连接：DPO 与知识蒸馏
作者在 Lemma 1-2 中证明了 DPO 损失与 KL 散度的上下界关系：

$$c_1 \cdot D_{\text{KL}}(\pi_{t-1} \| \pi_\theta) \leq \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{t-1}) \leq c_2 \cdot D_{\text{KL}}(\pi_{t-1} \| \pi_\theta) + C$$

其中 $c_1, c_2, C$ 是与 $\beta$ 相关的常数。这表明**最小化 DPO 损失等价于隐式最小化新旧模型的 KL 散度**，即进行了知识蒸馏。这为"DPO 天然适合 CL"提供了理论基础。

### $\varphi$-DPO：公平性调制

尽管 DPO 能缓解遗忘，但不能处理数据不平衡导致的公平性问题。受 focal loss 启发，$\varphi$-DPO 引入调制因子：

$$\mathcal{L}_{\varphi\text{-DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[(1-p_{w,l})^\gamma \cdot \log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{t-1}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{t-1}(y_l|x)}\right)\right]$$

其中 $p_{w,l} = \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{t-1}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{t-1}(y_l|x)}\right)$ 是模型对当前偏好对的"置信度"。

#### 调制机制的直觉
- 当模型对某个偏好对**已经很自信**（$p_{w,l}$ 接近 1）时，$(1-p_{w,l})^\gamma$ 接近 0，梯度贡献被下调——模型对"已学好"的样本不再浪费梯度
- 当模型对某个偏好对**不自信**（$p_{w,l}$ 接近 0）时，$(1-p_{w,l})^\gamma$ 接近 1，梯度贡献保持不变——模型集中精力学习"困难"样本
- $\gamma$ 越大，梯度重分配越激进

#### 公平性理论保证（Lemma 3）
设不同数据组 $g \in \{1, \ldots, G\}$，每组的梯度偏差为：

$$B_\gamma(\theta) = \max_{g_1, g_2} \left|\frac{\nabla_\theta \mathcal{L}_{\varphi}^{g_1}}{\nabla_\theta \mathcal{L}_{\varphi}^{g_2}}\right|$$

作者证明：当 $\gamma \to \infty$ 时，$B_\gamma(\theta) \to 0$，即无论数据分布多么不平衡，足够大的 $\gamma$ 都能使各组的梯度贡献趋于相等。直觉上，这是因为大 $\gamma$ 值会让模型只关注"各组中最困难的样本"——而各组最困难样本的数量是均衡的。

### 偏好对构建
本文针对 CoIN 和 MLLM-CL 两个持续学习基准构建偏好对：

1. **Preferred 回答 $y_w$**：人工标注的 ground truth 回答
2. **Rejected 回答 $y_l$**：
   - 用 LLM（如 GPT-4）基于 ground truth 生成"合理但错误"的回答（如事实错误、细节偏差）
   - 人工验证确保 rejected 回答确实劣于 preferred
3. 每个 $(x, y_w, y_l)$ 三元组附带组别标签 $g$，用于计算公平性指标

### 与其他 CL 方法的联合使用
$\varphi$-DPO 与经验回放自然兼容：回放缓冲区中的旧数据和新数据分属不同组，公平性调制因子自动平衡两者的梯度贡献。

## 实验关键数据

### CoIN Benchmark（分 8 个任务阶段）

| 方法 | Final Avg Acc ↑ | Forgetting ↓ | Fairness (Worst-group Gap) ↓ |
|------|----------------|--------------|------------------------------|
| Sequential FT | 34.2 | 42.1 | 18.3 |
| EWC | 48.7 | 28.5 | 14.2 |
| LwF | 51.3 | 25.2 | 13.8 |
| Experience Replay | 55.8 | 20.1 | 11.5 |
| DPO (as CL) | 58.2 | 16.4 | 9.7 |
| **$\varphi$-DPO** | **63.1** | **12.3** | **4.2** |

### MLLM-CL Benchmark

| 方法 | Domain Avg ↑ | Ability Avg ↑ | Backward Transfer ↑ | Worst-group Acc ↑ |
|------|-------------|--------------|---------------------|-------------------|
| Sequential FT | 41.5 | 38.2 | -15.3 | 22.1 |
| LwF | 52.1 | 49.8 | -8.7 | 35.4 |
| Experience Replay | 56.3 | 53.1 | -5.2 | 40.8 |
| DPO (as CL) | 59.7 | 56.8 | -3.1 | 45.2 |
| **$\varphi$-DPO** | **65.2** | **62.4** | **-1.4** | **55.6** |

### 消融实验
- **$\gamma$ 的影响**：$\gamma=0$（退化为标准 DPO）→ $\gamma=1$ → $\gamma=2$ → $\gamma=5$，公平性指标单调改善；$\gamma \geq 5$ 后趋于饱和
- **DPO vs SFT 作为 CL 范式**：DPO 的 forgetting 比 SFT + KD 低 4.1%，验证了 DPO 的隐式蒸馏效应
- **参考策略选择**：$\pi_{t-1}$ vs $\pi_0$（初始模型）：使用 $\pi_{t-1}$ 效果更好（forgetting 低 5.2%），因为它更好地保留了最近学到的知识
- **$\beta$ 敏感性**：$\beta \in [0.05, 0.2]$ 范围内表现稳定，$\beta = 0.1$ 最优

## 亮点
- **持续学习的新视角**：首次将 DPO 作为持续学习范式，证明 DPO 天然具有知识蒸馏效应，理论推导优雅
- **双重问题的统一解决**：一个框架同时处理遗忘和公平性，而非分别用两个方法拼凑
- **公平性的理论保证**：Lemma 3 提供了 $\gamma \to \infty$ 时梯度偏差趋于零的严格证明，而非仅凭经验
- **focal loss 思想的巧妙迁移**：将原本用于目标检测中类别不平衡的 focal loss 思想迁移到持续学习的组间不平衡问题，跨领域迁移自然合理
- **轻量级改动**：相比标准 DPO 仅增加了一个调制因子 $(1-p)^\gamma$，实现几乎零额外成本

## 局限性 / 可改进方向
1. **$\gamma$ 的自适应选择**：目前 $\gamma$ 是手动设定的超参数，理想情况下应根据各组的不平衡程度自适应调整
2. **偏好对的质量依赖**：rejected 回答由 LLM 生成 + 人工验证，可扩展性受限于标注成本
3. **长序列 CL 的验证不足**：目前最多测试 8 个阶段的持续学习，更长序列（如 50+ 阶段）下 $\pi_{t-1}$ 参考策略的累积偏差未被研究
4. **单一 $\gamma$ 适用所有组**：所有组共享同一个 $\gamma$，而实际中不同组可能需要不同程度的调制
5. **与参数高效微调的结合**：当前使用全量微调，与 LoRA 等 PEFT 方法结合时，DPO 的隐式蒸馏效果是否依然成立有待验证

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ DPO 作为 CL 范式 + focal 公平性调制，双重创新点均有理论支撑
- 实验充分度: ⭐⭐⭐⭐ 两个 CL 基准 + 消融完整，但持续学习步数有限
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，motivation 阐述充分
- 价值: ⭐⭐⭐⭐⭐ 开辟"DPO for CL"新方向，公平性视角独到

