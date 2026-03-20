# Beyond Confidence: The Rhythms of Reasoning in Generative Models

**会议**: ICLR 2026  
**arXiv**: [2602.10816](https://arxiv.org/abs/2602.10816)  
**代码**: 无  
**领域**: LLM分析 / 预测稳定性  
**关键词**: Token Constraint Bound, 预测鲁棒性, 隐状态扰动, 输出嵌入几何, prompt工程  

## 一句话总结
提出 Token Constraint Bound ($\delta_{\text{TCB}}$) 指标，通过量化 LLM 隐状态在多大扰动范围内能保持 next-token 预测不变，来度量预测的局部鲁棒性，揭示了传统 perplexity 无法捕捉的预测不稳定性。

## 研究背景与动机
1. **领域现状**：LLM 对输入上下文的微小变化极为敏感——格式微调可导致准确率波动 76%，示例顺序调整可使准确率从 54% 到 93%
2. **现有痛点**：
   - 准确率只给出聚合视图，无法评估单个预测的稳定性
   - Perplexity 混淆概率分布，忽略了内部状态几何结构
   - Softmax 归一化可导致高概率但不稳定的预测——高概率可能来自相对归一化而非鲁棒的内部状态
3. **核心矛盾**：一个高概率、高置信度的预测可能对应一个不稳定的内部状态平衡——现有指标无法区分"真正稳定的高置信"和"脆弱的高置信"
4. **本文要解决什么**：量化 LLM 在特定上下文下产生的内部状态 $\mathbf{h}$ 对小扰动的鲁棒性
5. **切入角度**：利用 Jacobian 矩阵分析 softmax 输出对隐状态的一阶敏感性
6. **核心 idea 一句话**：预测的鲁棒性 = 隐状态周围能保持输出分布不变的最大扰动半径，由输出嵌入的几何分散度决定

## 方法详解

### 整体框架
LLM 最后一层隐状态 $\mathbf{h} \in \mathbb{R}^d$ 经输出权重矩阵 $\mathbf{W} \in \mathbb{R}^{\mathcal{V} \times d}$ 和 softmax 映射为概率分布 $\mathbf{o}$。$\delta_{\text{TCB}}$ 量化的是：在 $\mathbf{h}$ 周围多大的扰动球内，$\mathbf{o}$ 的变化不超过容忍度 $\epsilon$。

### 关键设计

1. **Token Constraint Bound ($\delta_{\text{TCB}}$) 定义**:
   - 做什么：度量 LLM 预测对内部状态扰动的鲁棒性
   - 核心思路：利用一阶线性近似 $\Delta\mathbf{o} \approx \mathbf{J}_\mathbf{W}(\mathbf{h}) \Delta\mathbf{h}$，从 $\|\Delta\mathbf{o}\|_2 \leq \epsilon$ 推导出 $\|\Delta\mathbf{h}\|_2 \leq \epsilon / \|\mathbf{J}_\mathbf{W}(\mathbf{h})\|_F$，定义 $\delta_{\text{TCB}}(\mathbf{h}) = \epsilon / \|\mathbf{J}_\mathbf{W}(\mathbf{h})\|_F$
   - 设计动机：$\delta_{\text{TCB}}$ 越大说明模型的预测在更大范围的隐状态扰动下保持稳定

2. **与输出嵌入几何的精确联系**:
   - 做什么：推导 Jacobian 范数的解析表达式
   - 核心思路：证明 $\|\mathbf{J}_\mathbf{W}(\mathbf{h})\|_F^2 = \sum_{i=1}^{\mathcal{V}} o_i^2 \|\mathbf{w}_i - \boldsymbol{\mu}_\mathbf{w}(\mathbf{h})\|_2^2$，其中 $\boldsymbol{\mu}_\mathbf{w}(\mathbf{h}) = \sum_j o_j \mathbf{w}_j$ 是概率加权平均嵌入
   - 几何含义：敏感性由 token 嵌入相对于加权中心的分散度决定，且被 $o_i^2$ 加权——高概率 token 的嵌入位置影响最大

3. **两种预测体制的分析**:
   - **高置信体制**（$\mathcal{V}_{\text{eff}}$ 低）：$\boldsymbol{\mu}_\mathbf{w} \to \mathbf{w}_k$（主导 token），$\delta_{\text{TCB}} \to \infty$。此时 $\delta_{\text{TCB}}$ 与 top-2 logit margin 强正相关 ($r = 0.62$)
   - **不确定体制**（$\mathcal{V}_{\text{eff}}$ 高）：概率分散于多个 token，$\delta_{\text{TCB}}$ 与 $\sqrt{\mathcal{V}_{\text{eff}}}$ 正相关 ($r = 0.95$)。但关键洞察：即使 $\mathcal{V}_{\text{eff}}$ 高，若高概率 token 的嵌入几何上聚集，$\delta_{\text{TCB}}$ 仍可以很高

### 损失函数 / 训练策略
- $\delta_{\text{TCB}}$ 是分析指标，不涉及训练
- 计算只需前向传播获取 $\mathbf{h}$、$\mathbf{o}$ 和 $\mathbf{W}$，然后通过解析公式计算
- 设 $\epsilon = 1.0$ 作为归一化标准

## 实验关键数据

### 主实验 — 预测体制验证（LLaMA-3.1-8B）

| 数据集 | Corr($\delta_{\text{TCB}}, \mathcal{V}_{\text{eff}}$) | Corr($\delta_{\text{TCB}}, z_{top1} - z_{top2}$) |
|--------|------|------|
| Diverse Prompts (N=309) | **0.95** (强正相关) | -0.40 |
| Low-$\mathcal{V}_{\text{eff}}$ Targeted (N=360) | 0.08 (近零) | **0.62** (强正相关) |

### 消融实验 — 嵌入几何验证

| 嵌入操作 | 假设 $\delta_{\text{cluster}} > \delta_{\text{orig}} > \delta_{\text{disperse}}$ 成立比例 |
|---------|---|
| Low $\mathcal{V}_{\text{eff}}$ (< 20) | 95% |
| 整体 | 90% |

- 固定 $\mathbf{o}$ 不变，人为聚集/分散竞争 token 的嵌入 → $\delta_{\text{TCB}}$ 相应增大/减小
- **证实了几何结构独立于概率分布影响稳定性**

### 关键发现
- **$\delta_{\text{TCB}}$ 能区分 prompt 质量**：good prompt → 更高 $\delta_{\text{TCB}}$，即使准确率相同
- **识别 perplexity 遗漏的不稳定性**：文本生成中存在 perplexity 低但 $\delta_{\text{TCB}}$ 骤降的位置，可能是语义转折点或潜在错误
- **ICL 示例的效果在 $\delta_{\text{TCB}}$ 中可见**：有效的 few-shot 示例不仅提高准确率，还增加 $\delta_{\text{TCB}}$

## 亮点与洞察
- **概率高 ≠ 稳定**：这个核心洞察极有价值——softmax 归一化可能制造"虚假安全感"，$\delta_{\text{TCB}}$ 直接检测内部状态的真实稳定性
- **嵌入几何的主导角色**：即使概率分布相同，改变嵌入空间的几何结构就能改变预测稳定性——这对理解 LLM 的表示学习有启发
- **解析公式优雅**：$\|\mathbf{J}\|_F^2 = \sum o_i^2 \|\mathbf{w}_i - \boldsymbol{\mu}\|^2$ 将复杂的 Jacobian 范数归结为直觉清晰的加权分散度

## 局限性 / 可改进方向
- 基于一阶线性近似，对大扰动的预测可能不准确
- 仅在 LLaMA-3.1-8B 上验证，需要更多模型和规模的实验
- $\epsilon = 1.0$ 的选择缺乏理论依据
- 未探索如何将 $\delta_{\text{TCB}}$ 纳入训练目标以直接提升鲁棒性
- Frobenius 范数作为敏感性度量可能过于保守（vs 谱范数）

## 相关工作与启发
- **vs Perplexity**：PPL 度量序列似然，$\delta_{\text{TCB}}$ 度量局部预测鲁棒性——互补而非替代
- **vs 校准指标**：校准关注概率与正确性的一致性，$\delta_{\text{TCB}}$ 关注预测对扰动的稳定性——正交维度
- **vs 对抗鲁棒性**：对抗研究在输入空间找最坏扰动，$\delta_{\text{TCB}}$ 在隐状态空间量化安全边际

## 评分
- 新颖性: ⭐⭐⭐⭐ Jacobian 分析不新，但将其与输出嵌入几何联系并定义有实际意义的指标是新颖的
- 实验充分度: ⭐⭐⭐⭐ 理论验证+prompt分析+ICL分析+文本生成分析，但模型多样性不足
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，但行文略冗长
- 价值: ⭐⭐⭐⭐ 提供了一个新的 LLM 分析视角，对 prompt 工程和可靠性评估有实用价值
