# From Samples to Scenarios: A New Paradigm for Probabilistic Forecasting

**会议**: ICLR 2026
**arXiv**: [2509.19975](https://arxiv.org/abs/2509.19975)
**代码**: [GitHub](https://github.com/Fifthky/TimePrism)
**领域**: 时间序列
**关键词**: probabilistic forecasting, time series, scenario generation, discrete probability, linear model

## 一句话总结
提出 Probabilistic Scenarios 范式，用模型直接输出有限个 {场景, 概率} 对取代采样，并用仅含三层平行线性层的 TimePrism 在5个基准数据集上取得9/10 SOTA。

## 研究背景与动机
1. **领域现状**: 概率时间序列预测是不确定性决策的基础，主流方法分为参数分布模型、生成模型（扩散）和结构化概率模型（流/copula），均依赖采样来表示预测分布。
2. **现有痛点**: 采样范式存在三大固有缺陷——(i) **概率缺失**：生成的轨迹没有对应概率值；(ii) **覆盖不足**：有限样本难以捕获低概率高影响的尾部事件；(iii) **推理开销**：生成多样本的计算成本随样本数线性增长。
3. **核心矛盾**: 高质量概率预测需要大量样本来充分近似分布，但大量采样又导致计算不可承受，且采样本身不提供显式概率。
4. **本文要解决什么**: 设计一种不依赖采样的概率预测范式，单次前向传播即可输出完整的离散概率分布。
5. **切入角度**: 将学习目标从"近似连续概率空间"简化为"学习有限场景集上的概率分布"，类似 VQ-VAE 的思路但直接作用于输出轨迹空间。
6. **核心idea一句话**: 用一个简单线性模型直接生成 $N$ 个未来场景及其概率，彻底绕开采样。

## 方法详解
### 整体框架
Probabilistic Scenarios 范式定义模型为:
$$f(\mathbf{x}) = (\mathcal{Y}_{\text{pred}}, \mathbf{p})$$
其中 $\mathcal{Y}_{\text{pred}} = \{\mathbf{y}_n\}_{n=1}^N$ 是 $N$ 个预测场景，$\mathbf{p} = (p_1, \dots, p_N)$ 是对应概率向量，满足 $\sum p_n = 1$。

### 关键设计
1. **时间序列分解**: 将输入历史 $\mathbf{x} \in \mathbb{R}^{L \times D}$ 通过移动平均分解为趋势分量 $\mathbf{x}_{\text{trend}}$ 和季节分量 $\mathbf{x}_{\text{season}}$。
2. **趋势+季节线性层**: 趋势层生成 $M$ 个趋势预测，季节层生成 $K$ 个季节预测，通过组合得到 $N = M \times K$ 个场景：
   $$\mathcal{Y}_{\text{pred}} = \{\mathbf{y}_{t,m} + \mathbf{y}_{s,k} \mid m \in [M], k \in [K]\}$$
   这种组合设计使参数复杂度仅为 $\mathcal{O}(\sqrt{N})$，远优于直接生成 $N$ 个场景。
3. **概率层**: 第三个线性层以原始未分解历史为输入，输出 $N$ 维 logits 向量 $\boldsymbol{\pi}$，经 Softmax 转为概率向量。

### 损失函数 / 训练策略
总损失由两部分组成：
$$\mathcal{L}_{\text{Prism}} = \mathcal{L}_{\text{recon}} + \lambda \cdot \mathcal{L}_{\text{prob}}$$

- **场景重建损失** (WTA): 找到与真值最近的"胜者"场景 $n^* = \arg\min_n \|\mathbf{y}_{gt} - \mathbf{y}_n\|_2^2$，仅对该场景计算 MSE
- **概率损失**: 用交叉熵训练概率层将最高概率分配给胜者：
  $$\mathcal{L}_{\text{prob}} = -\log \frac{\exp(\pi_{n^*})}{\sum_j \exp(\pi_j)}$$
- 实际训练采用 relaxed WTA 以稳定训练，$\lambda=1$

## 实验关键数据
### 主实验
5个基准数据集 (Electricity, Exchange, Solar, Traffic, Wikipedia) 上的 Weighted CRPS：

| 模型 | Elec. | Exch. | Sol. | Traf. | Wiki. |
|------|-------|-------|------|-------|-------|
| TimeGrad | 0.232 | 0.845 | 0.241 | 0.162 | 0.517 |
| TACTiS-2 | 0.299 | 0.648 | 0.236 | 0.257 | 0.484 |
| TimeMCL | 0.370 | 1.12 | 0.290 | 0.262 | 0.640 |
| **TimePrism** | **0.133** | **0.468** | **0.085** | **0.111** | **0.506** |

Distortion 指标上 TimePrism 在全部5个数据集上均取得 SOTA。

### 消融实验
场景数 $N$ 影响实验 (Solar 数据集):

| N | CRPS | Distortion | FLOPs(相对) |
|---|------|------------|-------------|
| 1 | 0.199 | 0.266 | 1.0x |
| 16 | 0.137 | 0.307 | 4.2x |
| 256 | 0.093 | 0.158 | 19.9x |
| 625 | 0.085 | 0.101 | 34.8x |
| 1024 | 0.082 | 0.092 | 48.3x |

$N=625$ 时性能收益趋于饱和。

### 关键发现
- TimePrism 推理 FLOPs 恒定（$5.1 \times 10^5$），不随样本数增长，而 TimeGrad 100样本需 $1.9 \times 10^{10}$ FLOPs
- 可视化显示 TimePrism 能以高概率捕获常见高峰场景，同时以低概率识别罕见低峰场景，而采样模型无法区分两者
- 组合架构 ($N = M \times K$) 使参数增长在 $\mathcal{O}(\sqrt{N})$ 到 $\mathcal{O}(N)$ 之间

## 亮点与洞察
- **范式创新**: 从"采样近似连续分布"到"直接生成离散场景+概率"的根本性转变，概念简洁而有效
- **极简架构验证**: 仅用3个平行线性层（无非线性激活）即达 SOTA，证明范式本身的强大潜力
- **统一评估框架**: 提出 Weighted CRPS 和 Distortion 两个互补指标，并为两种范式分别给出公平可比的计算公式
- **效率优势**: 单次前向传播，推理成本比最强基线低1-5个数量级

## 局限性 / 可改进方向
- 线性模型对极高维度或无明显趋势/季节模式的序列可能不适用
- 固定输入/输出长度，缺乏变长序列的灵活性
- 多变量建模采用 weight-sharing 策略，跨变量关系建模较简单
- 最优场景数 $N$ 依赖数据复杂度，目前需手动设定
- WTA 损失可能导致部分场景头在训练早期被忽略（“赢家通吃”效应），relaxed WTA 只是部分缓解
- 未在 GIFT-Eval 等更大规模基准上验证
- 缺少对不同预测平 horizon 的敏感度分析

## 相关工作与启发
- 与 TimeMCL 对比：TimeMCL 也输出离散场景但不直接建模概率，CRPS 不如 SOTA；本文通过概率层统一场景保真度和概率匹配
- 与 VQ-VAE 的概念类比：将离散化直接应用于输出轨迹空间而非潜在空间
- 与 TACTiS-2 对比：TACTiS-2 能计算概率密度但仍需采样获取轨迹，本文直接输出离散场景
- 与 TimeGrad 对比：扩散模型需迭代采样，100样本的 FLOPs 是 TimePrism 的 $10^4$ 倍
- 未来可将此范式接入 Transformer、Diffusion 等强力骨干，解锁更强的多变量建模能力
- 自适应场景数机制也是有价值的未来方向

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 概率预测范式的根本性创新，从采样转向离散场景概率
- 实验充分度: ⭐⭐⭐⭐ 5个数据集+多基线+消融+可视化，但仅限时间序列领域
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，从问题到方案的逻辑链完整
- 价值: ⭐⭐⭐⭐⭐ 为概率预测开辟新方向，极简模型即达SOTA具有极强说服力
