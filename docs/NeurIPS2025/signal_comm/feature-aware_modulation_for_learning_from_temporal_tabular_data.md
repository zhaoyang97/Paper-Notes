---
title: >-
  [论文解读] Feature-aware Modulation for Learning from Temporal Tabular Data
description: >-
  [NeurIPS 2025][信号/通信][temporal shift] 论文认为时间表格学习真正难的不是“再加一个时间 embedding”这么简单，而是很多特征的语义会随时间漂移，因此提出 feature-aware modulation，通过时间上下文动态生成每个特征的偏移、缩放与非线性形状参数，把跨时间的语义重新对齐，最终在 TabReD 上让深度模型第一次在平均排名上稳定压过 GBDT。
tags:
  - "NeurIPS 2025"
  - "信号/通信"
  - "temporal shift"
  - "feature modulation"
  - "概念漂移"
  - "Yeo-Johnson"
  - "表格深度学习"
---

# Feature-aware Modulation for Learning from Temporal Tabular Data

**会议**: NeurIPS 2025  
**arXiv**: [2512.03678](https://arxiv.org/abs/2512.03678)  
**代码**: [https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation)  
**领域**: 时间表格学习 / 信号与通信 / 时序分布漂移  
**关键词**: temporal shift, feature modulation, 概念漂移, Yeo-Johnson, 表格深度学习

## 一句话总结

论文认为时间表格学习真正难的不是“再加一个时间 embedding”这么简单，而是很多特征的语义会随时间漂移，因此提出 feature-aware modulation，通过时间上下文动态生成每个特征的偏移、缩放与非线性形状参数，把跨时间的语义重新对齐，最终在 TabReD 上让深度模型第一次在平均排名上稳定压过 GBDT。

## 研究背景与动机

表格学习长期是 GBDT 的天下。

哪怕近年来 FT-Transformer、TabR、TabM、ModernNCA 这类深度模型进步很快，一到真实业务场景里的时间分布漂移，树模型还是常常更稳。

原因在于大多数表格学习方法默认数据满足 i.i.d. 假设。

可在现实里，时间会改变特征和标签的关系。

收入会受通胀影响。

房屋坐标本身不变，但“黄金地段”的含义会随城市发展变化。

用户行为、政策环境、医疗流程、金融风险偏好都会让相同数值在不同年份表达不同意义。

作者把这种现象总结成“特征语义演化”。

更具体地说，特征既有 objective semantics，也有 subjective semantics。

客观语义是数值本身的意义，比如经纬度、绝对工资值。

主观语义则是相对于分布上下文的意义，比如“高收入”“热门地段”“异常高风险”。

真正引发概念漂移的，往往是后者。

这也解释了为什么简单拼接时间 embedding 常常不够。

时间 embedding 是把时间信息直接塞给模型，希望模型自己学会解释变化。

但如果输入特征的语义本身没有先被校正，那么模型很可能把时间模态和输入模态纠缠在一起，导致过拟合短期模式、泛化到未来时失效。

因此作者提出一个更底层的问题。

是否可以不直接让模型记住每个时刻的模式，而是先把不同时间段的特征“翻译”到更一致的语义空间里。

如果可以，那么后续 backbone 就能在一个更稳定的表征空间里学习决策边界。

这就是 feature-aware modulation 的出发点。

## 方法详解

### 整体框架

输入是一个带时间戳的表格样本 $(\mathbf{x}, t)$。

模型首先用时间编码器得到时间嵌入 $\psi(t)$。

然后，一个轻量级 modulator 读取 $\psi(t)$，为每个特征维度产生三组参数：

$\gamma$，控制缩放。

$\beta$，控制平移偏移。

$\lambda$，控制非线性形状变化。

接着每个特征 $x_i$ 经过 Yeo-Johnson 变换与仿射调制，形成新特征

$\tilde{x}_i = \gamma_i(\psi(t)) \cdot \text{YJ}(x_i; \lambda_i(\psi(t))) + \beta_i(\psi(t))$。

这些调制后的特征再送入任意 backbone，例如 MLP 或 TabM。

调制不仅能放在输入层，也可以作用在中间表示层和输出 logits 层。

整套设计的要点是：时间不再作为一个并列输入特征和原始特征硬拼，而是作为“解释特征该如何被理解”的条件信号。

### 关键设计

1. **从“时间建模”转向“语义对齐”**

    - 功能：把时间漂移问题转写成表示空间中的语义校准问题。
    - 核心思路：作者认为 temporal tabular shift 本质上并不只是 covariate shift，也不只是 label shift，而是大量 subjectively interpreted feature 在时间上发生概念漂移。
    - 设计动机：如果直接让 backbone 学“某个年份该怎么决策”，模型会记住局部时间模式；如果先把特征语义对齐，再让 backbone 学统一边界，泛化会更稳。

2. **三参数特征调制：均值、尺度、偏度**

    - 功能：用最少的自由度刻画最关键的时间分布变化。
    - 核心思路：作者观察到时间漂移常体现在三类统计变化上：mean shift、std shift 和 skewness shift。于是分别用 $\beta$、$\gamma$、$\lambda$ 去对应这三种变化。
    - 设计动机：这不是无约束大超网络，而是有强归纳偏置的轻量调制器，因此更不容易过拟合。

3. **Yeo-Johnson 变换替代单纯 FiLM**

    - 功能：允许对特征分布做可微的非线性重塑。
    - 核心思路：FiLM 只能做缩放和平移，最多对均值和方差起作用。本文加入 Yeo-Johnson 变换后，可以动态改变分布形状，尤其适合偏斜、重尾或随时间变形的特征。
    - 设计动机：很多表格特征不是高斯型，时间漂移也不只表现为线性平移，因此必须有形状层面的调制能力。

4. **多层调制与输入层优先**

    - 功能：在不同表征层面做时间适配。
    - 核心思路：调制可作用在原始输入、中间表示和输出 logits。实验证明三层全开最好，但输入层单独调制已能拿到大部分收益。
    - 设计动机：输入层拥有最完整、最未失真的原始信息，是语义对齐最自然的位置；深层调制则用于补充更抽象的时间适配。

5. **和 backbone 解耦的轻量实现**

    - 功能：尽量不破坏现有表格模型结构。
    - 核心思路：modulator 只生成 $3m$ 个参数，$m$ 为特征维数，而不是像 hypernetwork 那样生成整网权重。
    - 设计动机：这样能把该方法较无痛地插进 MLP、TabM 等模型中，成本远低于完整动态网络。

### 损失函数 / 训练策略

分类任务使用交叉熵，回归任务用 MSE。

优化器为 AdamW，并采用早停。

超参搜索使用 Optuna，100 次试验，每个配置用 15 个随机种子取平均。

时间嵌入维度固定为 128，包含周期成分与趋势成分。

从训练策略上看，本文并没有在优化目标上搞复杂 tricks，真正的创新集中在表示层之前的调制机制。

这也是论文一个很有说服力的地方。

它没有依赖特殊 loss 才成立，而是在普通训练协议下就能稳定获益。

## 实验关键数据

### 主实验

实验基准是 TabReD，覆盖 8 个具有时间漂移的真实世界表格数据集。

评价使用分类任务上的 AUC 和回归任务上的 RMSE，并最终汇总为平均排名。

论文最重要的结果是，加入 temporal modulation 后，TabM 的平均排名达到 3.500，优于 CatBoost 的 4.375。

这意味着深度方法第一次在这个时间漂移设置下系统性超过 GBDT。

| 方法 | 类型 | 代表结果 | 平均排名 ↓ | 结论 |
|------|------|---------|------------|------|
| CatBoost | 静态 GBDT | HI 0.9639, CT 0.4792 | 4.375 | 传统强基线，整体很稳 |
| TabM | 静态深度 | HI 0.9640, CT 0.4813 | 7.250 | 深度模型强，但时间漂移下仍落后 |
| MLP + 时间嵌入 | 自适应深度 | HI 0.9471, CT 0.4801 | 14.375 | 直接拼接时间，提升有限 |
| TabM + 时间嵌入 | 自适应深度 | HI 0.9629, CT 0.4791 | 5.125 | 比静态深度更好，但仍未最好 |
| **TabM + Temporal Modulation** | **本文方法** | **HI 0.9641, CT 0.4773** | **3.500** | 首次在平均排名上超越 GBDT |
| **MLP + Temporal Modulation** | **本文方法** | **HI 0.9593, CT 0.4782** | **11.000** | 轻量 backbone 也明显受益 |

这里有两个值得特别注意的点。

一是 modulation 不只是给强模型锦上添花。

即便换成最普通的 MLP，也能比很多更复杂的静态或时间嵌入方法更强。

二是时间嵌入本身不是没用，但效果明显弱于 modulation。

这支持作者的论断：问题不在于“要不要看时间”，而在于“时间应不应该直接和原始特征混在一起”。

### 消融实验

作者系统比较了调制位置。

结论非常清晰：全层调制最好，但输入层单独调制已经吃到绝大多数收益。

| 输入层 | 中间层 | 输出层 | 平均提升 | 占完整提升比例 | 平均排名 ↓ |
|-------|-------|-------|---------|----------------|------------|
| ✗ | ✗ | ✗ | 0.00% | 0% | 5.500 |
| ✓ | ✗ | ✗ | 1.83% | 87.4% | 3.250 |
| ✓ | ✗ | ✓ | 1.54% | 73.6% | 3.625 |
| ✓ | ✓ | ✗ | 1.62% | 77.2% | 3.750 |
| ✓ | ✓ | ✓ | **2.09%** | **100%** | **2.500** |

作者进一步指出，如果去掉输入层调制，完整收益只剩 56.8%。

这个现象说明了一个朴素但重要的事实。

语义对齐越早做越划算。

一旦原始特征在网络早期就被错误地解释，后面再补救会越来越难。

| 观察点 | 论文现象 | 启示 |
|-------|---------|------|
| 输入层单调制 | 取得 87.4% 完整收益 | 最低成本、最易迁移的部署方式 |
| 全层调制 | 平均提升 2.09% | 多层调制具有互补性 |
| 无输入层调制 | 完整收益只剩 56.8% | 早期表示最关键 |
| 时间 embedding 维度增大 | 传统方法会退化 | 说明模态纠缠与缩放问题真实存在 |

### 关键发现

- 论文第一次比较清楚地把 temporal tabular learning 的困难归结为“特征语义漂移”，而不是泛泛的 non-i.i.d.。
- 输入层调制贡献了绝大多数收益，说明把语义先对齐，再让 backbone 学判别边界，是比直接时间拼接更自然的路线。
- modulation 对 MLP 和 TabM 都有效，说明它不是某个 backbone 的特殊技巧，而更像通用前端。
- pilot study 显示，经过调制后，不同时间段的特征分布虽然仍非完全 i.i.d.，但已经足够对齐，能让模型学到一致决策边界。

## 亮点与洞察

- **“客观语义 / 主观语义”这个分析框架非常好**。它把很多业务里难以言明的 temporal drift 变成了可解释的学习目标。
- **调制统计量的选择很克制**。只处理 mean、std、skewness 三类关键变化，却取得了很强效果，这说明好的结构先验比暴力增大模型更重要。
- **Yeo-Johnson 的引入恰到好处**。相比只做 FiLM，本文真正允许分布形状随着时间变化，这点对表格数据特别关键。
- **输入层即可获得 87.4% 收益**。这让方法非常具有工程吸引力，因为部署成本低。
- **论文本质上是在做“时间条件下的可解释特征工程自动化”**。这是一个很值得推广的视角。

## 局限与展望

首先，full-stage modulation 与 PLR embedding 不兼容。

这限制了它和一些最强表格 backbone 的深度结合方式。

其次，实验主要围绕 TabReD 展开。

虽然 TabReD 已经是时间表格学习的重要基准，但跨更多行业和超大规模表格场景的验证仍然必要。

第三，本文主要从 mean、std、skewness 三个统计量出发。

这对大多数漂移足够，但若时间语义变化更复杂，例如多峰分布切换、结构性稀疏模式重排，可能需要更强的调制族。

第四，时间嵌入仍然是预定义结构。

对于不具明显周期的事件驱动场景，仅靠这些先验编码是否足够，还需要继续验证。

第五，作者虽然给出 pilot study，可视化很直观，但对学习到的 $\gamma$、$\beta$、$\lambda$ 本身还缺少深入解释。

未来可尝试几个方向。

一个是设计与 PLR 兼容的 modulation 版本。

一个是把 modulation 扩展到表格-文本、多模态表格或带图结构的时序场景。

一个是让调制参数在时间之外再条件于群体、区域或环境变量，形成更细粒度的 context-aware tabular learner。

## 相关工作与启发

- **vs 直接时间 embedding**：后者把时间作为额外输入让模型自己消化，本文则让时间先改写特征解释方式，二者在 inductive bias 上完全不同。
- **vs FiLM / Hypernetwork**：本文比 FiLM 多了非线性分布形状调制，比 full hypernetwork 又轻量得多，是一个中间但更实用的方案。
- **vs GBDT**：树模型之所以在时间漂移下常更稳，某种程度上就是因为它对特征阈值变化更鲁棒；本文则试图在深度模型里手工补回这种稳定性来源。
- **对我自己的启发**：做时间表格模型时，不要急着堆复杂时序结构，先问一遍“这些特征今天和去年是不是还代表同一件事”。
- **迁移思路**：金融风控、医疗纵向随访、招聘画像、保险定价、房地产估值等场景，都很适合把时间条件下的特征语义对齐作为第一步。

## 评分

- 新颖性: ⭐⭐⭐⭐☆ 从语义漂移角度重写 temporal tabular learning，并用 feature modulation 落地，角度很新。
- 实验充分度: ⭐⭐⭐⭐☆ 主结果、pilot、消融和扩展分析都比较完整，唯一不足是 benchmark 仍偏单一。
- 写作质量: ⭐⭐⭐⭐☆ 动机很清楚，例子也很直观，方法不难理解。
- 价值: ⭐⭐⭐⭐⭐ 方法轻量、可迁移、效果扎实，对时间表格学习非常有现实意义。
---
title: >-
  [论文解读] Feature-aware Modulation for Learning from Temporal Tabular Data
description: >-
  [NeurIPS 2025][时间分布漂移] 本文针对时间表格数据中的分布漂移问题，提出特征感知时间调制机制，通过基于时间上下文的可学习变换来动态调整特征的偏移（$\beta$）、缩放（$\gamma$）和偏度（$\lambda$），实现跨时间的特征语义对齐，在 TabReD 基准测试上首次让深度学习方法系统性超越 GBDT。
tags:
  - NeurIPS 2025
  - 时间分布漂移
  - 特征调制
  - Yeo-Johnson变换
  - 概念漂移
  - 表格数据
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Task Aware Modulation Using Representation Learning for Upscaling of Terrestrial Carbon Fluxes](../../AAAI2026/signal_comm/task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)
- [\[NeurIPS 2025\] Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning](contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)
- [\[ICCV 2025\] Boosting Multimodal Learning via Disentangled Gradient Learning](../../ICCV2025/signal_comm/boosting_multimodal_learning_via_disentangled_gradient_learning.md)
- [\[CVPR 2025\] Neural Video Compression with Context Modulation](../../CVPR2025/signal_comm/neural_video_compression_with_context_modulation.md)
- [\[ICML 2026\] Joint Model and Data Sparsification via the Marginal Likelihood](../../ICML2026/signal_comm/joint_model_and_data_sparsification_via_the_marginal_likelihood.md)

</div>

<!-- RELATED:END -->
