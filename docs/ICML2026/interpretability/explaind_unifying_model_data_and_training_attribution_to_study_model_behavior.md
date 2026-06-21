---
title: >-
  [论文解读] ExPLAIND: Unifying Model, Data, and Training Attribution to Study Model Behavior
description: >-
  [ICML2026][可解释性][路径核] ExPLAIND 把"模型组件归因、数据归因、训练轨迹归因"这三条平时各做各的可解释性路线统一进一个理论框架：它把用 AdamW 训练的模型严格重写成核机器（精确路径核 EPK 的扩展），由此导出可加的、按参数/样本/训练步索引的影响分数，再沿任意维度累加就能在任意粒度上解释模型行为，并用它重新刻画了 Grokking 的学习阶段和 EuroLLM 预训练的两阶段动态。
tags:
  - "ICML2026"
  - "可解释性"
  - "路径核"
  - "归因"
  - "训练动力学"
  - "Grokking"
  - "影响分数"
---

# ExPLAIND: Unifying Model, Data, and Training Attribution to Study Model Behavior

**会议**: ICML2026  
**arXiv**: [2505.20076](https://arxiv.org/abs/2505.20076)  
**代码**: https://github.com/mainlp/explaind  
**领域**: 可解释性  
**关键词**: 路径核、归因、训练动力学、Grokking、影响分数

## 一句话总结
ExPLAIND 把"模型组件归因、数据归因、训练轨迹归因"这三条平时各做各的可解释性路线统一进一个理论框架：它把用 AdamW 训练的模型严格重写成核机器（精确路径核 EPK 的扩展），由此导出可加的、按参数/样本/训练步索引的影响分数，再沿任意维度累加就能在任意粒度上解释模型行为，并用它重新刻画了 Grokking 的学习阶段和 EuroLLM 预训练的两阶段动态。

## 研究背景与动机
**领域现状**：事后可解释性（post-hoc interpretability）大体沿三条主线归因模型行为——归因到**模型组件**（哪一层/哪个神经元）、归因到**训练数据**（哪个样本影响了预测）、或归因到**训练动力学**（训练过程怎么演化的）。同时每种方法又往往被绑死在"局部—全局"谱系上的某个粒度：要么给出细粒度的局部解释，要么给出粗粒度的全局刻画。

**现有痛点**：这三个视角和不同粒度几乎都是**孤立使用**的。只看组件的解释会忽略单个训练样本的影响、也看不到组件在优化中如何演化；以数据为中心的解释又看不清模型不同部分怎么内化这些样本；而单一粒度的分析会错过只在另一粒度才浮现的模式。这种割裂让我们对"数据、组件、训练动态如何**共同**塑造行为"始终缺乏统一视角，关键的交互被遗漏。已有探查训练动力学的工作（在不同 checkpoint 做 probing/circuit finding）也大多把每个训练步当独立事件，**缺乏把 checkpoint 之间联系起来的理论**。

**核心矛盾**：可解释性需要一个能同时跨"视角"（组件/数据/训练步）和跨"粒度"（局部↔全局）、且 checkpoint 之间有严格理论连接的统一框架，而现有方法要么没有理论基础、要么把维度割裂开。

**本文目标**：① 给出一个理论扎实、把组件/数据/训练轨迹整合、且支持任意粒度的统一框架；② 让它适配现代真实训练（AdamW、weight decay、动态学习率、mini-batch），而非只在理想梯度下降下成立；③ 用它去重新解释 Grokking 和 LLM 预训练这类涌现现象。

**切入角度**：作者从 **Exact Path Kernel（EPK，Bell et al. 2023）** 出发——EPK 把梯度下降训练的模型严格（而非近似）重写成一个核机器，核心是逐步比较训练样本梯度与测试样本梯度的点积。但原始 EPK 不覆盖一阶/二阶动量估计、weight decay、动态学习率、mini-batch 这些现实成分。

**核心 idea**：把 EPK 推广到 AdamW，得到对最终预测沿"数据×参数×训练步"的**精确可加分解**，再定义一个"影响张量"，沿不同轴累加即可在任意粒度上得到参数级/数据级/步级归因——用同一套分数把三个视角统一起来。ExPLAIND 即 *Exact Path-Level Attribution Integrating Network and Data*。

## 方法详解

### 整体框架
ExPLAIND 不是一个 pipeline，而是一个"分解—累加"的分析框架：先在理论上把 AdamW 训练得到的最终预测**精确分解**成无数个原子影响项（每一项绑定一个训练步 $s$、一个参数 $\theta^{(i)}$、一个训练样本 $x_k$、一个输出维 $j$），再把这些项收进一个多维"影响张量 $\Gamma$"，最后通过**选择保留哪些轴、对哪些轴求和**，就能在参数级/数据级/步级三个视角、以及从单参数到整层、从单步到整段训练的任意粒度上，读出对模型行为的归因。整套框架先在 CNN/Transformer/小 LLM 上验证"分解是精确的"，再用参数剪枝验证"分数确实有意义"，最后落到 Grokking 和 EuroLLM 两个案例研究。

这是纯机制/理论方法（核心是矩阵分解与累加），不适合画 pipeline 框架图，下面用公式把分解讲清。

### 关键设计

**1. 把 AdamW 训练的模型扩展成精确路径核：让分解适配现代优化器**

原始 EPK 只覆盖朴素梯度下降，无法表达 AdamW 里的动量、weight decay、动态学习率与 mini-batch，这让它在真实训练上失效。作者的 Theorem 3.1 把 EPK 推广到 AdamW：从 AdamW 的参数更新出发，用指示变量处理 mini-batch，把每一步的预测变化直接写成"由训练样本诱导的参数梯度"加上"解耦 weight decay 带来的额外项"。最终预测被精确分解为

$$f_{\theta_N}(x)=f_{\theta_0}(x)-\sum_{k=1}^{M}\sum_{s=0}^{N-1}\phi_s^{test}(x)\,\phi_s^{train}(x_k)-\sum_{s=0}^{N-1}\phi_s^{test}(x)\,\mathbf{r}_s$$

其中测试特征图 $\phi_s^{test}(x):=\int_0^1\nabla_\theta f_{\theta_s(t)}(x)\,dt$ 沿相邻两步参数之间积分（用 100 个积分步即可），训练特征图 $\phi_s^{train}(x_k)$ 累加了各步带学习率权重 $\alpha_{s,i}$、被二阶矩 $\sqrt{\hat v_{s+1}}+\epsilon$ 归一化后的样本梯度，$\mathbf{r}_s:=\alpha_s\lambda\theta_s$ 专门捕捉 weight decay 的效应。Corollary 3.2 给出带动量的梯度下降的对应版本，Corollary 3.3 进一步把分解推广到**中间层激活**和**输出的可微函数（如 loss）**，从而能分解的对象远不止最终 logits。

**2. 影响张量与按轴累加：用同一套可加分数支撑"任意视角×任意粒度"**

有了精确分解，作者把"训练步 $s$、参数 $\theta^{(i)}$、训练样本 $x_k$、对预测 $x$ 的输出维 $j$"的原子影响定义为

$$\psi(s,\theta^{(i)},x_k,x)_j:=\phi_s^{test}(x)_{j,i}\cdot\phi_s^{train}(x_k)_i$$

正则项的影响 $\psi^{reg}$ 同理。关键性质是这些分数**可加且求和恰好等于模型预测**（式 2）。把所有原子分数收进影响张量 $\Gamma(\mathcal S,\Theta,\mathcal X,\mathcal X_{pred})_{\mathcal J}$，再定义累加影响 $\Psi:=\mathrm{sum}(\Gamma)$——沿哪个轴求和、保留哪个轴，就决定了你得到哪个视角、什么粒度的解释：想看单个参数对某预测的影响就对训练集求和得 $\Psi(\theta^{(i)},x)$，想"放大到层级"就把某层全部参数的分数加起来得 $\Psi_s(\Theta_{L},x)$。为消除不同预测/输出间的符号差异，还定义了取绝对值的累加重要度 $\bar\Psi$；为研究正则相对影响，定义了 $\Psi$ 与 $\Psi^{reg}$ 之差 $D$；为比较两个预测的表示，用分数向量的余弦相似度 $Sim(x,x')$。正是这种"一套分数、按需累加"的设计，让组件/数据/训练步三个原本割裂的视角被统一进同一框架。

**3. 用参数剪枝验证分数有意义、用两种近似让框架能 scale 到 LLM**

分解精确不等于分数"有用"，所以作者做了两重验证与提效。**验证**：按核重要度 $\Psi_S(\theta)$ 给 CNN 参数排序，只保留 TOP-$cD$ 个参数、其余置零（不剪输出层），在 70%~99% 稀疏度上与经典的 Li et al. (2017) 幅值剪枝（每步剪完还要重训）相当，且在所有稀疏度上 KL 散度都更低——说明这是无需重训的、能更忠实复现原模型的分数，证明影响分数确实量化了参数对预测的贡献（作者强调这只为验证分数有效，并非把 ExPLAIND 当 SOTA 剪枝方法）。**提效**：朴素实现要存全部分数，内存复杂度高达 $\mathcal O(NDMO)$（$N$ 步、$D$ 参数、$M$ 样本、$O$ 输出）。两招破局——① 若只关心组件沿训练轨迹的影响，可在算点积前先把训练特征图里的梯度沿数据累加（early accumulation），降复杂度；② 按一批留出样本上的绝对 loss 变化给训练步排序，只取 TOP-$X$ 步做子采样，MNIST 上稀疏度约 60% 起重建预测仍很准，即步数几乎砍半精度不掉。这两招正是后面 scale 到 EuroLLM 的前提。

### 损失函数 / 训练策略
ExPLAIND 是分析框架，不引入新训练目标。它分析的训练用标准 AdamW（含 weight decay、动态学习率、mini-batch）；测试特征图统一用 100 个积分步。EPK 表示的精确性在 ResNet9/CIFAR-2、Transformer/MOD-113、CNN/MNIST 上验证：100 积分步时 EPK 复现 100% 的分类决策、KL 散度近 0。

## 实验关键数据

### 主实验：EPK 表示的精确性
ExPLAIND 的根基是"分解必须精确"。作者在三种模型/任务上验证 EPK 重建与原模型预测的一致性：100 积分步时全部完美复现。

| 模型 / 数据 | 积分步 | EPK Acc. | KL 散度 |
|--------|------|------|------|
| ResNet9 / CIFAR-2 | 100 | 1.0 | 0.0 |
| Transformer / MOD-113 | 100 | 1.0 | 0.0 |
| CNN / MNIST | 100 | 1.0 | 0.0 |
| Transformer / MOD-113 | 10 | 0.748 | 0.885 |

（积分步只用 10 时会有偏差，故全文统一用 100 步。）

### 案例研究关键发现

| 案例 | ExPLAIND 揭示的结构 | 意义 |
|------|------|------|
| Grokking（模 113 加法 Transformer） | 三相：decoder 主导记忆 → 中间层交替形成 circuit → 外层（嵌入+decoder）在更高正则影响下对齐到表示流水线 | 细化了 Nanda 等的三阶段说，指出末期是外层"对齐复用"而非新建 |
| 因果验证：层替换 | 只把随机初始化模型的 Attention/Linear-1 换成 grokked 版，200 步内立即泛化、跳过记忆阶段 | 证明外层围绕中间表示流水线的对齐是因果性的 |
| EuroLLM-1.7B 预训练 | 两相：先外层 MLP（先靠输入侧再靠输出侧）驱动学习，约 60K 步相变后中间/低层与注意力层相对影响上升 | 首次把 LLM 预训练动态做参数级统一归因 |

### 关键发现
- **Grokking 不是"凭空生成新机制"，而是"对齐并复用已有表示"**：ExPLAIND 把记忆/circuit 形成/cleanup 三相用影响分数刻画出来，并通过"把 grokked 外层换进随机模型即刻泛化"的消融给出因果证据——末期外层围绕已形成的表示流水线对齐，正则影响则抑制低效的记忆解。数据视角还发现核里涌现出与模运算等价类对齐的循环几何，且频率从高到低持续精炼。
- **可 scale 到真实 LLM**：借助早累加 + 步子采样，无需重训、仅用 37 个现成 checkpoint，单张 H100 约 15 分钟即可分解一批的 loss 轨迹，重建误差均值仅 $4.46\times10^{-8}$（loss 变化均值约 $-6.05\times10^{-4}$）。
- **训练归因随时间剧烈变化**：无论 Grokking 还是 EuroLLM，数据/组件归因在不同训练阶段差异巨大、关键模式只在特定阶段涌现——这提示未来可解释性方法应主动去"暴露这些关键阶段"，而非只看终态。

## 亮点与洞察
- **"把训练好的模型严格写成核机器"是最硬核的一招**：EPK→AdamW 的扩展让"最终预测 = 一堆可加影响项之和"成立且精确（不是近似），这给数据/组件/训练步归因提供了共同的、求和即还原的理论底座，本身就有独立价值。
- **"影响张量 + 按轴累加"是极优雅的统一接口**：同一套原子分数，保留/求和不同轴就切换视角与粒度，把三条割裂的可解释性路线收进一个对象——这种"一次分解、任意切片"的设计思想可迁移到其他需要多视角归因的分析任务。
- **用剪枝来"验证分数而非刷 SOTA"很克制也很有说服力**：训练-free 剪枝在高稀疏度下 KL 更低，恰好证明影响分数捕捉到了参数的真实贡献。
- **对 Grokking 的"外层对齐复用"解释 + 层替换因果实验**，给一个被反复研究的现象提供了新的、可证伪的视角。

## 局限与展望
- **只做参数级、非因果**：作者明确 ExPLAIND 不提供因果解释；在 CIFAR/MNIST 上做因果数据归因时，朴素导出的数据影响分数仅与 TracIn 相当、略逊于 TRAK，远不如它在参数归因上的表现。
- **算力开销大**：即便用早累加与步子采样优化，整体开销仍可观，全量分数内存是 $\mathcal O(NDMO)$；大模型还受"无法存每步 checkpoint"限制，只能靠现成 checkpoint 当子采样。
- **案例规模有限**：Grokking 是小 Transformer、LLM 案例是 1.7B 的 EuroLLM 且只覆盖第一训练阶段；模 Transformer 上的结论能否推广到更大模型、更复杂任务仍待验证。
- **改进方向**：探索能否识别激活级的机制性 circuit、研究学习近似累加分数以进一步提效、以及与更多统一/非统一归因方法做更充分的对比。

## 相关工作与启发
- **vs EPK（Bell et al. 2023）**：EPK 把梯度下降模型严格写成核机器并研究数据归因，但不覆盖动量/weight decay/动态学习率/mini-batch、也只到数据视角；ExPLAIND 把它扩到 AdamW 并泛化到参数与训练步三维。
- **vs LCA / POLCA（损失变化分配）**：思路相近（分解 loss），但它们是近似解、且缺少把训练数据连到预测的理论联系；ExPLAIND 的分解精确且把数据—参数—步统一连接。
- **vs TracIn / TRAK（数据归因）**：TracIn 也用跨步梯度点积，但缺乏到模型预测的理论连接；TRAK 在因果数据归因上仍强于 ExPLAIND 的朴素数据分数——这是 ExPLAIND 当前的相对弱项。
- **vs DualXDA（Yolcu et al. 2025）**：沿"数据+输入特征"统一；ExPLAIND 改为统一"参数+训练步+数据"，但不提供输入特征视角，二者互补。
- **vs probing / circuit finding 类训练动态分析**：它们在各 checkpoint 独立做、缺乏跨步理论；ExPLAIND 用路径核把 checkpoint 在理论上串起来。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把 EPK 扩到 AdamW 并统一三视角×多粒度，理论与视角都很原创。
- 实验充分度: ⭐⭐⭐⭐ EPK 精确性 + 剪枝验证 + 两个案例研究层层递进，但因果数据归因偏弱、规模有限。
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨、案例叙述清楚；公式密度高，对读者门槛不低。
- 价值: ⭐⭐⭐⭐⭐ 给可解释性提供了统一且理论扎实的归因工具箱，并能 scale 到真实 LLM。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Hidden Breakthroughs in Language Model Training](../../ICLR2026/interpretability/hidden_breakthroughs_in_language_model_training.md)
- [\[ICLR 2026\] Evolution of Concepts in Language Model Pre-Training](../../ICLR2026/interpretability/evolution_of_concepts_in_language_model_pre-training.md)
- [\[ICML 2026\] OmniSapiens: A Foundation Model for Social Behavior Processing via Heterogeneity-Aware Relative Policy Optimization](omnisapiens_a_foundation_model_for_social_behavior_processing_via_heterogeneity-.md)
- [\[NeurIPS 2025\] Model-Behavior Alignment under Flexible Evaluation: When the Best-Fitting Model Isn't the Right One](../../NeurIPS2025/interpretability/model-behavior_alignment_under_flexible_evaluation_when_the_best-fitting_model_i.md)
- [\[ICML 2026\] Discovering Implicit Large Language Model Alignment Objectives](discovering_implicit_large_language_model_alignment_objectives.md)

</div>

<!-- RELATED:END -->
