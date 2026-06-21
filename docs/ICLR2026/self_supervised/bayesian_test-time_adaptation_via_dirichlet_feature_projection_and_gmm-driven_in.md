---
title: >-
  [论文解读] Bayesian Test-Time Adaptation via Dirichlet feature projection and GMM-Driven Inference for Motor Imagery EEG Decoding
description: >-
  [ICLR2026][自监督学习][测试时适应] BTTA-DG 把每条 EEG 试次的逐时刻预测序列压成一个 Dirichlet 参数向量，用历史试次拟合的 GMM 当似然、深度模型输出当先验，做一次无梯度的贝叶斯后验校准，在运动想象脑机接口的跨被试/跨 session 迁移上达到 SOTA 且实时（15.7 ms/试次）。
tags:
  - "ICLR2026"
  - "自监督学习"
  - "测试时适应"
  - "EEG 运动想象"
  - "Dirichlet 分布"
  - "高斯混合"
  - "贝叶斯校准"
---

# Bayesian Test-Time Adaptation via Dirichlet feature projection and GMM-Driven Inference for Motor Imagery EEG Decoding

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=VDg6Pv4S3v](https://openreview.net/forum?id=VDg6Pv4S3v)  
**代码**: 待确认  
**领域**: 测试时适应 / 自监督 / EEG 脑机接口  
**关键词**: 测试时适应, EEG 运动想象, Dirichlet 分布, 高斯混合, 贝叶斯校准

## 一句话总结
BTTA-DG 把每条 EEG 试次的逐时刻预测序列压成一个 Dirichlet 参数向量，用历史试次拟合的 GMM 当似然、深度模型输出当先验，做一次无梯度的贝叶斯后验校准，在运动想象脑机接口的跨被试/跨 session 迁移上达到 SOTA 且实时（15.7 ms/试次）。

## 研究背景与动机
**领域现状**：基于 EEG 的运动想象（Motor Imagery, MI）脑机接口靠解码感觉运动节律来控制外设。近年大规模 EEG 预训练模型把表征学习推得很高，但真要部署时会撞上 EEG 信号的非平稳性——同一套模型换个被试、换个 session，分布就漂了（cross-subject / cross-session shift），不微调几乎没法用。

**现有痛点**：测试时适应（TTA）本来是个对症的方向——推理阶段用无标签的在线数据现场调模型。但已有的 EEG-TTA 分两派，各有硬伤：① **梯度派**（Tent 的熵最小化、伪标签、一致性正则、T-TIME、OTTA 这类）要反向传播更新参数，计算贵，而且在 EEG 在线场景里 batch size 常常是 1，单条噪声试次就会产生误导性梯度，把预训练结构覆盖掉，引发灾难性遗忘；② **数据对齐派**（BN 统计重算、Euclidean Alignment 这类）不更新参数所以快，但只是浅层统计对齐，抓不住深层"时序预测嵌入"在新域里到底怎么变形了。

**核心矛盾**：既要计算高效、不毁预训练权重（避免灾难性遗忘），又要能建模深层的分布变化、还得有理论依据——梯度派和对齐派各占一头，没人同时占全。

**切入角度**：作者的关键观察是——**域漂移更可靠地体现在"模型逐时刻预测序列的分布"里，而不是任何单次预测里**。一条 EEG 试次经过编码器会输出一串随时间变化的类别概率向量 $X=[x_1,\dots,x_T]$，与其盯着平均后的那个点估计，不如把整条概率轨迹的"集中程度"建模出来。

**核心 idea**：用 Dirichlet 分布（"分布的分布"）把这串时序概率轨迹投影成一个低维参数 $\alpha$，再用 GMM 对历史 $\alpha$ 建密度、用贝叶斯推断把它和模型先验融合——整个适应过程**无梯度**，只校准输出不动权重。

## 方法详解

### 整体框架
BTTA-DG 处理的是在线、单试次到达的跨被试 TTA：源域是若干被试的有标签 EEG，目标域是某一个被试逐条到来的无标签试次，目标是不碰源数据、不要目标标签就把预训练模型 $f_\theta$ 适配过去。整条流水线是"轻量主干提特征 → 把时序预测轨迹投影成 Dirichlet 参数 → 用历史参数的 GMM 做贝叶斯后验校准"。

具体地：先用一个轻量的 **SincAdaptNet** 当主干，它的可学习 Sinc 带通滤波器把 MI 相关的 mu/beta/gamma 频段抽出来，编码器输出每个试次的逐时刻类别概率轨迹 $X\in\mathbb{R}^{|L|\times T}$，时间平均后得到模型先验 $f_{cls}(X)=\frac1T\sum_j x_j$。然后 **Dirichlet 特征投影** 用极大似然把整条轨迹 $X$ 压成一个 $\alpha\in\mathbb{R}^{|L|}_+$，每个分量 $\alpha_i$ 是对类 $i$ 的"证据浓度"、总尺度 $\alpha_0$ 反映整体不确定性。最后 **GMM-驱动的贝叶斯推断** 把历史高置信试次的 $\alpha$ 按预测类别存进 memory bank、每类拟合一个 GMM 当似然 $p_{GMM}(\alpha\mid y)$，和模型先验 $p_\theta(y)$ 相乘归一化得到校准后验，取 argmax 作为最终预测，再把当前 $\alpha$ 按置信度/熵阈值回写进 memory bank。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["在线 EEG 试次<br/>(batch=1)"] --> B["SincAdaptNet 主干<br/>可学习 Sinc 带通滤波"]
    B --> C["逐时刻预测轨迹 X<br/>+ 时间平均得先验 pθ(y)"]
    C --> D["Dirichlet 特征投影<br/>MLE 把 X 压成 α"]
    D --> E["GMM-驱动贝叶斯推断<br/>历史 α 拟合 GMM 当似然"]
    C -->|先验 pθ(y)| E
    E -->|校准后验 argmax| F["最终预测 ŷ_cal"]
    E -->|高置信/低熵才回写| G["memory bank 更新<br/>+ EM 重拟合 GMM"]
    G -.历史 α.-> E
```

### 关键设计

**1. SincAdaptNet 主干：用可解释带通滤波替自由卷积核，给后续投影喂干净特征**

EEG 单试次在线适配里 batch size 经常是 1，普通卷积主干既依赖 batch 统计、学到的核又是黑箱，频段乱混。SincAdaptNet 用 `Spat-Conv → Sinc-Conv → IncCh-Conv → Cls-Conv` 的结构，并在时序滤波和通道扩展后插 LayerNorm（而非 BatchNorm），从根上摆脱单样本下 batch 统计不稳的问题。核心是 Sinc-Conv：受 SincNet 启发，它不学自由形态的卷积核，而是只学每个滤波器的低截止频率 $f_{low}$ 和带宽 $f_{band}$（于是 $f_{high}=f_{low}+f_{band}$），据此生成一个加窗 sinc 带通核。这样用极少参数就把 MI 相关的 mu(8–13 Hz)、beta(13–30 Hz)、gamma(>30 Hz) 节律分离出来，频谱含义清楚（实验里 Table 6 显示约 25% 滤波器落在 μ 带、31% 落 β 带、36% 落 γ 带，<8% 落在 MI 节律之外）。受 SwAV 启发，编码器输出经 softmax 映射到归一化概率空间，得到每个时刻的瞬时类别概率向量 $x_j\in\Delta^{|L|-1}$，整条轨迹既是后续 Dirichlet 投影的输入，时间平均又给出贝叶斯推断要用的先验。

**2. Dirichlet 特征投影：把一整条时序预测轨迹压成一个可解释的浓度向量**

浅层对齐抓不住"深层预测分布在新域里怎么变形"，而单次预测又太吵。作者的解法是把 Dirichlet 分布——"类别分布上的分布"——引进 EEG-TTA：假设一条试次的 $T$ 个时刻概率向量 $x_j$ 是 $\mathrm{Dir}(\alpha)$ 的 i.i.d. 采样，于是用一个投影 $\mathcal{P}:\mathbb{R}^{|L|\times T}\to\mathbb{R}^{|L|}_+$，通过极大似然把整条轨迹估成一个 $\hat\alpha_{MLE}=\arg\max_\alpha\sum_{j=1}^T\log D(x_j;\alpha)$，其中

$$D(x_j;\alpha)=\frac{\Gamma(\alpha_0)}{\prod_{i=1}^{|L|}\Gamma(\alpha_i)}\prod_{i=1}^{|L|}x_{ij}^{\alpha_i-1}.$$

$\hat\alpha_{MLE}$ 由 Minka 的定点迭代高效求解：$\alpha_i^{new}=\psi^{-1}\!\big(\psi(\alpha_0^{old})+\frac1T\sum_{j=1}^T\log x_{ij}\big)$（$\psi$ 为 digamma 函数），且 $\alpha$ 是估出来的、不是手调的。为什么有效：这个低维向量比一个类别标签信息量大得多——$\alpha_i$ 编码了对类 $i$ 的证据浓度、$\alpha_0$ 编码了整条试次的预测不确定性/方差，等于把"模型预测了什么 + 有多自信、多一致"一起压进一个语义丰富的向量；新域带来的信号差异会让这些预测分布产生位移，而 Dirichlet 投影正好把这种**分布层面的漂移**显式地、紧凑地刻画出来。可视化（Figure 3）佐证：Dirichlet 特征空间类间 KL 散度 >31.85、类内协方差 <0.27，类簇分得很开。

**3. GMM-驱动的贝叶斯推断：用历史参数的密度当似然，无梯度地校准后验**

有了每条试次的 $\hat\alpha$，怎么"适应"而不动权重？作者维护一个按预测标签组织的 memory bank $\mathcal{M}_y$，只存高置信试次的 $\alpha$，对每一类用 GMM 做非参数密度估计，得到类条件似然

$$p_{GMM}(\alpha\mid y)=\sum_{k=1}^K\pi_{y,k}\,\mathcal{N}(\alpha;\mu_{y,k},\Sigma_{y,k}).$$

GMM 同时编码了历史试次的全局分布（各成分）和当前待校准试次的邻域信息（当前 $\alpha$ 越靠近某簇，似然越大）。对当前试次，把 GMM 似然和深度模型先验 $p_\theta(y)=f_\theta(s_i)$ 按贝叶斯公式融合：

$$p_{cal}(y\mid\hat\alpha_{MLE})=\frac{p_{GMM}(\hat\alpha_{MLE}\mid y)\,p_\theta(y)}{\sum_{y'=1}^{|L|}p_{GMM}(\hat\alpha_{MLE}\mid y')\,p_\theta(y')},$$

最终预测 $\hat y_{cal}=\arg\max_y p_{cal}(y\mid\hat\alpha_{MLE})$。之后再把当前 $\hat\alpha$ 按置信度阈值 $\tau_{conf}$ 和熵阈值 $\tau_{ent}$ 决定是否回写 memory bank：满了就丢最老的，只让高置信、低不确定性的试次进来，每条最多贡献 $1/M$ 给 GMM，防止少数近期试次主导后验；每次插入后用标准 EM 在对应 bank 上重拟合 GMM（因为 Dirichlet 特征维度极低，全量 EM 开销很小，比在线增量聚类更简单且无精度损失）。为什么有效：整个校准只是密度估计 + 贝叶斯融合，**完全绕过梯度优化**，既躲开了灾难性遗忘，又避免了反传的计算开销，还自带理论依据。

### 损失函数 / 训练策略
源域上正常预训练 SincAdaptNet 分类器；测试阶段**不做任何梯度更新**——适应完全靠 Dirichlet 投影（MLE 定点迭代）+ GMM（EM 拟合）+ 贝叶斯后验融合完成。预处理只用 1–48 Hz 带通滤波 + Euclidean Alignment（EA）。关键超参为 GMM 成分数 $K$、置信度阈值 $\tau_{conf}$、熵阈值 $\tau_{ent}$，三者在合理区间内都不敏感。

## 实验关键数据

### 主实验
在 MOABB 的三个 MI 数据集（BNCI2014001/002, BNCI2015001）和 SHU MI 上做跨被试 LOSO，所有方法独立跑 10 次取均值。

| 数据集（跨被试 LOSO） | 指标 | BTTA-DG | 之前最好基线 | 提升 |
|--------|------|------|----------|------|
| BNCI2014001 | Acc(%) | **78.70** | OTTA 77.58 | +1.12 |
| BNCI2014002 | Acc(%) | **80.29** | OTTA 78.29 | +2.00 |
| BNCI2015001 | Acc(%) | **77.92** | OTTA 76.20 | +1.72 |
| BNCI2014001 跨 session | Acc(%) | **86.50** | OTTA 83.91 | +2.59 |

值得注意的是，多个梯度派基线（Tent、PL）在 TTA 后反而**掉点**——因为在线单试次（batch=1）下噪声试次诱导误导梯度更新 BN、覆盖预训练结构（灾难性遗忘）；BTTA-DG 冻结网络、在概率参数空间适应，规避了这一失败模式。

**计算效率**（Table 7，BNCI2014001）：BTTA-DG 平均推理 15.7 ms/试次、141.6 MFLOPs，比 T-TIME 快 17.8%、比 OTTA 快 24.2%；只有纯重算 BN 统计的 BN-adapt（5.1 ms）更快但性能差一截。

### 消融实验
Table 8 逐组件拆解（多数据集均值）：

| 配置 | 2014001 跨session | 2014002 | 说明 |
|------|------|------|------|
| SincAdaptNet (Source Only) | 80.62 | 76.40 | 主干基线 |
| BTTA-DG w/o EA | 81.88 | 77.55 | 去 EA，保留 Dirichlet+GMM，仍涨 → 模块自带适应力 |
| SincAdaptNet + EA | 82.33 | 78.05 | 标准域对齐 |
| + EA + GMM | 82.47 | 78.25 | 直接在均值概率上做 GMM，仅微涨 |
| + EA + Dirichlet | 84.04 | 78.88 | Dirichlet 投影是涨点主力 |
| **BTTA-DG (Full)** | **86.50** | **80.29** | 三件套齐活，比源模型绝对涨约 2–6% |

### 关键发现
- **Dirichlet 投影是核心增益来源**：在均值概率上直接套 GMM（+EA+GMM）几乎没用（82.47 vs 82.33），但把投影换成 Dirichlet（+EA+Dirichlet）一下涨到 84.04——说明价值在于"建模整条时序轨迹的分布"，而非 GMM 本身。
- **超参不敏感**：$K\in[2,12]$ 精度稳定在 78.2–78.7%；提高 $\tau_{conf}$(0.53–0.65) 过滤低置信试次反而把精度从 78.0% 推到 78.7%；低熵阈值(0.65–0.70) 同样稳。
- **类不平衡下不崩反而专精**：测试集类比从 1:1 推到 1:0.25 时，整体精度平滑下降（78.70%→68.77%），但少数类精度反而升（80.40%→85.19%）——模型会向稀有事件特化，对真实 BCI 场景是好性质。
- **生理可解释性**：学到的 Sinc 滤波器累计响应在 mu(11.2 Hz)/beta(30.5 Hz)/gamma(55.3 Hz) 处能量集中，<8% 落在 MI 节律外，空间核也呈现 CSP 式的额/中央/顶/枕区拓扑。

## 亮点与洞察
- **"适应点估计 → 适应分布表征"的范式转换**最值得记：把每条试次从一个类别预测升级成一个 Dirichlet 浓度向量，等于同时编码了"预测什么 + 多自信 + 多一致"，这比单次输出对域漂移鲁棒得多。这个"用低维分布参数刻画域漂移"的思路可迁到任何有时序/多样本预测的 TTA 任务。
- **把"适应"从优化问题改写成密度估计+贝叶斯融合**，是绕开灾难性遗忘和反传开销的巧妙一招——冻结网络、只在一个极低维参数空间里做 GMM+EM，既快又稳，还自带概率解释。
- **Sinc 带通主干**用极少参数换来频谱可解释性，对需要生理可信度的医学/神经信号场景是可复用的好 trick。

## 局限与展望
- **依赖大致平衡的 EEG 数据**：memory bank 上的密度估计在类别极端不平衡或强非平稳时会吃力，作者承认更极端的不平衡处理是未来工作。
- **GMM 假设的局限**：用高斯混合刻画 Dirichlet 参数密度，对高度多模态或重尾的真实分布是否够，论文未深究；$K$ 虽不敏感但仍是预设。
- **未与大预训练 EEG 模型联用**：作者把"BTTA-DG 叠加到大 EEG 预训练模型上看增益是否可加"列为未来方向，目前主干仍是轻量自研网络。
- **跨模态泛化仅是设想**：把 Dirichlet+GMM 校准迁到 fNIRS/ECoG 等其他神经模态尚未验证。

## 相关工作与启发
- **vs 梯度派 TTA（Tent / T-TIME / OTTA）**：他们靠熵最小化/伪标签反传更新参数，在线 batch=1 时易被噪声试次带偏、灾难性遗忘且慢；本文冻结网络、无梯度校准，既避免遗忘又快 17–24%，且性能更高。
- **vs 数据对齐派（BN-adapt / Euclidean Alignment）**：他们只做浅层统计对齐，抓不住深层预测分布的形变；本文用 Dirichlet 投影显式建模整条时序轨迹的分布漂移，消融显示这正是涨点主力。
- **vs SincNet / SwAV 的借鉴**：主干借 SincNet 的可学习带通核拿到频谱可解释性，借 SwAV 的归一化概率空间思想把预测映射到 simplex 上再做 Dirichlet 建模——把两条已有思路嫁接到 EEG-TTA 这个新场景。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把 Dirichlet 分布引入 EEG-TTA，用分布参数替点估计做无梯度贝叶斯校准，思路确实新。
- 实验充分度: ⭐⭐⭐⭐ 四数据集 + 跨被试/跨 session + 效率/可解释/类不平衡/敏感性多角度，扎实；但主干较轻、未与大预训练 EEG 模型对比。
- 写作质量: ⭐⭐⭐⭐ 动机—方法—理论分析链条清晰，公式与可视化到位。
- 价值: ⭐⭐⭐⭐ 实时、抗遗忘、可解释，对实际 BCI 部署有直接意义，方法范式也可外推到其他神经模态。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Architecture-Agnostic Test-Time Adaptation via Backprop-Free Embedding Alignment](architecture-agnostic_test-time_adaptation_via_backprop-free_embedding_alignment.md)
- [\[ICLR 2026\] NEO — No-Optimization Test-Time Adaptation through Latent Re-Centering](neo_no-optimization_test-time_adaptation_through_latent_re-centering.md)
- [\[CVPR 2026\] Energy Waveify and Redistribution for Test-Time Adaptation: A Control System Perspective](../../CVPR2026/self_supervised/energy_waveify_and_redistribution_for_test-time_adaptation_a_control_system_pers.md)
- [\[ICLR 2026\] Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting](test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting.md)
- [\[ICLR 2026\] ZeroSiam: An Efficient Asymmetry for Test-Time Entropy Optimization without Collapse](zerosiam_an_efficient_asymmetry_for_test-time_entropy_optimization_without_colla.md)

</div>

<!-- RELATED:END -->
