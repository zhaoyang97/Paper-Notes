---
title: >-
  [论文解读] Learning Mixtures of Linear Dynamical Systems via Hybrid Tensor-EM Method
description: >-
  [ICLR 2026][时间序列][线性动态系统混合 (MoLDS)] 本文提出一种混合 Tensor-EM 框架学习"线性动态系统混合 (MoLDS)":先用基于同时矩阵对角化 (SMD) 的张量矩方法做全局一致初始化,再用完整的 Kalman 滤波-平滑 EM 做局部精修,兼顾全局可辨识性与统计最优性,并首次把 MoLDS 成功用到非人灵长类的真实神经数据上。
tags:
  - "ICLR 2026"
  - "时间序列"
  - "线性动态系统混合 (MoLDS)"
  - "张量矩方法"
  - "同时矩阵对角化 (SMD)"
  - "Kalman EM"
  - "神经数据建模"
---

# Learning Mixtures of Linear Dynamical Systems via Hybrid Tensor-EM Method

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=1GkzDABbME](https://openreview.net/forum?id=1GkzDABbME)  
**代码**: 待确认  
**领域**: 时间序列 / 动态系统辨识  
**关键词**: 线性动态系统混合 (MoLDS)、张量矩方法、同时矩阵对角化 (SMD)、Kalman EM、神经数据建模  

## 一句话总结
本文提出一种混合 Tensor-EM 框架学习"线性动态系统混合 (MoLDS)":先用基于同时矩阵对角化 (SMD) 的张量矩方法做全局一致初始化,再用完整的 Kalman 滤波-平滑 EM 做局部精修,兼顾全局可辨识性与统计最优性,并首次把 MoLDS 成功用到非人灵长类的真实神经数据上。

## 研究背景与动机
**领域现状**：线性动态系统 (LDS) 是建模高维时间序列(尤其神经数据)的主力工具,但单个 LDS 难以刻画异质性——不同实验条件下的轨迹往往有不同的动力学。混合线性动态系统 (MoLDS) 把每条轨迹视作来自某一个潜在 LDS 分量,从而能同时恢复子系统数量、各自参数和混合权重,天然适合"很多 trial、多种条件"的神经实验结构。

**现有痛点**：MoLDS 在复杂、噪声大的真实场景里很难落地。学习它有两条互补但都不完美的路线:(1) 张量矩方法把高阶矩重排成结构化张量再分解,有全局可辨识性保证,但在真实噪声下矩估计本身就不准,参数恢复会退化;(2) EM 等似然方法灵活,但对初始化极度敏感、容易陷进糟糕的局部极小,而 MoLDS 参数空间高维、似然面高度多峰,这个毛病被放大。

**核心矛盾**：全局可辨识(张量)与统计高效(EM)无法兼得——纯张量在高噪声下精度差,纯随机初始化 EM 不稳定且收敛慢。

**本文目标**：构造一个既稳健又高效的 MoLDS 学习管线,在数据有限、高噪声、分量分离差的困难场景下都能可靠恢复,并真正用于真实神经数据分析。

**核心 idea**：**「张量做全局初始化 + Kalman EM 做局部精修」的混合策略**——用张量方法把估计直接送进最大似然估计的"吸引域",再让 EM 收敛到统计最优。其中两点新意:用 SMD 替代鲁棒张量幂法 (RTPM) 做更稳的张量分解,以及补上此前张量方法缺失的噪声参数 $(Q,R)$ 初始化。

## 方法详解

### 整体框架
MoLDS 建模 $N$ 条输入-输出轨迹,每条由 $K$ 个潜在 LDS 分量之一生成;分量标签 $z_i\sim\text{Multinomial}(p_{1:K})$ 未知,目标是恢复混合权重与各分量 LDS 参数 $\{p_k,(A_k,B_k,C_k,D_k,Q_k,R_k)\}$(只在排列与相似变换的歧义下可辨识)。整条管线分两阶段串联:**Stage 1 张量初始化**给出全局一致的权重与系统矩阵估计,中间补一步从残差协方差初始化噪声参数,**Stage 2 EM 精修**用 Kalman 滤波-平滑把所有参数推到统计最优。

```mermaid
flowchart LR
    A[输入-输出轨迹] --> B[滞后输入重构<br/>MoLDS→MLR]
    B --> C[构造 2/3 阶矩张量 M2,M3]
    C --> D[白化 + SMD 分解<br/>恢复权重与 Markov 参数]
    D --> E[Ho-Kalman 实现<br/>恢复 A,B,C,D]
    E --> F[残差协方差<br/>初始化 Q,R]
    F --> G[Kalman EM 精修<br/>E步责任度 / M步闭式MLE]
    G --> H[输出 MoLDS 参数]
```

### 关键设计

**1. 把 MoLDS 重构成 MLR,让混合结构暴露在高阶矩里**：直接对 LDS 做张量分解很难,本文借用"滞后输入表示"把 MoLDS 转写成混合线性回归 (MLR)——用输入及其历史值拼成增广协变量,LDS 的脉冲响应(Markov 参数)就成了每个分量的回归向量。这样混合结构就显现在输入-输出数据的二、三阶交叉矩里,可以走成熟的谱张量工具。具体先构造去偏的二、三阶矩

$$M_2=\frac{1}{2|N_2|}\sum_{j\in N_2}\tilde y_j^2\big(v_j\otimes v_j-I_d\big),\quad M_3=\frac{1}{6|N_3|}\sum_{j\in N_3}\tilde y_j^3\big(v_j^{\otimes3}-E(v_j)\big),$$

其中 $v_j$ 是归一化的滞后输入、$\tilde y_j$ 是输出,$I_d,E$ 是 Stein 型修正项,$N_2,N_3$ 取不相交样本子集以保证无偏。

**2. 用 SMD 做白化后的张量分解,换稳健性**：对 $M_2$ 估出白化变换 $W$,把三阶矩白化成对称正交可分解张量

$$\hat T=M_3(W,W,W)=\sum_{k=1}^{K}p_k\,\alpha_k^{\otimes3},$$

其中 $\alpha_k$ 是各分量回归向量(尺度化 Markov 参数)的白化表示。该分解在排列与符号歧义下唯一,直接给出 $\{\alpha_k,p_k\}$。关键替换在于:不用以往工作的鲁棒张量幂法 (RTPM),而改用**同时矩阵对角化 (SMD)** 来做这步分解。SMD 对张量切片同时对角化,数值上更稳、对噪声更鲁棒——这正是真实神经数据这种高噪声场景最缺的特性。拿到 $\alpha_k$ 后再用 **Ho-Kalman 实现**从 Markov 参数反解出状态空间矩阵 $(\hat A_k,\hat B_k,\hat C_k,\hat D_k)$,作为下一阶段的可靠初值。

**3. 补上噪声参数初始化,衔接两阶段**：张量阶段恢复不了过程/观测噪声 $(Q_k,R_k)$——它们不进矩结构。本文用张量阶段估出的系统矩阵算出每条轨迹的残差,再用残差协方差给 $(\hat Q_k^{(0)},\hat R_k^{(0)})$ 一个有原则的初值。这一步以往张量方法直接缺失,却是 Kalman EM 能正常起步的前提,是把两阶段真正缝起来的关键。

**4. 责任度加权的完整 Kalman EM 精修**：精修阶段把经典混合 EM 推广到 MoLDS。E 步用每个分量的 Kalman 滤波似然算"轨迹级责任度"

$$\gamma_{i,k}^{(t)}=\frac{\exp\!\big(\log\hat p_k^{(t)}+\log p(y_{i}\mid u_{i},\hat\theta_k^{(t)})\big)}{\sum_{k'}\exp\!\big(\log\hat p_{k'}^{(t)}+\log p(y_{i}\mid u_{i},\hat\theta_{k'}^{(t)})\big)},$$

再用 Kalman 平滑器算责任度加权的充分统计量 $S_k^{(t)}$;M 步对所有 LDS 参数(含 $Q,R$)走闭式 MLE 更新、对混合权重取 $\hat p_k=\frac1N\sum_i\gamma_{i,k}$。相比 MLR 文献里常用的交替最小化(用硬指派),这里用的是带概率责任度的软指派,能正确处理噪声下的不确定性,也才配得上真实数据的需求。

## 实验关键数据

### 主实验(合成数据 + 真实神经数据)

| 场景 | 设置 | 对比方法 | 结论 |
|---|---|---|---|
| 简单合成 MoLDS | $K=3$, $n=2$, $m=p=1$ | SMD-Tensor vs RTPM-Tensor | SMD 在 **91%** 的 $(N,T)$ 配置下 Markov 误差更低,趋势更稳 |
| 复杂合成 MoLDS | $K=6$, $n=3$, $m=p=2$, $D=0$ | Tensor-EM vs 纯张量 vs 随机 EM | Tensor-EM 同时取得最低 Markov 误差与权重误差,收敛迭代数远少于随机 EM |
| Area2 真实数据 | 猴体感皮层, 65 单元→6 PC, 8 方向 center-out | Tensor-EM / Random-EM / 纯张量 | 验证集 BIC 选出 **3 分量**,无监督聚类与"按方向监督 LDS"高度一致 |
| PMd 真实数据 | 猴背侧运动前皮层, 16 PC, 连续方向 sequential reach | Tensor-EM / Random-EM | 验证集选 **4 分量**,各分量特化到不同运动方向、脉冲响应轮廓各异 |

### 消融与对比发现

| 对比维度 | 纯张量 | 随机 EM | Tensor-EM(本文) |
|---|---|---|---|
| Markov 参数误差 | 中(高噪声退化) | 高且方差大 | **最低** |
| 混合权重误差 | 中 | 最差 | **最低** |
| 收敛迭代数 | — | 多 | **少** |
| 跨 run 稳定性 | — | 差 | **稳定**(张量初值减少随机性) |

### 关键发现
- **初始化决定一切**：随机 EM 在复杂 MoLDS 上经常陷局部极小、方差极大;张量初值把它送进吸引域后,既更准又更快收敛。
- **SMD > RTPM**：在 MoLDS/神经数据这种高噪声设定,SMD 的数值稳健性带来系统性优势。
- **真实数据可用且可解释**:Area2 上 Tensor-EM 的无监督 trial 聚类与"按方向监督训练的 LDS"几乎一致,且脉冲响应高度相似;同样无监督训练的 SLDS 给不出有意义的跨 trial 聚类,凸显 MoLDS 专攻"trial 间异质性"而非"trial 内切换"。

## 亮点与洞察
- **方法论上把"全局可辨识 + 统计最优"真正缝合**:不是简单堆两个方法,而是补齐了 $(Q,R)$ 初始化这一缺失环节,让张量阶段的输出能干净地喂进完整 Kalman EM。
- **用 SMD 替换 RTPM 是个朴素但关键的工程/理论选择**——把张量分解的稳健性瓶颈打开,直接受益于高噪声真实数据。
- **从纯合成走向真实神经数据**:此前 MoLDS 张量工作几乎只在合成数据上验证,本文是把它落到非人灵长类皮层记录、并和监督基线对齐的少数尝试,对神经科学"群体是否跨条件复用共享潜在动力学"这一问题给了可操作的分析工具。

## 局限与展望
- **强结构假设**:依赖输入持续激励、各分量可控可观、分量充分分离等标准条件;真实数据若违反(如分量高度重叠)恢复质量会下降。
- **聚合参数误差是粗指标**:$A,B,C$ 可在相似变换下变化而不改动力学,论文也承认聚合误差不如 Markov 参数误差精确。
- **分量数 $K$ 仍靠验证集 BIC/NLL 搜索**,没有端到端自动确定;面向更大规模、连续分布方向(如 PMd)时需要分箱近似。
- **线性假设**:MoLDS 假设每条短轨迹由单个线性 LDS 良好描述,对强非线性或 trial 内 regime 切换的数据,SLDS/dLDS 更合适——两类方法是互补而非替代。

## 相关工作与启发
- **混合模型谱系**:MoG、MLR 捕捉异质性但无显式时间结构;MoLDS 经滞后输入与 MLR 建立连接,从而继承谱张量与优化工具,又保留对动力学的建模力。
- **LDS 变体**:SLDS、dLDS 面向单条长轨迹的 regime 切换/渐变,MoLDS 面向"多条独立短轨迹各属一个 LDS",定位不同。
- **张量 + EM 范式**:MLR 文献早已证明"张量初始化 + 交替最小化"有效(Yi 2016 等);本文把它升级到更复杂的 MoLDS,并用带概率责任度的完整 Kalman EM 替换硬指派的交替最小化。
- **启发**:在"全局保证但噪声敏感"与"灵活但易陷局部"的方法对之间,补齐衔接环节(如此处的 $(Q,R)$ 初始化)往往比单独优化任一端更有价值;这一"代数初始化 + 似然精修"的思路可迁移到其他潜变量时序模型。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 单点创新(SMD 替换 + $(Q,R)$ 初始化 + 完整 Kalman EM)虽不颠覆,但把 MoLDS 学习的两条路线干净缝合并首次落到真实神经数据,组合贡献扎实。
- **实验充分度**: ⭐⭐⭐⭐ — 合成数据上系统扫 $(N,T)$ 与 $K$、对齐多基线,真实数据上有监督 LDS/SLDS 对照与验证集模型选择,证据链完整;但仅两个神经数据集、缺更大规模与跨物种验证。
- **写作质量**: ⭐⭐⭐⭐ — 动机-矛盾-方法-验证脉络清晰,算法与公式表述规范,图示丰富;部分关键细节(矩构造、Ho-Kalman)压在附录。
- **价值**: ⭐⭐⭐⭐ — 为神经科学"异质动力学建模"提供了可靠且可解释的实用工具,方法范式对潜变量时序模型有迁移价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Tensor learning with orthogonal, Lorentz, and symplectic symmetries](tensor_learning_with_orthogonal_lorentz_and_symplectic_symmetries.md)
- [\[ICML 2026\] Embedding Hybrid Systems into Continuous Latent Vector Fields](../../ICML2026/time_series/embedding_hybrid_systems_into_continuous_latent_vector_fields.md)
- [\[ICLR 2026\] Learning Linear State-Space Models with Sparse System Matrices](learning_linear_state-space_models_with_sparse_system_matrices.md)
- [\[ICLR 2026\] SONATA: Synergistic Coreset Informed Adaptive Temporal Tensor Factorization](sonata_synergistic_coreset_informed_adaptive_temporal_tensor_factorization.md)
- [\[ICLR 2026\] Weight-Space Linear Recurrent Neural Networks](weight-space_linear_recurrent_neural_networks.md)

</div>

<!-- RELATED:END -->
