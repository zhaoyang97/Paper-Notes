---
title: >-
  [论文解读] EvoBrain: Dynamic Multi-Channel EEG Graph Modeling for Time-Evolving Brain Networks
description: >-
  [NeurIPS2025 Spotlight][医学图像][EEG] 提出 EvoBrain——首次从理论上证明 **显式动态图建模** 优于隐式静态图、**time-then-graph** 架构表达力严格优于其他两种动态 GNN 范式(graph-then-time / time-and-graph)，并据此设计双流 Mamba + Laplacian PE 增强的 GCN 模型，在 TUSZ 和 CHB-MIT 数据集的癫痫检测与早期预测任务上取得 AUROC 提升 23%、F1 提升 30% 的显著效果，同时训练速度比 SOTA 快 17 倍。
tags:
  - "NeurIPS2025 Spotlight"
  - "医学图像"
  - "EEG"
  - "动态图"
  - "癫痫检测"
  - "Mamba"
  - "时空建模"
  - "GCN"
  - "表达力分析"
---

# EvoBrain: Dynamic Multi-Channel EEG Graph Modeling for Time-Evolving Brain Networks

**会议**: NeurIPS2025 Spotlight  
**arXiv**: [2509.15857](https://arxiv.org/abs/2509.15857)  
**代码**: [GitHub](https://github.com/Kotoge/EvoBrain)  
**领域**: 脑电信号分析 / 动态图神经网络  
**关键词**: EEG, 动态图, 癫痫检测, Mamba, 时空建模, GCN, 表达力分析

## 一句话总结

提出 EvoBrain——首次从理论上证明 **显式动态图建模** 优于隐式静态图、**time-then-graph** 架构表达力严格优于其他两种动态 GNN 范式(graph-then-time / time-and-graph)，并据此设计双流 Mamba + Laplacian PE 增强的 GCN 模型，在 TUSZ 和 CHB-MIT 数据集的癫痫检测与早期预测任务上取得 AUROC 提升 23%、F1 提升 30% 的显著效果，同时训练速度比 SOTA 快 17 倍。

## 研究背景与动机

- **临床背景**: 癫痫是一种脑网络疾病，不同脑区之间的异常连接是癫痫发作的重要标志。脑电图(EEG)是临床监测癫痫的核心工具，自动化检测/预测对临床意义重大
- **已有方案**: 近年来动态 GNN 被用于建模多通道 EEG 的时空特征，但存在两个根本性问题：
  1. **静态图结构**: 大部分方法虽号称"动态"，实际上构建一个固定的邻接矩阵 $\mathbf{A} \in \mathbb{R}^{N \times N}$ 共享给所有时间步，无法反映癫痫发作过程中脑连接性的演变
  2. **时空建模不充分**: 现有动态 GNN 可分为 graph-then-time、time-and-graph、time-then-graph 三种范式，但哪种更优缺乏理论依据，导致实际应用中性能参差不齐
- **动机**: 从理论和实验两个层面解决上述问题，设计既有理论保证又高效的癫痫检测模型

## 核心问题

论文形式化提出两个关键问题：

1. **Problem 1: 隐式 vs 显式动态图建模**
    - 隐式(Implicit): 固定邻接矩阵 $\mathbf{A}_{:,:,t} = \hat{\mathbf{A}}, \forall t$，只有节点特征随时间变化
    - 显式(Explicit): 邻接矩阵随时间变化 $\mathbf{A}_{:,:,t} = f(\mathbf{x}_{:,t})$，节点和边都可演化
    - 问题：两者在表达力上的关系是什么？

2. **Problem 2: 三种动态 GNN 架构的表达力排序**
    - graph-then-time: 先对每个时间步独立做 GNN，再用 RNN 处理时间序列
    - time-and-graph: GNN 与 RNN 交替处理，每步同时做图学习和时序建模
    - time-then-graph: 先用 RNN 分别建模节点/边的时间演化，再对最终图做 GNN
    - 问题：哪种架构表达力最强？

## 理论分析

论文基于 1-WL(Weisfeiler-Lehman)图同构测试框架，给出了严格的理论证明：

### Theorem 1: 显式 > 隐式

显式动态图建模的函数类 $\mathcal{F}_{\text{explicit}}$ 严格包含隐式动态图建模 $\mathcal{F}_{\text{implicit}}$：

$$\mathcal{F}_{\text{implicit}} \subset \mathcal{F}_{\text{explicit}}$$

证明思路：构造两个时序 EEG 图，它们节点特征完全相同但仅在某一时间步的邻接矩阵不同。隐式模型将邻接压缩为静态表示无法区分，而显式模型可以。

### Theorem 3: 架构表达力排序

$$\text{graph-then-time} \precneqq \text{time-and-graph} \precneqq \text{time-then-graph}$$

- **Lemma 1**: graph-then-time 中，历史隐状态 $\mathbf{h}_{i,t-1}$ 直接传递给 RNN cell，无需额外 GNN 编码跨时间步交互，因此表达力弱于 time-and-graph
- **Lemma 2**: time-and-graph 对每个时间步独立处理图信息，导致跨时间步的表示可能相同；而 time-then-graph 先学节点和边的完整时间演化再做图推理，能区分具有不同时序结构的图

### 合成 EEG 任务验证 (Lemma 2 证明)

论文构造了一个巧妙的合成任务来证明 time-and-graph 无法区分而 time-then-graph 可以的图对：

- **设置**: 两个 2 步时序图，每步 8 个节点($\mathcal{C}_{8,1}$ 和 $\mathcal{C}_{8,2}$)，第一步结构相同，第二步拓扑不同
- **time-and-graph 失败**: 由于 1-WL GNN 对 $\mathcal{C}_{8,1}$ 和 $\mathcal{C}_{8,2}$ 输出相同表示（节点特征相同），即使结构不同也无法区分 → 最终 $\mathbf{Z}^{(\text{top})} = \mathbf{Z}^{(\text{btm})}$
- **time-then-graph 成功**: RNN$^{\text{edge}}$ 分别处理两个图的边时间序列，由于 $t_2$ 时刻边不同 → $\mathbf{h}^{\text{edge(top)}} \neq \mathbf{h}^{\text{edge(btm)}}$ → 经 GNN 后 $\mathbf{Z}^{(\text{top})} \neq \mathbf{Z}^{(\text{btm})}$

**核心洞察**: time-then-graph 先用独立 RNN 编码边的完整时间演化，保留了结构差异信息；而 time-and-graph 在每步独立做 GNN，丢失了跨时间步的结构差异。

### 意义

这是动态 GNN 在 EEG 分析场景下的**首个节点级表达力理论分析**，相比 Gao & Ribeiro (2022) 的边/结构层面分析，更贴合 EEG 图构建依赖节点间相似度的实际。同时，Lemma C.1 证明了仅用边特征不足以区分某些时序 EEG 图（存在边特征相同但节点特征不同的图对），因此**必须同时建模节点和边表征**——这正是 EvoBrain 采用双流 Mamba 的理论依据。

## 方法详解

### 1. 显式动态脑图构建

- 将 EEG 信号切分为 $T$ 个快照(snapshot)
- 对每个快照，计算通道 $v_i$、$v_j$ 之间的归一化交叉相关(normalized cross-correlation)作为边权：

$$a_{i,j,t} = |x_{i,t} * x_{j,t}|, \quad \text{if } v_j \in \mathcal{N}(v_i), \text{ else } 0$$

- 通过 top-$\tau$ 策略保留相关性最高的边，生成稀疏有向加权图
- 最终获得 $T$ 个动态图序列 $\mathbf{G}$，其中节点和边均随时间演变

### 2. 双流 Mamba 时序建模

基于 time-then-graph 架构，使用两个独立的 Mamba 流分别处理节点和边的时间序列：

**输入预处理**：对 EEG 信号做 STFT(短时傅里叶变换)，取非负频率分量的 log 幅值，z-normalization 后得到 $\mathcal{X} \in \mathbb{R}^{N \times T \times d}$

**Mamba 状态空间模型**（线性 RNN + 选择性状态更新）：

$$\mathbf{h}_t^e = \underbrace{(1 - \Delta_t^e \cdot \mathbf{D})}_{\text{选择性遗忘}} \mathbf{h}_{t-1} + \underbrace{\Delta_t^e \cdot \mathbf{B}_t^e}_{\text{选择性更新}} \mathbf{x}_t^e$$

- 遗忘项类比突触衰减/抑制过程，淡化过时信息
- 更新项类比神经调节门控(如多巴胺信号)，选择性增强重要新输入
- $\Delta_t^e$ 通过 softplus 保证正值，调控短期/长期记忆权衡

两个流分别输出节点时序表征 $\mathbf{h}_i^{\text{node}} = \mathbf{y}_{i,T}^{\text{node}}$ 和边时序表征 $\mathbf{h}_{ij}^{\text{edge}} = \mathbf{y}_{ij,T}^{\text{edge}}$。

### 3. Laplacian 位置编码(LapPE) + GCN 空间建模

- **LapPE 的神经科学动机**: 脑功能与特定区域密切相关(如新皮层、Broca区)，但标准 GNN 对结构等价的节点给出相同表示，丢失位置信息
- 从边特征 $\mathbf{H}^{\text{edge}}$ 计算加权邻接矩阵 $\mathbf{A}' = \tau_{\text{edge}}(f_{\text{edge}}(\mathbf{H}^{\text{edge}}))$
- 对归一化拉普拉斯矩阵 $\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}'\mathbf{D}^{-1/2}$ 做特征分解
- 取前 $K$ 个最小特征值对应的特征向量作为位置编码 $\mathbf{p}_i$，拼接到节点特征上：

$$\mathbf{x}_i^{\text{node}} = [\mathbf{h}_i^{\text{node}}; \mathbf{p}_i]$$

- **GCN 空间学习**: 多层 GCN 聚合邻居信息 → max pooling → 全连接层 + softmax 输出分类

## 实验关键数据

### 数据集
- **TUSZ v1.5.2**: 最大公开 EEG 癫痫数据库，5612 段记录，3050 个标注发作，19 通道。训练/验证/测试按患者划分（530/61/45 名患者），12s 窗口下训练集约 19.7 万样本（正类 6.9%），60s 窗口约 3.9 万样本（正类 9.3%），类别极度不平衡
- **CHB-MIT**: 844 小时 22 通道头皮 EEG，22 名患者，163 次发作。随机 15% 患者数据用于测试

### 任务
- **癫痫检测**: 区分发作/非发作段(二分类)
- **癫痫早期预测**: 区分发作前(pre-ictal, 1分钟)与正常状态——更具临床价值

### 主要结果(TUSZ 数据集)

| 模型 | 检测 AUROC(60s) | 检测 F1(60s) | 预测 AUROC(12s) | 预测 F1(12s) |
|------|----------------|-------------|----------------|-------------|
| EvolveGCN (graph-then-time) | 0.670 | 0.340 | 0.622 | 0.437 |
| DCRNN (time-and-graph) | 0.808 | 0.435 | 0.634 | 0.401 |
| GRAPHS4MER (time-then-graph) | 0.778 | 0.439 | 0.632 | 0.438 |
| GRU-GCN (time-then-graph) | 0.822 | 0.438 | 0.659 | 0.453 |
| LaBraM (Foundation) | 0.793 | 0.469 | 0.661 | 0.482 |
| **EvoBrain** | **0.865** | **0.483** | **0.675** | **0.470** |

- 相比 EvolveGCN: AUROC +23%, F1 +30%
- 相比 LaBraM: 检测 AUROC +9%，且参数量仅为其 1/30
- CHB-MIT 上 EvoBrain 达到 AUC 0.94

### 效率
- 训练速度比 DCRNN(time-and-graph) 快 **17×**
- 推理速度快 **14×**（因为 GNN 只在最终时刻处理一次，而非每个快照都做）

### 计算复杂度理论分析

论文给出了三种架构的复杂度对比（$V$=节点数, $T$=时间步, $E_t$=第 $t$ 步边数, $E_{\text{agg}}$=聚合图边数, $d$=特征维度）：

| 架构 | 复杂度 |
|------|--------|
| graph-then-time | $\mathcal{O}(VTd^2 + \sum_{t} E_t d)$ |
| time-and-graph | $\mathcal{O}(VTd^2 + \sum_{t} E_t d^2)$ |
| time-then-graph | $\mathcal{O}((V + E_{\text{agg}})Td^2)$ |

- time-then-graph 的关键优势：当聚合图边数 $E_{\text{agg}} \ll \sum_t E_t$ 时（稀疏图场景下通常成立），其复杂度显著低于 time-and-graph
- 原因：time-and-graph 每个时间步都做 GNN（含两个 GNN：$\text{GNN}_{\text{in}}^L$ 和 $\text{GNN}_{\text{rc}}^L$），而 time-then-graph 只在最终聚合图上做一次 GNN

### GPU 显存占用

| 模型 | 训练 (MB) | 推理 (MB) |
|------|-----------|----------|
| EvoBrain | 51.35 | 46.64 |
| GRU-GCN | 54.61 | 52.09 |
| GRAPHS4MER | 369.46 | 93.02 |
| DCRNN | 21.10 | 20.54 |
| EvolveGCN | 22.06 | 20.07 |

- DCRNN/EvolveGCN 显存最低但计算速度慢 10× 以上
- EvoBrain 在 time-then-graph 模型中显存最优（仅 GRAPHS4MER 的 1/7），兼顾速度与显存

### 消融实验
- 相同 GRU + GCN 下，time-then-graph 架构效果最优，验证了理论分析
- 去掉 FFT 预处理或 LapPE 均导致性能下降
- Mamba 在长序列(60s)上比 GRU 优势更明显
- 替换 GCN 为 GIN 性能相当，但去掉 GNN 仅用 RNN 性能显著下降

### 实现细节

- **优化器**: Adam, 学习率 1e-4, 训练 100 epochs
- **损失函数**: Binary cross-entropy
- **稀疏度**: top-$\tau$=3（每个节点仅保留相关性最高的 3 条边）
- **模型规模**: 双流 Mamba 各 2 层 + 2 层 GCN（64 hidden units），总参数量 **114,794**（仅 11.5 万参数）
- **数据增强**: 训练时随机缩放 EEG 幅值(×0.8~1.2)
- **Dropout**: 0（不使用）
- **硬件**: NVIDIA A6000 GPU + Xeon Gold 6258R CPU

### 参数量对比

| 模型 | 参数量 |
|------|--------|
| EvoBrain | 114,794 |
| GRU-GCN | 183,834 |
| EvolveGCN | 200,301 |
| DCRNN | 280,769 |
| LSTM | 536,641 |
| BIOT | 3,187,201 |
| LaBraM | 5,803,137 |
| EEGPT | 51,221,121 |

- EvoBrain 参数量最小，仅为 LaBraM 的 ~1/50、EEGPT 的 ~1/450
- 在参数量远小于 Foundation Models 的情况下，检测性能反超，说明归纳偏置(动态图+time-then-graph)的重要性

## 临床分析亮点

论文展示了动态图结构的可视化分析，具有重要临床价值：

- **正常状态**: 边连接弱且稀疏，分布广泛
- **发作前状态**: 特定区域连接逐渐增强(可用于早期预警)
- **局灶性发作**: 特定脑区持续强连接(可辅助定位发作起始区 SOZ)
- **全面性发作**: 全脑范围强连接

这些动态图模式与神经科学观测一致，为手术规划和治疗策略提供了潜在辅助。

## 亮点

1. **理论驱动设计**: 不是凭直觉选架构，而是先给出严格数学证明再据此设计模型，方法论值得借鉴
2. **双流 Mamba**: 将节点和边的时间演化独立建模，既符合理论要求(分别捕获节点/边动态)，又利用了 Mamba 的线性复杂度优势
3. **神经科学对齐**: LapPE 用于区分不同脑区位置、FFT 特征对应临床频率分析、动态图结构可视化与临床观测一致
4. **效率与性能兼顾**: 17× 加速的同时性能大幅领先，得益于 time-then-graph 架构只需做一次 GNN
5. **临床可解释性**: 动态图可视化直接对应癫痫不同阶段的脑区连接变化

## 局限与展望

1. **早期预测仍有差距**: 在预测任务上虽然领先 GNN 模型，但 LaBraM 等大规模预训练模型在某些指标上仍有优势(受益于大规模预训练和更多参数)
2. **任务泛化性**: 仅在癫痫任务上验证，尚未推广到其他 EEG 任务(如情感识别、运动想象、睡眠分期等)
3. **预发作定义固定**: pre-ictal 固定为发作前 1 分钟，但临床上预发作阶段的持续时间因人而异
- **数据集人群偏差**: 训练数据来自特定人群，对其他人群的泛化性未验证，可能导致不同人群间诊断准确率不均
- **显存非最优**: 虽然推理显存(46.64 MB)远低于 GRAPHS4MER(93.02 MB)，但高于 DCRNN(20.54 MB)和 EvolveGCN(20.07 MB)，在极端边缘部署场景下可能受限
- **可能方向**: 将 EvoBrain 与大规模 EEG 预训练结合；引入自适应 pre-ictal 窗口长度；扩展到其他神经信号(如 iEEG、MEG)

## 与相关工作的对比

| 维度 | Tang et al. 2022 (DCRNN) | Tang et al. 2023 (GRAPHS4MER) | Gao & Ribeiro 2022 (GRU-GCN) | EvoBrain |
|------|------------------------|------------------------------|------------------------------|----------|
| 架构范式 | time-and-graph | time-then-graph | time-then-graph | time-then-graph |
| 图结构 | 静态(固定邻接) | 内部学习但基于整段数据 | 静态 | **显式动态**(逐快照构建) |
| 时序模型 | GRU | S4(结构化状态空间) | GRU | **Mamba**(选择性状态空间) |
| 图模型 | GCN | GCN | GCN | **GCN + LapPE** |
| 理论分析 | 无 | 无 | 有(边/结构级) | **有(节点级，更贴合EEG)** |
| 边特征建模 | 无 | 无 | 有 | **有(独立Mamba流)** |
| 效率 | 慢(每步做GNN) | 中等 | 快 | **最快(17×加速)** |

与 EEG Foundation Models 的对比：
- **LaBraM** (580万参数): 利用大规模多数据集预训练获得强泛化能力，在预测任务上部分指标优于 EvoBrain，但参数量是 EvoBrain 的 30 倍
- **BIOT / EEGPT**: 虽有大规模预训练，但在癫痫检测任务上不如 EvoBrain，说明任务特定的归纳偏置(动态图 + 时空分离建模)仍然重要
- EvoBrain 的优势在于**轻量高效**且**可解释性强**，更适合临床部署

与动态图学习通用方法的区别：
- Gao & Ribeiro (2022) 的理论分析主要关注边/结构级表示，未显式考虑节点特征；而 EEG 图构建本质上依赖通道(节点)间的相似性度量，因此 EvoBrain 的节点级表达力分析更具针对性
- 通用动态图方法(如 FreeDyG、DeepTGC)关注链路预测或节点分类，而 EvoBrain 聚焦于图级分类(癫痫检测)，需要全图表示能力

## 启发与关联

1. **理论先行的模型设计范式**: 本文先证明了架构表达力的严格偏序关系，再据此选择 time-then-graph + 显式动态图的设计方向。这种"理论驱动设计"的方法论值得在其他领域推广——例如在视频理解中，是否也可以先分析不同时空融合策略的表达力再做模型设计？

2. **双流独立建模节点/边演化**: 将节点和边的时间序列分开用两个 Mamba 流处理，既满足理论要求又降低了计算量。这种设计可推广到其他动态图场景：如交通网络(节点=路口流量，边=路段拥堵度)、社交网络(节点=用户状态，边=交互频率)等

3. **Mamba 在图学习中的应用**: 本文是 Mamba 与 GNN 结合的一个成功案例。Mamba 的选择性状态更新机制与神经调节过程的类比很有启发性，为其在生物信号处理中的应用提供了合理的解释框架

4. **LapPE 的神经科学动机**: 不仅仅把 LapPE 当作图表示学习的技术手段，而是赋予其"区分脑区位置"的神经科学意义，使得方法设计与领域知识深度结合

5. **可解释性与性能不矛盾**: EvoBrain 的动态图可视化不仅辅助临床理解，图结构本身也是模型设计的核心(而非事后解释)，体现了"内在可解释性"而非"事后可解释性"

6. **潜在扩展方向**:
    - 将 EvoBrain 的动态图框架应用于 Brain-Computer Interface (BCI) 中的运动想象解码
    - 结合大规模 EEG 预训练(如 LaBraM 式的预训练策略)进一步提升泛化性
    - 自适应 pre-ictal 窗口：用变化点检测替代固定 1 分钟窗口
    - 多模态融合：结合 fMRI 的空间分辨率与 EEG 的时间分辨率

## 评分
- 新颖性: ⭐⭐⭐⭐ (理论分析+显式动态图+双流Mamba的结合是首创，但time-then-graph架构本身已有先例)
- 实验充分度: ⭐⭐⭐⭐⭐ (两个数据集、检测+预测两个任务、12种baseline、消融实验充分、临床可视化分析、效率对比)
- 写作质量: ⭐⭐⭐⭐ (理论部分严谨，方法描述清晰，图表信息量大；但符号较多，初读门槛略高)
- 价值: ⭐⭐⭐⭐ (兼具理论贡献和实用价值，对动态GNN在EEG领域的应用有重要参考意义)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs](dyg-mamba_continuous_state_space_modeling_on_dynamic_graphs.md)
- [\[NeurIPS 2025\] BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals](brainomni_a_brain_foundation_model_for_unified_eeg_and_meg_signals.md)
- [\[CVPR 2026\] Virtual Nodes Guided Dynamic Graph Neural Network for Brain Tumor Segmentation with Missing Modalities](../../CVPR2026/medical_imaging/virtual_nodes_guided_dynamic_graph_neural_network_for_brain_tumor_segmentation_w.md)
- [\[NeurIPS 2025\] EEGReXferNet: A Lightweight Gen-AI Framework for EEG Subspace Reconstruction via Cross-Subject Transfer Learning and Channel-Aware Embedding](eegrexfernet_a_lightweight_gen-ai_framework_for_eeg_subspace_reconstruction_via_.md)
- [\[NeurIPS 2025\] DCA: Graph-Guided Deep Embedding Clustering for Brain Atlases](dca_graph-guided_deep_embedding_clustering_for_brain_atlases.md)

</div>

<!-- RELATED:END -->
