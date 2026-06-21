---
title: >-
  [论文解读] InfoScan: Information-Efficient Visual Scanning via Resource-Adaptive Walks
description: >-
  [ICLR 2026][模型压缩][Visual Mamba] InfoScan 给 Mamba 类视觉骨干换掉了固定的栅格/Hilbert 扫描顺序，用「熵+局部方差」量化每个 patch 的信息量，再用强化学习学一条「先看信息密集区」的自适应扫描路径，在分类/检测/分割上以更少参数同时拿到更高精度。
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "Visual Mamba"
  - "内容自适应扫描"
  - "信息论"
  - "强化学习扫描策略"
  - "高分辨率表示学习"
---

# InfoScan: Information-Efficient Visual Scanning via Resource-Adaptive Walks

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=aiGqJwOE2x](https://openreview.net/forum?id=aiGqJwOE2x)  
**代码**: [https://github.com/SIAT-CV-wuyifeng/Infoscan-ICLR2026](https://github.com/SIAT-CV-wuyifeng/Infoscan-ICLR2026)  
**领域**: 高效视觉骨干 / 状态空间模型 (Mamba)  
**关键词**: Visual Mamba, 内容自适应扫描, 信息论, 强化学习扫描策略, 高分辨率表示学习  

## 一句话总结
InfoScan 给 Mamba 类视觉骨干换掉了固定的栅格/Hilbert 扫描顺序，用「熵+局部方差」量化每个 patch 的信息量，再用强化学习学一条「先看信息密集区」的自适应扫描路径，在分类/检测/分割上以更少参数同时拿到更高精度。

## 研究背景与动机
**领域现状**：高分辨率视觉表示学习长期被 ViT 的二次复杂度拖累，token 数随分辨率暴涨；为提效衍生出 token 稀疏化、层级下采样，以及近期的 Mamba/状态空间模型（VMamba、Vim 等）——后者用线性复杂度的结构化扫描把 2D patch 网格压成 1D 序列来捕获长程依赖。

**现有痛点**：这些 SSM 骨干的扫描顺序（raster、zigzag、Hilbert 曲线）都是**内容无关、预先固定**的，对所有 patch 一视同仁。这隐含了「图像信息均匀分布」的强先验，而真实图像里物体边界这类高熵区和大片纯色背景的价值天差地别。固定扫描让语义最丰富的区域和最平淡的背景被同等、同时刻地处理，浪费算力。

**核心矛盾**：现有提效方案多是**被动/事后**的——要么套固定稀疏模式，要么先把所有 patch 扫一遍再做 token 剪枝/重加权。瓶颈始终卡在「全量前向之后」，没人在「输入扫描这一步」就把算力前置分配给重要区域。

**本文目标**：把扫描顺序从「内容无关的几何遍历」升级为「内容自适应的决策过程」，在特征聚合开始之前就优先获取高信息区，从而在效率—精度权衡上做到更优。

**核心 idea（信息增益驱动的前置感知优先级）**：将扫描形式化为「最大化累积折扣信息增益」$\pi^* = \arg\max_{\pi}\sum_{t=1}^{N}\gamma^{t-1} I_{s_t}$ 的策略优化问题——折扣因子 $\gamma$ 鼓励早一步采集高信息 patch，于是「先看哪里」本身成了可学习的目标。

## 方法详解

### 整体框架
InfoScan 建立在 **VISS（Visual Information State Space）块**之上：相比标准 VSS 块，它把负责 2D 扫描的 SS2D 组件替换为三个协作模块——**信息打分模块 ISM** 量化每个 patch 的信息量、**Patch 选择模块 PSB** 通过代价函数选最优 patch 尺寸、**路径规划模块 PPM** 把扫描建模成马尔可夫决策过程并用强化学习学出自适应路径。三者构成「先评估信息→再定粒度→最后规划路线」的流水线，整个网络以层级化 VISS 块堆叠成骨干，可直接替换 ViT/Mamba 用于分类、检测、分割。

```mermaid
flowchart LR
    A[图像 patch 网格] --> B[ISM 信息打分<br/>熵+局部方差]
    B --> C[PSB 最优 patch 尺寸<br/>效率/信息代价权衡]
    C --> D[PPM 路径规划<br/>MDP + 强化学习]
    D --> E[自适应扫描序列 Sπ]
    E --> F[VISS 块状态空间建模]
    F --> G[分类/检测/分割头]
```

### 关键设计

**1. 信息打分模块（ISM）：用熵与方差给每个 patch 标价**。InfoScan 不靠网络隐式学重要性，而是直接用图像统计量给出内容自适应先验，复合得分 $I(S)=\omega_1\hat{H}+\omega_2\hat{V}$（$\omega_1+\omega_2=1$）。其中 $\hat{H}$ 是把 RGB 三通道各量化成 $C$ 个 bin、按 $H=-\sum_k p_k\log p_k$ 算出的香农熵（衡量全局色彩多样性），$\hat{V}$ 是 $3\times3$ 邻域内的局部强度方差（衡量纹理复杂度），两者都做零均值单位方差标准化。在 ImageNet 验证集上网格搜索得到 $\omega_1=0.6,\omega_2=0.4$——略偏向全局色彩多样性，且此权重直接固定迁移到所有下游任务。它还定义了边界显著度 $I_b(e)=I(S_1)\cdot I(S_2)$，鼓励扫描路径在高信息区之间穿行以增强上下文连贯。

**2. Patch 选择模块（PSB）：把「切多大块」解成一道优化题**。patch 边长 $N_p$ 存在根本权衡——太小会割裂空间上下文且 patch 数暴增、太大丢细节又增加单 patch 时延，于是把它写成最小化总代价 $C_{total}(N_p)=\lambda C_e(N_p)+(1-\lambda)C_{info}(N_p)$。效率项用在目标硬件上拟合的幂律时延模型 $C_e(N_p)=k_1 I\cdot N_p^{\alpha-2}$（$\alpha$ 反映 patch 处理的有效时间复杂度）；信息损失项建成 U 形 $C_{info}(N_p)=k_2/N_p^{\beta}+k_3 N_p^{\gamma}$，第一项随 $N_p$ 增大衰减（全局上下文更完整）、第二项随之增长（局部分辨率变差）。求解时先在可行区间定位极小值，再在连续松弛空间做黄金分割搜索，最后取整到合法 $N_p$；权衡系数 $\lambda$ 只按预设时延/显存预算标定一次便固定。

**3. 路径规划模块（PPM）：把扫描重构成引导式随机游走的 MDP**。把图像切成 $n\times n$ 网格，状态 $s_t=(i_t,j_t)$ 可附带访问图 $V_t$，动作集为四连通 $A=\{\uparrow,\downarrow,\leftarrow,\rightarrow\}$，转移在策略 $\pi_\theta(a_t\mid s_t)$ 下确定 $s_{t+1}=f(s_t,a_t)$。不同于均匀转移的传统随机游走，这里是**内容引导的随机游走**，目标是学策略 $\pi_\theta$ 最大化期望累积回报 $\mathbb{E}_{\pi_\theta}[\sum_t \gamma^t r(s_t,a_t,s_{t+1})]$，从而在一个框架里统一「探索未访问区」与「利用语义密集区」两个目标。

**4. 奖励驱动扫描：用「间隔重复」让重要区域被多看几眼**。奖励函数受人类认知里「间隔重复、集中复习」的启发设计成三项：$r=\underbrace{I(s_{t+1})\cdot\alpha^{k(s_{t+1})}}_{\text{自适应复访激励}}+\underbrace{\lambda(1-V_t(s_{t+1}))}_{\text{探索奖励}}+\underbrace{\beta N_{visited}(s_{t+1})}_{\text{邻域信息增益}}$。其中复访激励的衰减因子 $\alpha$ 是**内容自适应**的：当 $I(s)>\theta$（高显著区）取较大的 $\alpha_{high}$、否则取 $\alpha_{low}$，使高价值 patch「被记得更久」、得到稀疏但周期性的重扫；探索奖励确保全图系统覆盖。论文报告该奖励设计能捕获 98% 的可达信息增益。

## 实验关键数据

### 主实验表格
ImageNet-1K 分类（224²，Thr./Train 为每 GPU img/s）：

| 方法 | Params(M) | FLOPs(G) | Acc(%) |
|------|-----------|----------|--------|
| DeiT-S | 22 | 4.6 | 74.70 |
| Swin-S | 50 | 8.7 | 83.23 |
| VMamba-S | 50 | 8.7 | 83.24 |
| VMamba-B | 89 | 15.4 | 84.32 |
| **InfoScan-T** | **10** | **2.5** | **83.43** |
| **InfoScan-S** | **24** | **4.8** | **84.64** |
| **InfoScan-B** | **38** | **8.4** | **85.19** |

InfoScan-S 仅 24M 参数即超 VMamba-S（50M）+1.4%；InfoScan-T 用 10M 参数就达到 83.43%，逼近 VMamba-B 的 84.32% 而参数少近 9 倍。

MSCOCO2017 检测（Mask R-CNN，APb/APm）：InfoScan-B 以 78M 参数拿到最高 49.8 APb / 44.7 APm，优于 Swin-B（107M）、ConvNeXt-B（108M），少约 30M 参数。

### 消融实验表格
核心模块消融（512×512）：

| Patch 选择 | 路径规划 | ImageNet Top-1 | ADE-20K mIoU | BraTS-2021 mIoU |
|:---:|:---:|:---:|:---:|:---:|
| ✗ | ✗ | 82.5 | 45.3 | 18.7 |
| ✓ | ✗ | 83.4 | 45.7 | 18.9 |
| ✓ | ✓ | **85.9** | **45.9** | **19.3** |

奖励三项消融（M1 复访 / M2 探索 / M3 邻域增益）：

| M1 | M2 | M3 | ImageNet Top-1 | ADE-20K mIoU | BraTS mIoU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ✗ | ✗ | ✗ | 80.4 | 42.3 | 16.7 |
| ✓ | ✗ | ✗ | 81.1 | 42.7 | 17.8 |
| ✓ | ✓ | ✗ | 84.2 | 43.6 | 18.3 |
| ✓ | ✓ | ✓ | **85.9** | **45.9** | **19.1** |

### 关键发现
- **重复扫描收益递减、自适应路由才是关键**：把固定模式扫三遍（Triple Raster/Hilbert）几乎无增益甚至倒退（Raster 三遍让 ADE-20K mIoU 降到 41.8%），说明增益来自「按内容路由」而非「多扫几次」。
- **随机起点能补一点覆盖率**：随机化 Hilbert 起点把 Top-1 从 83.6% 提到 84.5%、ADE mIoU 从 42.9% 提到 43.5%，但固定模式即便随机化仍不及自适应扫描。
- **跨自然/医学图像泛化**：分割在 ADE-20K 与脑瘤 BraTS-2021 上同时验证，InfoScan-S 以少 56% 参数匹配 UPerNet-ResNet-101。

## 亮点与洞察
- **把「扫描顺序」从超参提升为可学习目标**：以往 SSM 视觉骨干只是在 raster/Hilbert 间挑一个，本文第一次把「先看哪、后看哪」明确写成最大化累积折扣信息增益的优化问题，并给出端到端可训练的实现。
- **前置（proactive）效率 vs 事后（reactive）剪枝**：主流提效是先全量前向再剪 token，InfoScan 把效率瓶颈搬到输入扫描阶段、在特征聚合前就分配算力，思路上更彻底。
- **无须额外网络的信息先验**：用熵+局部方差这种轻量统计量做打分，避免再训一个重要性预测器，且权重一次标定全任务复用，工程上很省。
- **认知科学动机落地**：把「间隔重复/集中复习」翻译成内容自适应的奖励衰减 $\alpha^{k}$，让重要区域被周期性重扫，是个清晰可解释的归纳偏置。

## 局限与展望
- **扫描策略训练成本与稳定性**：RL 学扫描路径引入策略网络与奖励调参（$\alpha_{high},\alpha_{low},\theta,\lambda,\beta$ 多个超参），论文把多数细节放进附录，实际复现的训练开销与稳定性还需更多披露。
- **信息打分基于低层统计量**：熵/方差刻画的是颜色与纹理复杂度，对「语义重要但低纹理」的区域（如纯色却关键的物体）可能低估，得分先验与语义重要性未必对齐。
- **权重固定迁移的假设**：$\omega_1=0.6,\omega_2=0.4$ 在 ImageNet 上搜得后固定到所有任务，跨域（如医学影像）是否始终最优值得进一步检验。
- **部分实验口径**：消融在 512×512 下报告的 Top-1（85.9%）与主表 224² 口径不同，跨表对比需注意输入分辨率差异。

## 相关工作与启发
- **高效/自适应计算**：稀疏注意力、动态 token 剪枝、条件计算都试图按输入复杂度分配算力，但多为静态或事后；InfoScan 的「前置感知优先级」是对这条线的一个新解法。
- **扫描策略谱系**：从 raster/zigzag 的坐标固定序，到 Hilbert/Z-order 空间填充曲线（最小化相邻 patch 欧氏距离提升局部性），再到本文的内容自适应游走——可看作 SSM 视觉骨干「扫描归纳偏置」演进的一步。
- **启发**：把「序列化顺序」当成可学习对象的思路，可能迁移到其他需要把高维结构压成 1D 序列的场景（点云、视频时空 token、长文档），凡是「顺序影响上下文聚合」的地方都值得问一句「能不能让模型自己学顺序」。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 把 SSM 视觉骨干的扫描顺序首次形式化为「最大化累积折扣信息增益」并用 RL 学自适应路径，相对固定扫描是清晰的范式升级。
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖分类/检测/分割、自然+医学三类数据，含核心模块与奖励三项的消融，并专门做了「重复 vs 路由」的解耦分析；但训练开销与部分口径披露不足。
- **写作质量**: ⭐⭐⭐⭐ — 动机—数学框架—模块设计逻辑连贯，公式与可视化到位；附录承担了较多关键细节。
- **价值**: ⭐⭐⭐⭐ — 给高分辨率高效视觉表示提供了「内容自适应扫描」这一可复用、可迁移的新方向，在效率—精度权衡上有实打实的参数/精度优势。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)
- [\[ICLR 2026\] INSTANT: Compressing Gradients and Activations for Resource-Efficient Training](instant_compressing_gradients_and_activations_for_resource-efficient_training.md)
- [\[ICLR 2026\] InfoTok: Adaptive Discrete Video Tokenizer via Information-Theoretic Compression](infotok_adaptive_discrete_video_tokenizer_via_information-theoretic_compression.md)
- [\[AAAI 2026\] Sharp Eyes and Memory for VideoLLMs: Information-Aware Visual Token Pruning for Efficient and Reliable VideoLLM Reasoning](../../AAAI2026/model_compression/sharp_eyes_and_memory_for_videollms_information-aware_visual_token_pruning_for_e.md)
- [\[ICLR 2026\] Expressive yet Efficient Feature Expansion with Adaptive Cross-Hadamard Products](expressive_yet_efficient_feature_expansion_with_adaptive_cross-hadamard_products.md)

</div>

<!-- RELATED:END -->
