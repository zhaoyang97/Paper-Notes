---
title: >-
  [论文解读] TaCo: A Benchmark for Lossless and Lossy Codecs of Heterogeneous Tactile Data
description: >-
  [ICLR 2026][机器人][触觉数据压缩] 本文提出 TaCo——首个面向触觉数据编解码器的综合 benchmark，在 5 个异构触觉数据集、30 个编解码器、4 类下游任务上系统评测无损与有损压缩，并训练出首批纯触觉数据驱动的无损编解码器 TaCo-LL 与有损编解码器 TaCo-L，在全部任务上刷新 SOTA。
tags:
  - "ICLR 2026"
  - "机器人"
  - "触觉数据压缩"
  - "无损/有损编解码"
  - "异构触觉传感器"
  - "灵巧抓取"
  - "神经编解码器"
---

# TaCo: A Benchmark for Lossless and Lossy Codecs of Heterogeneous Tactile Data

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=1PYXFkS6Hy](https://openreview.net/forum?id=1PYXFkS6Hy)  
**代码**: https://github.com/sjtu-medialab/RwkvCompress （TaCo-L 基于 LALIC，复用该仓库设置）  
**领域**: 机器人 / 具身智能 / 触觉感知 / 数据压缩 / Benchmark  
**关键词**: 触觉数据压缩, 无损/有损编解码, 异构触觉传感器, 灵巧抓取, 神经编解码器

## 一句话总结
本文提出 TaCo——首个面向触觉数据编解码器的综合 benchmark，在 5 个异构触觉数据集、30 个编解码器、4 类下游任务上系统评测无损与有损压缩，并训练出首批纯触觉数据驱动的无损编解码器 TaCo-LL 与有损编解码器 TaCo-L，在全部任务上刷新 SOTA。

## 研究背景与动机

**领域现状**：触觉感知是具身智能的关键模态，给灵巧操作和环境交互提供细粒度的力/形变信息。但触觉传感器分辨率、采样率不断提高，原始数据量爆炸式增长——以 GelSight 为例，640×480×30fps×24bit/pixel 的原始触觉视频码率高达约 200 Mbps。在远程遥操作、灵巧手实时触觉反馈、以及为机器人模型训练大规模存储交互数据这些场景下，带宽和存储都是硬约束，触觉数据压缩因此变得必要。

**现有痛点**：尽管"需要压缩"这件事早有共识，现有工作却高度碎片化。一类是经典信号处理（降维、小波变换、阈值截断 + delta 编码等），通常只利用简单的信号稀疏性或量化策略，缺乏严格的压缩指标，且只针对很窄的场景、泛化性差；另一类数据驱动的神经编解码器虽然在图像/视频上已超越传统方法，却**从未在触觉数据上训练或系统评测过**。整个领域既没有统一的数据集、标准指标，也没有 baseline 模型。

**核心矛盾**：触觉数据天然**异构**——视觉触觉传感器（GelSight、DIGIT）通过内部相机拍摄弹性体表面形变，输出本质是 RGB 图像/视频；而力学触觉传感器（如 Paxini）输出的是多接触点的 3D 力向量序列，结构上和图像完全不同。这种异构性叠加复杂的时空冗余，使得"在某一种触觉数据上预训练的神经编码器"很难迁移到另一种，也让"压缩效率"与"下游任务性能"之间的 trade-off 无人系统刻画。

**本文目标**：把"触觉数据该怎么压"这个问题拆成三个可量化的子问题——(1) 用什么统一表征让现成编解码器能跨异构数据通用；(2) 现有 30 个编解码器在 5 个数据集、4 类任务上到底表现如何；(3) 纯触觉数据训练的编解码器能把性能上界推到多高。

**切入角度**：作者的关键观察是——大量触觉信号都能**自然地转成类图像格式**，从而直接复用成熟、可调码率失真的图像/视频编解码器。这条路在触觉领域几乎无人探索，却恰好能把异构数据统一到同一评测框架下。

**核心 idea**：把异构触觉数据统一图像化，搭一个"5 数据集 × 30 编解码器 × 4 任务"的三维评测矩阵，并首次在触觉数据上端到端训练无损（TaCo-LL）与有损（TaCo-L）编解码器，用数据揭示压缩效率与任务性能的关键权衡。

## 方法详解

### 整体框架
TaCo 不是一个新模型，而是一套"benchmark + 两个数据驱动 baseline"。整体可以分三层看：**底层是统一的异构触觉数据表征**（把视觉触觉、力学触觉都映射成三通道图像），**中层是 30 个编解码器组成的评测池**（17 个现成编解码器 + 13 个神经编解码器，含 14 个无损 / 16 个有损），**上层是 4 类下游任务**（无损存储、人眼可视化、语义分类、灵巧抓取）。在这套框架内，作者再补上两块原本缺失的拼图：把 SOTA 无损图像压缩器 DualComp-I 和 SOTA 有损图像压缩器 LALIC 拿到触觉数据上重训，得到 **TaCo-LL（无损）** 与 **TaCo-L（有损）**，作为"触觉感知压缩"上界的探针。

数据划分上，作者从 Touch and Go 与 ObjectFolder 各取 70% 训练 TaCo-LL/TaCo-L，剩余 30% 加上完整的 SSVTP、YCB-Slide、ObjTac 用于全部方法的评测——后三个数据集对 TaCo 而言完全是"未见过的分布"，用于检验跨传感器、跨数据集的泛化。整套训练在 2 张 A100 上完成。

### 关键设计

**1. 异构触觉数据的统一图像化表征：让一套编解码器吃下所有传感器**

最大的障碍是触觉数据结构不统一，导致没法用同一套编解码器横向比较。作者按传感器原理分两类处理：视觉触觉传感器（GelSight 的 Touch and Go / ObjectFolder、DIGIT 的 SSVTP / YCB-Slide）本身输出 RGB 图像或视频序列，直接喂给图像/视频编解码器即可；力学触觉传感器（ObjTac，由 Paxini 力传感器采集）则有 $N=60$ 个接触点、每点是一个 3D 力向量，作者把**每个 3D 力向量映射成一个 RGB 像素**，再沿时间维堆叠时长 $T$ 的力读数，生成分辨率为 $T\times 60$ 的"图像"。这样无论哪种传感器，最终都是三通道图像，标准图像/视频编解码器与神经编解码器全部可用，30 个编解码器才能在同一坐标系里被公平评测。

**2. 三维评测矩阵：5 数据集 × 30 编解码器 × 4 任务，把"压缩—任务"权衡铺开**

这是 benchmark 的主体。**数据集维度**覆盖 5 个，跨传感器类型（视觉/力学）、分辨率（120×160 到 640×480）和数据规模（共 250K+ 帧）。**编解码器维度**分两大类：现成编解码器包括通用无损（gzip/zstd/bzip2）、图像无损（PNG/FLIF/WebP/JPEG-XL/JPEG2000/BPG）、图像有损（JPEG-XL/JPEG2000 及 HM/VTM 的 intra 与屏幕内容 SCC 模式）、视频有损（VVenC/x265/SVT-AV1）；神经编解码器包括预训练无损（DLPR/P2LLM/DualComp-I 及基于 LLM 的 LMIC，用 RWKV-7B 与 Llama3-8B）和预训练有损（ELIC/TCM/LALIC 三个图像 + DCVC-DC/FM/RT 三个视频）。**任务维度**有 4 类，对应"为谁压缩"：无损压缩服务存储；有损压缩分别服务人眼可视化（PSNR/MS-SSIM）、机器语义（材质/物体分类的 top-1 acc）、机器人控制（灵巧抓取成功率）。这个矩阵的价值在于：它第一次让"某个编解码器在码率省了多少"与"下游任务掉了多少点"能被一一对照。

**3. TaCo-LL：首个纯触觉数据训练的无损编解码器**

预训练神经无损方法（DLPR/P2LLM/DualComp-I）虽能捕获帧内相关性，但受限于"在自然图像上训练"的域不匹配，在非视觉或结构差异大的触觉数据上表现受限。TaCo-LL 把 SOTA 无损图像压缩器 DualComp-I 直接在触觉数据上重训：tokenization 时把输入切成 $16\times16\times3$ 的 patch 以保留局部空间相关性，再按 raster-scan 展平——视觉触觉数据按子像素 $(R_1,G_1,B_1,R_2,G_2,B_2,\dots)$ 顺序展开，三轴力信号则把三个力分量当三个颜色通道展开为 $(x_1,y_1,z_1,x_2,y_2,z_2,\dots)$。网络 $f_a$ 自回归预测下一个符号的分布 $p(x_i\mid x_{<i})$，接算术编码器生成码流，损失就是熵（编码 $x$ 的理论下界）：

$$L = \mathbb{E}\big[-\log_2 p(x_i\mid x_{<i})\big]$$

解码端用算术解码器和同一网络自回归无损还原。TaCo-LL 提供 12M/48M/96M 三档参数，让"压缩率—复杂度"可调。

**4. TaCo-L：首个纯触觉数据训练的有损编解码器**

有损这边，预训练的 ELIC/LALIC/TCM 在部分数据集上不错，但对结构迥异的 ObjTac（力学数据）几乎泛化失败。TaCo-L 基于 SOTA 有损图像压缩器 LALIC 重训，沿用其 VAE 架构：把输入随机裁剪或零填充到 $256\times256$，由于视觉触觉和力学触觉都已是三通道，**无需 tokenization**。分析变换 $g_a$ 将信号映射为隐表示 $y$，量化得 $\hat y$ 后算术编码；超先验分支 $h_a/h_s$ 生成边信息 $z$ 来估计 $\hat y$ 的密度模型，合成变换 $g_s$ 重建 $\hat x$。$g_a/g_s$ 各含四次下/上采样。训练用率失真损失：

$$L = \lambda \cdot D(x,\hat x) + \mathbb{E}\big[-\log_2 p_{\hat y\mid\hat z}(\hat y\mid\hat z)\big]$$

其中 $\lambda$ 控制码率。在触觉数据上重训让 TaCo-L 学到触觉特有的隐分布，从而把率失真曲线整体压到所有 baseline 之下。

## 实验关键数据

### 主实验

**无损压缩（bits/Byte，越低越好，未压缩为 8）**：TaCo-LL-96M 在全部 5 个数据集上夺得最优。

| 数据集 | gzip | JPEG-XL | DualComp-I | TaCo-LL-96M | 压缩率 |
|--------|------|---------|-----------|-------------|--------|
| Touch and Go | 2.298 | 0.739 | 0.948 | **0.447** | 18× |
| ObjectFolder | 3.969 | 3.657 | 3.126 | **2.709** | 3× |
| SSVTP | 2.234 | 1.478 | 1.442 | **1.066** | 8× |
| ObjTac | 0.571 | 0.382 | 0.540 | **0.360** | 22× |
| YCB-Slide | 2.185 | 1.431 | 1.388 | **1.073** | 8× |

**有损压缩（BD-Rate %，以 HM-Intra 为锚点，越负越好）**：TaCo-L 在全部 5 个数据集上最优，尤其 Touch and Go 达到 −61.8%。

| 数据集 | ELIC | LALIC | VTM-Intra | TaCo-L |
|--------|------|-------|-----------|--------|
| Touch and Go | −40.2% | −51.6% | −21.7% | **−61.8%** |
| ObjectFolder | 0.6% | 0.2% | −19.7% | **−24.3%** |
| SSVTP | −5.8% | 4.3% | −16.0% | **−19.2%** |
| YCB-Slide | −9.2% | −4.6% | −24.4% | **−27.4%** |
| ObjTac | 44.5% | 32.8% | −22.0% | **−27.0%** |

值得注意的是预训练神经编解码器（ELIC/LALIC/TCM）在力学数据 ObjTac 上 BD-Rate 大幅为正（即比锚点更差），印证了"自然图像预训练难泛化到力学触觉"的域不匹配；而 ObjTac 像素呈现屏幕内容般的大块均匀区域，使 VTM-SCC/HM-SCC 在该数据集上特别有效（约 −44%）。

### 下游任务实验

**分类（top-1 acc，压缩到极低码率仍接近未压缩）**：

| 数据集 | 分类器 | 未压缩 | TaCo-L | TaCo-L 码率 |
|--------|--------|--------|--------|-------------|
| Touch and Go | SVM | 76.63% | 75.12% | 0.193 bpp (124×) |
| ObjectFolder | SVM | 44.11% | 43.08% | 0.453 bpp (53×) |
| YCB-Slide | SVM | 98.75% | 98.01% | 0.126 bpp (190×) |

**灵巧抓取（Isaac Sim，100 物体，成功率）**：在码率仅 0.0251 bpp 下，TaCo-L 抬升成功率 $s_{lift}$ 62.2%（未压缩 63.8%，仅掉 1.6%），抗扰动成功率 $s_{disturb}$ 59.9%（未压缩 61.7%）。

| 编解码器 | BPP | $s_{lift}$ Avg | $s_{disturb}$ Avg |
|---------|-----|--------------|------------------|
| 未压缩 | 24 | 63.8% | 61.7% |
| JPEG-XL | 0.0505 | 55.0% (−8.8%) | 53.2% (−8.5%) |
| VTM-Intra | 0.0498 | 63.1% (−0.7%) | 61.3% (−0.4%) |
| LALIC | 0.0397 | 60.0% (−3.8%) | 58.2% (−3.5%) |
| TaCo-L | **0.0251** | 62.2% (−1.6%) | 59.9% (−1.8%) |

### 关键发现
- **域内训练是触觉压缩的胜负手**：把 DualComp-I/LALIC 拿到触觉数据上重训（TaCo-LL/TaCo-L），就在无损和有损上全面超过其预训练版本和所有现成编解码器；预训练神经编解码器一遇到力学数据 ObjTac 就泛化失败（BD-Rate 转正）。
- **物理触觉压缩率有天花板**：现实物理触觉数据最高约 22×（ObjTac），而仿真触觉信号稀疏，抓取实验中压缩率可达 1000×——说明压缩潜力强依赖数据本身的冗余度。
- **抓取对压缩比分类更敏感**：分类任务在 124×~190× 压缩下几乎不掉点，但抓取这种闭环控制任务即便用最好的 TaCo-L 也会掉约 1.6~1.8%，VTM-Intra 反而最稳（但码率是 TaCo-L 的近 2 倍）——压缩效率与控制鲁棒性之间存在真实权衡。
- **复杂度可控**：TaCo-LL 比 P2LLM/Llama3-8B/RWKV-7B 用更少参数（12M~96M）拿到更好压缩，编解码速度 317~614 KB/s；TaCo-L 编解码 FPS 在不同分辨率下为 4~48 FPS。

## 亮点与洞察
- **"力向量→RGB 像素"的映射很巧**：把 3D 力向量当颜色通道、沿时间堆成 $T\times60$ 图像，让力学触觉这种非视觉模态也能直接复用成熟图像/视频编解码器，是统一异构数据的关键 trick，可迁移到任何低维多通道时序传感器（IMU、肌电等）的压缩。
- **benchmark 把"为谁压缩"显式化**：无损/人眼/机器/机器人四类任务对应四种失真容忍度，揭示了"分类抗压、抓取怕压"的非对称结论——这对实际部署选码率比单看 PSNR 有用得多。
- **最让人"啊哈"的是泛化失败的对照**：同一个 LALIC，预训练版在 ObjTac 上 BD-Rate +32.8%，重训成 TaCo-L 后变 −27.0%，把"触觉感知压缩到底值不值得做"用一组数据钉死了。

## 局限与展望
- **TaCo-LL/TaCo-L 只是把两个现成 SOTA 重训**，并未提出触觉特有的编码结构（如显式建模力的物理先验或多接触点空间拓扑），上界探索仍偏保守。
- **跨数据集泛化未充分拆解**：训练只用了 Touch and Go + ObjectFolder 两个视觉触觉数据集，力学数据 ObjTac 完全没进训练集，TaCo 在 ObjTac 上的好成绩有多少来自架构、多少来自数据相似性，文中没有进一步消融。
- **抓取实验在仿真里完成**（Isaac Sim），其触觉信号比真实更稀疏导致压缩率虚高（达 1000×），真实灵巧手闭环下的压缩—控制权衡还需实机验证。
- 改进方向：把时序帧间冗余（inter-frame）纳入数据驱动编解码器（目前 TaCo-L 是 intra-frame），以及为力学触觉设计专门的网络分支。

## 相关工作与启发
- **vs 经典触觉压缩（稀疏线性预测 / 小波稀疏化 / 比特截断+delta 编码）**：他们针对窄场景做信号稀疏或量化，缺乏严格压缩指标也难泛化；TaCo 用统一图像化表征 + 标准率失真指标，把触觉压缩纳入主流编解码评测体系。
- **vs 预训练神经编解码器（DLPR / DualComp-I / LALIC / DCVC 系列）**：它们在自然图像/视频上训练，迁移到触觉（尤其力学触觉）因域不匹配而退化；TaCo-LL/TaCo-L 证明在触觉数据上重训能显著翻盘。
- **vs LLM 压缩器（LMIC，用 RWKV-7B/Llama3-8B）**：把语言模型当通用压缩器虽通用，但 7B~8B 参数下速度极慢（20~86 KB/s）且压缩率不及小得多的 TaCo-LL，说明触觉这种结构化数据更吃专用小模型而非通用大模型。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个触觉数据编解码 benchmark + 首批纯触觉训练的无损/有损编解码器，填补明确空白；但 TaCo-LL/TaCo-L 本质是现成模型重训。
- 实验充分度: ⭐⭐⭐⭐⭐ 5 数据集 × 30 编解码器 × 4 任务，含分类、仿真抓取等下游验证，覆盖面很全。
- 写作质量: ⭐⭐⭐⭐ 动机清晰、表格详实，benchmark 结论一目了然。
- 价值: ⭐⭐⭐⭐⭐ 为触觉数据压缩这一被忽视方向立了标准、数据集和 baseline，部署导向的"压缩—任务"权衡结论实用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] AnyTouch 2: General Optical Tactile Representation Learning For Dynamic Tactile Perception](anytouch_2_general_optical_tactile_representation_learning_for_dynamic_tactile_p.md)
- [\[ICLR 2026\] Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets](cross-embodiment_offline_reinforcement_learning_for_heterogeneous_robot_datasets.md)
- [\[ICLR 2026\] Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning](memory_benchmark_robots_a_benchmark_for_solving_complex_tasks_with_reinforcement.md)
- [\[ICLR 2026\] DexMove: Learning Tactile-Guided Non-Prehensile Manipulation with Dexterous Hands](dexmove_learning_tactile-guided_non-prehensile_manipulation_with_dexterous_hands.md)
- [\[ICLR 2026\] RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification](rf-matid_dataset_and_benchmark_for_radio_frequency_material_identification.md)

</div>

<!-- RELATED:END -->
