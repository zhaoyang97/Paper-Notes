---
title: >-
  [论文解读] PRISM: Progressive Robust Learning for Open-World Continual Category Discovery
description: >-
  [ICLR 2026][自监督学习][持续类别发现] PRISM 提出"开放世界持续类别发现"（OW-CCD）这一更现实的设定——数据流里既冒出新类别又夹带域偏移，并用"高频分流 + 稀疏最优传输匹配 + 不变知识迁移"三件套，在 SSB-C 与 DomainNet 上稳定刷新 CCD SOTA（CUB-C 干净域涨 15.1%）。
tags:
  - "ICLR 2026"
  - "自监督学习"
  - "持续类别发现"
  - "开放世界"
  - "域偏移"
  - "高频谱"
  - "最优传输"
---

# PRISM: Progressive Robust Learning for Open-World Continual Category Discovery

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=5JwUWsewWH](https://openreview.net/forum?id=5JwUWsewWH)  
**领域**: 自监督 / 类别发现  
**关键词**: 持续类别发现, 开放世界, 域偏移, 高频谱, 最优传输

## 一句话总结
PRISM 提出"开放世界持续类别发现"（OW-CCD）这一更现实的设定——数据流里既冒出新类别又夹带域偏移，并用"高频分流 + 稀疏最优传输匹配 + 不变知识迁移"三件套，在 SSB-C 与 DomainNet 上稳定刷新 CCD SOTA（CUB-C 干净域涨 15.1%）。

## 研究背景与动机
**领域现状**：类别发现（Category Discovery）想用已标注已知类的知识，去自动发现无标注数据里的新概念。从 NCD（无标注全是新类）到 GCD（无标注里已知+未知混合），再到 CCD（Continual Category Discovery，把持续学习接进来），任务越来越贴近真实——模型要在不断到来的无标注数据流里持续发现新类，同时不遗忘旧类。

**现有痛点**：但现有 CCD 方法几乎都默认一个隐含假设——**每个阶段的数据来自同一个固定域、整条流是平稳分布**。这在开放环境里几乎不成立：一个在线平台持续接收来自不同相机、不同用户、不同风格/光照的动物图片，域在变的同时还会冒出稀有物种。一旦域偏移真实存在，现有 CCD 既保不住已知类的识别能力，也难以可靠地发现新类。

**核心矛盾**：直接搬域适应（DA）来对齐分布并不奏效。传统 DA 假设源/目标标签空间重叠，而这里目标流里有大量未知新类；**朴素对齐会引发负迁移，把新类信号也一起抹平**，反而抑制了新类发现。而且多数 DA 只盯着对齐已知类，对"如何探索未知标签空间"几乎不给指导。

**切入角度**：作者从**频谱分析**切入观察到一个有用现象——图像的**高频分量**更多承载域不变的全局语义（结构、形状），**低频分量**则编码域相关的风格细节（色调、纹理）。于是"用什么特征判断已知/未知"这件事有了抓手：高频特征对域偏移更鲁棒，更适合把已知类和未知类分开。

**核心 idea**：用**分而治之**取代"先对齐再发现"——先用高频信息把数据流分成已知/未知两堆，已知堆用稀疏最优传输打可靠伪标签，未知堆则通过"跨域保持类别关系排序不变"的方式从已知类迁移语义知识，从而在域偏移下稳定地发现新类。

## 方法详解

### 整体框架
PRISM 要解决的是：**一个夹带域偏移的无标注数据流，逐阶段到来，里面既有旧已知类、也有新未知类，模型要在线地既认对旧类又发现新类**。整体是"一次 base 预训练 + T 轮在线发现"的范式。Base 阶段在有标注已知类数据上用交叉熵预训练特征提取器 $f$ 和分类头 $g$，给后续发现打底。进入每一轮在线发现，新到的无标注流 $D^u_t$ 依次经过三个模块串行处理：先用 **HCS（高频分流）** 把流切成"像已知"和"像未知"两个子集；已知子集交给 **SAM（稀疏分配匹配）** 用近端最优传输打伪标签；未知子集交给 **IKT（不变知识迁移）** 在跨域扰动下保持其与已知类原型的关系排序一致，再配合 Affinity Propagation 自动聚类出新类簇；最后把"已知伪标签 + 新类簇"合并，用交叉熵增量更新模型并动态扩展在线分类器。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["无标注数据流 D^u_t<br/>(旧已知类 + 新未知类 + 域偏移)"] --> B["高频分流 HCS<br/>取高频特征→密度打分→GMM 二分"]
    B -->|π(x)≥0.5 已知子集| C["稀疏分配匹配 SAM<br/>近端最优传输打可靠伪标签"]
    B -->|π(x)<0.5 未知子集| D["不变知识迁移 IKT<br/>跨域排序一致→AP 聚类出新类簇"]
    C --> E["合并伪标签 + 新类簇<br/>交叉熵增量更新 + 扩展在线分类器"]
    D --> E
    E --> F["输出：认对旧类 + 发现新类"]
```

### 关键设计

**1. HCS 高频分流：用域不变的高频信息把已知/未知分开**

这一步针对的痛点是"域偏移下怎么判断一个无标注样本是旧类还是新类"。作者不直接对齐分布（会负迁移），而是先做频域分解：对输入图像 $x_i$ 做离散傅里叶变换 $F(x_i)$，用一个半径为 $r$ 的方形二值掩码 $M$ 把频谱分成低频 $F_l = M \odot F(x_i)$ 和高频 $F_h = (I-M) \odot F(x_i)$，再逆变换回空间域得到高频图像 $x_i^h$。只把高频图喂进预训练的 $f$ 得到高频表示，定义一个密度打分

$$S(x) = \nu\!\left(\max_c \frac{f(x^h)\cdot e_c}{\|f(x^h)\|\,\|e_c\|}\right),$$

其中 $e_c$ 是上一阶段已知类 $c$ 的原型，$\nu(\cdot)$ 把分数 min–max 归一到 $[0,1]$。直觉是：$S(x)$ 越大说明该样本高频表示越贴近某个已知原型、越可能是已知类，越小越可能是未知类。实测 $S(x)$ 呈**双峰分布**，于是用两分量高斯混合模型 $P(x)=\pi(x)\,N(x|\mu_{kno},\sigma^2_{kno})+(1-\pi(x))\,N(x|\mu_{unk},\sigma^2_{unk})$ 拟合，EM 估出"属于已知分量"的后验 $\pi(x)$，再以 $\pi(x)\ge 0.5$ 为界把流切成已知子集 $D^u_{t,kno}$ 和未知子集 $D^u_{t,unk}$。之所以有效，是因为高频承载的是结构性、域不变的语义，比原图或熵/能量分数受风格干扰更小——消融里 origin image / entropy / energy 三种分流都明显逊于 HCS。

**2. SAM 稀疏分配匹配：用近端最优传输给已知类打稀疏可靠的伪标签**

已知子集 $x_{kno}$ 和上一阶段的已知原型共享语义空间，可以用最优传输（OT）自动找样本—原型对应关系来缓解域差异。但直接线性规划解 OT 代价高，加熵正则虽然提速却会得到**过于稠密、模糊的传输计划**，导致分配不准。SAM 的改法是把熵正则换成 $\ell_2$ 近端项：

$$\min_{\gamma\in\Delta}\ \sum_{i}\sum_{j}\Big[\gamma_{ij}C_{ij}+\tfrac{\varepsilon}{2}\big(\gamma_{ij}-\gamma^{(l)}_{ij}\big)^2\Big],$$

其中代价 $C_{ij}=-\log\big(g(f(x_{i,kno}))_j\big)$（越小越可能属于第 $j$ 类），近端项 $\tfrac{\varepsilon}{2}\sum(\gamma_{ij}-\gamma^{(l)}_{ij})^2$ 抑制迭代间震荡、逼出稀疏稳定解，可行集 $\Delta$ 约束行/列边际。作者再取其 Fenchel–Legendre 对偶把约束问题转成对 $(\psi,\phi)$ 的无约束优化（更高效），传输计划有闭式更新 $\gamma^{(l+1)}_{ij}=\max\{0,(\psi_i+\phi_j+\varepsilon\gamma^{(l)}_{ij}-C_{ij})/\varepsilon\}$。相比标准 $\ell_2$-正则 OT，SAM 的分配模式更稀疏更干净（论文 Fig.3），伪标签更可信。

**3. IKT 不变知识迁移：把"已知—未知类关系排序"做成跨域不变量来稳健发现新类**

类别发现的核心是"靠语义关联从已知类把知识迁到未知类"，但域偏移会让这种关联被风格因子扭曲，模型学到的是虚假关系。IKT 的思路是只保留**域不变的类别关系**。具体做法：先在历史阶段统计低频谱的通道级均值/方差 $\mu(F_l),\sigma(F_l)$ 及其跨样本方差，对一个未知样本，从这些高斯分布里采样扰动统计量 $\hat\mu,\hat\sigma$，**只替换它的低频统计（即换风格）、保留原高频**，重建出风格迁移后的样本 $\hat x^t_{i,unk}$。然后对原视图和风格视图分别提特征，把它们与各已知原型 $e^{t-1}_c$ 的余弦相似度转成 Plackett–Luce（PL）模型的强度参数 $\kappa_{i,c}=\exp(\cos(z,e_c))$。由于枚举 $C_{t-1}!$ 种排列不可行，采用 ListMLE 式的**因子化 PL 似然**（顺序归一化，解析可算）：

$$P(\xi\mid\kappa_i)=\prod_{k=1}^{C_{t-1}}\frac{\kappa_{i,\xi(k)}}{\sum_{k'=k}^{C_{t-1}}\kappa_{i,\xi(k')}}.$$

最后用 KL 散度把原视图和风格视图的 PL 似然对齐，强制排序跨域一致：$L_{rank}=\frac{1}{N_{t,unk}}\sum_i \ell_{KL}\big(P(\cdot|\kappa_i),P(\cdot|\hat\kappa_i)\big)$。这样模型保住的是"未知样本相对各已知原型的全局相对排序"——语义更近的类贡献更强迁移（如 CUB 里 Indigo Bunting 与 Lazuli Bunting 即便视觉细微差异也共享高层语义），而非被风格带偏的虚假关联，从而在域偏移下稳定迁移、可靠发现新类。

### 损失函数 / 训练策略
在线阶段对已知样本用 SAM 伪标签、对未知样本用 Affinity Propagation（非参数聚类，无需预设新类数，适合开放世界）自动聚簇并扩展分类器，合并后用交叉熵增量更新、不回看历史数据。总目标为 $L_{total}=L_{ce}+\lambda_1 L_{rank}$。骨干为 DINO 预训练 ViT-B/16，每阶段只微调最后一个 transformer block，SGD 训 30 epoch，$\lambda_1=1$、阶段数 $T=3$、掩码比 $r=0.3$、近端强度 $\varepsilon=0.5$。

## 实验关键数据

### 主实验
两大基准：SSB-C（语义偏移基准 + 9 种损坏×5 级严重度，三个细粒度数据集）与 DomainNet（6 域大规模、域间差距大）。评测主指标为持续聚类精度 cACC（各阶段 ACC 的累计平均），按 All/Old/New 拆分，报告已知域与未知域两侧。

| 基准 / 任务 | 指标(All) | PRISM | 最强基线 | 提升 |
|--------|------|------|----------|------|
| CUB-C 干净域 | cACC | 49.3 | 34.2 (VB-CGCD) | +15.1 |
| CUB-C 损坏域 | cACC | 44.0 | 31.7 (VB-CGCD) | +12.3 |
| DomainNet Real→Painting (Real) | cACC | 60.9 | 57.3 (VB-CGCD) | +3.6 |
| DomainNet Real→Painting (Painting) | cACC | 39.2 | 32.4 (VB-CGCD) | +6.8 |

PRISM 在所有任务上对 CCD（G&M / Happy / PA-CGCD / DEAN / PromptCCD / VB-CGCD）和重实现的 GCD（GCD / SimGCD / SPTNet / RLCD）均显著领先，尤其在域偏移更剧烈的未知域提升更大。

### 消融实验
组件级消融（Real→Painting，报告 All）：

| HCS | SAM | IKT | Real(All) | Painting(All) | 说明 |
|------|------|------|------|------|------|
| ✗ | ✗ | ✗ | 54.6 | 28.7 | baseline，域偏移下崩 |
| ✓ | ✓ | ✗ | 58.1 | 35.0 | 加 HCS+SAM，已知类大涨 |
| ✓ | ✗ | ✓ | 56.9 | 33.2 | 加 IKT，新类发现更好 |
| ✓ | ✓ | ✓ | 60.9 | 39.2 | 完整模型最佳 |

分流策略对比（Real→Painting）：origin image 55.0/29.6、entropy 54.4/29.9、energy 55.8/30.6，而 HCS 达 60.9/39.2，验证高频分流优于熵/能量/原图。

### 关键发现
- HCS+SAM 主要拉高**已知类**精度（保住识别能力），IKT 主要提升**新类**发现（New 列从 49.9 → 55.1），两类模块分工互补，缺一即掉点。
- 分流环节用什么特征至关重要：高频信息比原图/熵/能量都更能滤掉风格噪声、保留语义结构，是性能的根基。
- 域偏移越大的未知域（如 Painting/Sketch/Infograph）相对基线的相对增益越明显，说明 PRISM 的鲁棒性确实来自对域不变信息的利用。

## 亮点与洞察
- **频谱视角做分而治之**：把"高频域不变、低频域相关"这一观察同时用在两处——HCS 用高频判已知/未知、IKT 用扰动低频做风格增强，一个 insight 贯穿全方法，干净利落。
- **把类别关系做成"排序不变量"**：IKT 不去对齐绝对特征，而是用 Plackett–Luce 因子化似然保持"未知样本对各已知原型的相对排序"跨域一致，巧妙绕开了"枚举全排列不可行"和"绝对对齐引发负迁移"两个坑，可迁移到任何需要跨域保结构的场景。
- **稀疏 OT 替代熵正则**：用 $\ell_2$ 近端项 + 对偶无约束求解，得到比 Sinkhorn 更稀疏可靠的伪标签，是个可复用的"想要稀疏分配时别用熵正则"的实用 trick。

## 局限与展望
- 方法依赖"高频=域不变"的经验性假设，对那些域差异本身体现在结构/高频上的场景（如医学模态差异、强几何形变）是否仍成立，论文未深入讨论。
- 流程串行且环节多（DFT 分解、GMM-EM、近端 OT 迭代、双视图 PL 排序、AP 聚类），在线每阶段开销不小，论文未给出与轻量基线的效率/时延对比。
- 阶段数仅设 $T=3$、掩码比 $r$ 与近端 $\varepsilon$ 为固定超参，更长流（如 $T\gg3$）下误差累积与超参敏感性如何，留待验证。

## 相关工作与启发
- **vs GCD / SimGCD**：它们在静态数据集上同时拿到标注与无标注、且默认单一平稳分布；PRISM 面向流式 + 域偏移的 OW-CCD，并显式处理域不变性，故在损坏/跨域设定下领先一大截。
- **vs DEAN / PromptCCD / VB-CGCD（CCD 基线）**：这些方法虽做持续发现，但默认单域流；PRISM 的核心增量正是把"域偏移"显式纳入，用高频分流 + 不变知识迁移补上它们缺失的鲁棒性。
- **vs 域适应 / 开集域适应（OSDA）**：传统 DA 假设标签空间重叠、朴素对齐会负迁移并抹平新类；PRISM 改"对齐特征"为"保持类别关系排序不变"，既迁移知识又不压制新类发现。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次提出 OW-CCD 设定，并用统一的频谱视角串起三个互补模块
- 实验充分度: ⭐⭐⭐⭐ SSB-C + DomainNet 双基准、对比充分、消融到位，但缺效率/长流分析
- 写作质量: ⭐⭐⭐⭐ 动机清晰、公式完整，方法环节偏多读起来略密
- 价值: ⭐⭐⭐⭐ 把持续类别发现推进到更现实的开放世界设定，方法组件有复用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Beyond the Static World: Continual Category Discovery under Visual Drift](../../CVPR2026/self_supervised/beyond_the_static_world_continual_category_discovery_under_visual_drift.md)
- [\[CVPR 2026\] Decouple Your Discovery and Memory in Continual Generalized Category Discovery](../../CVPR2026/self_supervised/decouple_your_discovery_and_memory_in_continual_generalized_category_discovery.md)
- [\[ECCV 2024\] PromptCCD: Learning Gaussian Mixture Prompt Pool for Continual Category Discovery](../../ECCV2024/self_supervised/promptccd_learning_gaussian_mixture_prompt_pool_for_continual_category_discovery.md)
- [\[CVPR 2026\] Seeing Through the Shift: Causality-Inspired Robust Generalized Category Discovery](../../CVPR2026/self_supervised/seeing_through_the_shift_causality-inspired_robust_generalized_category_discover.md)
- [\[AAAI 2026\] GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery](../../AAAI2026/self_supervised/goal_geometrically_optimal_alignment_for_continual_generalized_category_discover.md)

</div>

<!-- RELATED:END -->
