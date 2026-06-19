---
title: >-
  [论文解读] Asynchronous Event Error-Minimizing Noise for Safeguarding Event Dataset
description: >-
  [ICCV 2025][LLM安全][不可学习样本] 提出首个面向异步事件数据的不可学习样本生成方法（UEvs），设计了事件误差最小化噪声（E²MN）及自适应投影机制，使事件数据集在保持合法使用功能的同时阻止未授权模型从中学习。 随着事件相机数据集（如 N-Caltech101、DVS128 Gesture 等）大量公开…
tags:
  - "ICCV 2025"
  - "LLM安全"
  - "不可学习样本"
  - "事件相机"
  - "数据集保护"
  - "误差最小化噪声"
  - "异步事件流"
---

# Asynchronous Event Error-Minimizing Noise for Safeguarding Event Dataset

**会议**: ICCV 2025  
**arXiv**: [2507.05728](https://arxiv.org/abs/2507.05728)  
**代码**: [https://github.com/rfww/uevs](https://github.com/rfww/uevs)  
**领域**: 数据安全 / 事件相机  
**关键词**: 不可学习样本, 事件相机, 数据集保护, 误差最小化噪声, 异步事件流

## 一句话总结

提出首个面向异步事件数据的不可学习样本生成方法（UEvs），设计了事件误差最小化噪声（E²MN）及自适应投影机制，使事件数据集在保持合法使用功能的同时阻止未授权模型从中学习。

## 研究背景与动机

随着事件相机数据集（如 N-Caltech101、DVS128 Gesture 等）大量公开，未授权使用的风险日益突出。传统图像领域的不可学习样本（Unlearnable Examples, UEs）通过在图像中嵌入不可感知噪声来阻止模型学习真实语义特征，但**直接将图像 UEs 应用于事件数据不可行**，原因有三：

**二值极性约束**：事件数据极性只有 $\pm 1$，是高度离散的，传统连续噪声不兼容

**异步时空结构**：事件流是稀疏的时空点云（x, y, t, p），而非规则 2D 网格

**表征转换间隙**：事件需先转为事件栈（event stack）才能输入 DNN，但噪声注入后无法直接从栈重建回事件流

数据污染（如坐标偏移、极性翻转）虽能降低质量，但容易被数据增强抵消，且保护效果不可靠。

## 方法详解

### 整体框架

UEvs 框架流程：事件流 → 事件栈转换 → 代理模型生成 E²MN → 自适应投影稀疏化 → 生成不可学习事件栈 → 检索策略重建不可学习事件流。

### 关键设计

#### 1. 事件栈表征

将事件流按时间分 $C=16$ 个 bin，每个 bin 的像素值取三种之一：
- $0$：极性 $p=-1$ 的事件
- $0.5$：无事件
- $1.0$：极性 $p=+1$ 的事件

大通道数（C=16）避免了单/三通道导致的事件覆盖问题。

#### 2. 事件误差最小化噪声（E²MN）

核心优化目标（min-min 双层优化）：
$$\arg\min_\theta \mathbb{E}_{(\mathcal{E},l) \in \mathcal{D}} [\min_\delta \mathcal{L}(f'_\theta(\mathcal{R}(\mathcal{E}) + \delta), l)] \quad \text{s.t.} \|\delta\|_\infty \leq \epsilon$$

内层优化：找到 $L_\infty$ 约束下使代理模型损失最小的噪声 $\delta$（通过 PGD）
外层优化：更新代理模型参数使分类损失最小

两种噪声形式：
- **Sample-wise 噪声**：逐样本生成，每个事件流有独立噪声，保护效果最强
- **Class-wise 噪声**：同类共享噪声，效率高且可扩展到新数据

外层损失增加相似度正则化：
$$\mathcal{L}^* = \lambda_1 \mathcal{L} + \lambda_2 \mathcal{L}_s$$
$\mathcal{L}_s$ 增大干净特征与不可学习特征的差异，确保投影后噪声仍有效。

#### 3. 自适应投影机制

将连续噪声投影到 $\{-0.5, 0, +0.5\}$，使之与事件栈兼容：
$$\mathbf{P}(\delta) = \begin{cases} -0.5, & \delta_{i,j} < \mu - \tau \times \pi \\ 0.0, & \mu - \tau \times \pi \leq \delta_{i,j} \leq \mu + \tau \times \pi \\ +0.5, & \delta_{i,j} > \mu + \tau \times \pi \end{cases}$$

$\tau$ 控制有效性与隐蔽性的平衡：大 $\tau$ 更隐蔽但保护力弱，小 $\tau$ 保护力强但可见噪声增多。

投影后的噪声嵌入效果（混淆矩阵）：
- +0.5 加到无事件位置 → 生成新事件
- -0.5 加到有事件位置 → 删除事件
- 加到同极性位置 → 保持原样

#### 4. 事件流检索重建

从不可学习事件栈重建回事件流：
- 原有事件：从原始流中检索对应时间戳
- 新增事件：基于所在时间 bin $\Delta_t$ 初始化自适应时间戳

### 损失函数 / 训练策略

- 代理模型：ResNet18（SGD, lr=1e-4, momentum=0.9）
- 噪声生成：PGD 10 步，$\epsilon=0.5$，步长 $\alpha=0.8/255$
- 投影平衡参数 $\tau=3/4$
- 终止条件：代理模型分类准确率 >99%
- 训练限制：代理模型每 epoch 仅训练 $M=10$ 次迭代，防止过度学习真实特征

## 实验关键数据

### 主实验（各 DNN 在不可学习数据集上的测试准确率 %）

| 噪声形式 | 模型 | N-Caltech101 | CIFAR10-DVS | DVS128 Gesture | N-ImageNet |
|----------|------|:---:|:---:|:---:|:---:|
| Clean | RN18 | 78.32 | 65.19 | 74.14 | 56.60 |
| CS (坐标偏移) | RN18 | 50.43 (-27.8) | 46.80 (-18.4) | 75.35 (+1.2) | 42.60 (-14.0) |
| PI (极性翻转) | RN18 | 40.78 (-37.5) | 22.72 (-42.5) | 31.94 (-42.2) | 41.80 (-4.8) |
| **Class-wise** | **RN18** | **1.90 (-76.6)** | **22.33 (-42.9)** | **14.93 (-59.4)** | **10.00 (-46.8)** |
| **Sample-wise** | **RN18** | **0.52 (-77.6)** | **15.51 (-49.6)** | **14.54 (-59.4)** | **10.20 (-46.2)** |
| **Class-wise** | **Swin_B** | **0.52 (-90.2)** | **29.03 (-46.2)** | **28.12 (-55.2)** | **10.00 (-62.0)** |

### 消融实验（N-Caltech101, sample-wise 噪声）

| 设置 | RN18 | RN50 | VG16 | DN121 | ViT_B | Swin_B |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| E1 (完整方法) | 0.52 | 6.09 | 14.13 | 11.49 | 1.09 | 21.14 |
| E2 (无相似度损失) | 3.85 | 15.22 | 38.31 | 10.17 | 37.51 | 27.51 |
| E3 (混合 $\Delta_c \lor \Delta_s$) | 5.17 | 5.40 | 22.23 | 6.15 | 1.21 | 16.66 |
| E5 (加数据增强) | 13.04 | 8.04 | 14.99 | 10.74 | 9.88 | 25.73 |
| E6 (FGSM替代PGD) | 27.74 | 33.08 | 44.28 | 34.92 | 8.85 | 24.41 |

### 关键发现

1. UEvs 在 N-Caltech101 上将 Swin_B 准确率从 **90.70% 降至 0.52%**（class-wise），降幅超 90 个百分点
2. 跨架构泛化性强：仅用 ResNet18 作为代理模型，对 7 种不同架构（含 ViT、Swin）均有效
3. 感知质量良好：PSNR 20.36（class-wise）/ 18.22（sample-wise），SSIM 0.791 / 0.571
4. 相似度损失 $\mathcal{L}_s$ 对投影后保持有效性至关重要（去掉后 VG16 从 14.13% 回升至 38.31%）
5. 对常见数据增强（随机裁剪/翻转/EventDrop）有一定鲁棒性

## 亮点与洞察

- **填补空白**：首次将不可学习样本概念扩展到异步事件数据，解决了事件数据的独特挑战（二值极性、稀疏性、时间戳重建）
- **投影机制的精巧设计**：将连续噪声映射为三值（-0.5, 0, +0.5），恰好对应事件的删除、无操作、新增——与事件物理模型完美对应
- **混合噪声策略**（$\Delta_c \lor \Delta_s$）提供了灵活的实际应用方案

## 局限与展望

- 仅验证了分类任务，对检测、分割等其他事件视觉任务的迁移性未探索
- Sample-wise 噪声随数据集规模增大内存消耗急剧增加
- 缺乏对抗性防御（adversarial training）下的鲁棒性深入评估
- 投影参数 $\tau$ 的自适应选择机制有待开发
- 事件流重建的时间戳质量可能影响下游时序任务

## 相关工作与启发

- 继承了 Huang et al. (ICLR 2021) 的 UEs 核心思想，但解决了事件数据的特殊挑战
- 与 EventTrojan 等事件后门攻击工作形成攻防对偶
- 启发未来方向：基于生成模型（如噪声生成器）提升 sample-wise 噪声的生成效率

## 评分

- 新颖性: ⭐⭐⭐⭐ （首开事件数据保护先河）
- 实验充分度: ⭐⭐⭐⭐ （4 数据集、7 模型、多种基线和消融）
- 写作质量: ⭐⭐⭐⭐ （Pipeline 清晰，但数学符号偶有不一致）
- 价值: ⭐⭐⭐⭐ （为事件数据安全开辟新研究方向）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] SERE: Structural Example Retrieval for Enhancing LLMs in Event Causality Identification](../../ACL2026/llm_safety/sere_structural_example_retrieval_for_enhancing_llms_in_event_causality_identifi.md)
- [\[NeurIPS 2025\] MaskSQL: Safeguarding Privacy for LLM-Based Text-to-SQL via Abstraction](../../NeurIPS2025/llm_safety/masksql_safeguarding_privacy_for_llm-based_text-to-sql_via_abstraction.md)
- [\[NeurIPS 2025\] CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming](../../NeurIPS2025/llm_safety/cpret_a_dataset_benchmark_and_model_for_retrieval_in_competitive_programming.md)
- [\[ICCV 2025\] Temporal Unlearnable Examples: Preventing Personal Video Data from Unauthorized Exploitation](temporal_unlearnable_examples_preventing_personal_video_data_from_unauthorized_e.md)
- [\[NeurIPS 2025\] Enhancing Sample Selection Against Label Noise by Cutting Mislabeled Easy Examples](../../NeurIPS2025/llm_safety/enhancing_sample_selection_against_label_noise_by_cutting_mislabeled_easy_exampl.md)

</div>

<!-- RELATED:END -->
