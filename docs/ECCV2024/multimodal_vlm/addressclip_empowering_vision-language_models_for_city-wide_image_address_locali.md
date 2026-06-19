---
title: >-
  [论文解读] AddressCLIP: Empowering Vision-Language Models for City-wide Image Address Localization
description: >-
  [ECCV 2024][多模态VLM][图像地址定位] 提出 AddressCLIP 框架，通过图像-文本对齐（地址+场景描述的对比学习）和图像-地理匹配（基于GPS距离的流形学习）两大核心组件，将图像地址定位（IAL）问题建模为端到端的视觉-语言对齐任务，在自建的三个IAL数据集上取得最高85.92%的Top-1准确率。
tags:
  - "ECCV 2024"
  - "多模态VLM"
  - "图像地址定位"
  - "CLIP"
  - "对比学习"
  - "流形学习"
  - "视觉-语言模型"
---

# AddressCLIP: Empowering Vision-Language Models for City-wide Image Address Localization

**会议**: ECCV 2024  
**arXiv**: [2407.08156](https://arxiv.org/abs/2407.08156)  
**代码**: [GitHub](https://github.com/xsx1001/AddressCLIP)  
**领域**: 多模态VLM  
**关键词**: 图像地址定位, CLIP, 对比学习, 流形学习, 视觉-语言模型

## 一句话总结

提出 AddressCLIP 框架，通过图像-文本对齐（地址+场景描述的对比学习）和图像-地理匹配（基于GPS距离的流形学习）两大核心组件，将图像地址定位（IAL）问题建模为端到端的视觉-语言对齐任务，在自建的三个IAL数据集上取得最高85.92%的Top-1准确率。

## 研究背景与动机

传统的图像地址定位依赖两阶段流程：先通过图像地理定位预测GPS坐标，再通过逆地理编码（Reverse Geocoding）转换为可读地址。这种方式存在三个核心问题：

**GPS坐标缺乏语义性**：数字坐标对人类不可读，无法直接用于推荐、导航等下游任务

**GPS到地址的转换存在歧义**：街道交叉口处的图像可能被归属到不同街道，转换结果不唯一

**流程非端到端**：需要构建检索数据库，存储和检索开销大，且误差会在两阶段间累积

作者提出了一个全新的问题定义——**图像地址定位（Image Address Localization, IAL）**，目标是直接从图像预测出人类可读的文本地址（如"Forbes Avenue & Craig Street, Oakland"），无需中间GPS坐标，也不需要检索数据库。

这一问题的实际应用场景包括：社交媒体地点标注、新闻图片地点验证、旅游平台位置推荐等。

## 方法详解

### 整体框架

AddressCLIP 基于预训练 CLIP 模型，将 IAL 建模为视觉-文本对齐问题。框架包含三个核心损失函数：

- **图像-地址对比损失** $\mathcal{L}_{address}$：学习图像与地址文本的对齐
- **图像-描述对比损失** $\mathcal{L}_{caption}$：利用场景描述文本补充地址信息的不足
- **图像-地理匹配损失** $\mathcal{L}_{geography}$：利用GPS坐标约束图像特征的空间分布

推理时，将查询图像的嵌入与所有候选地址的文本嵌入计算相似度，选择最相似的地址作为预测结果。

### 语义地址分割策略

原始行政地址存在两个问题：（1）街道长度不一致，长街道导致地址定位粒度过粗；（2）交叉口处的地址存在歧义。

作者设计了**语义地址分割策略（Semantic Address Partition）**：
- 识别每条主街道的所有交叉街道
- 在交叉点处将主街道分割为子街段
- 移除过近的交叉点避免子街段过短
- 合并过短的子街段（<5个位置点）缓解长尾分布

最终地址表示形式为："主街道名 + 交叉街道名（1-2条）+ 社区名"，例如 "Forbes Ave & Craig St, Oakland"。

### 图像-文本对齐（Image-Text Alignment）

**问题**：地址文本过于简洁，无法提供环境、地标等上下文信息。

**解决方案**：引入BLIP生成的场景描述文本作为地址的补充。场景描述以"A street view of"为prompt生成，包含建筑物、街道标志等视觉细节。

图像特征 $V_i = \mathcal{V}(I_i)$，地址特征 $T_i^A = \mathcal{T}(A_i)$，描述特征 $T_i^C = \mathcal{T}(C_i + A_i)$（描述文本后追加地址信息）。

**图像-地址对比损失**：

$$\mathcal{L}_{address} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{\exp(V_i \cdot T_i^A / \tau)}{\sum_{j=1}^{N}\exp(V_i \cdot T_j^A / \tau)} + \log\frac{\exp(T_i^A \cdot V_i / \tau)}{\sum_{k=1}^{N}\exp(T_i^A \cdot V_k / \tau)}\right]$$

**图像-描述对比损失** $\mathcal{L}_{caption}$ 形式完全类似，将 $T^A$ 替换为 $T^C$。

关键设计：场景描述仅用于训练，推理时只需地址文本编码器。实验验证描述文本中追加地址信息（$C_i + A_i$）优于单独描述（$C_i$），因为地址信息提供了明确的地理锚点。

### 图像-地理匹配（Image-Geography Matching）

**动机**：城市中的地址文本可能地理位置相距很远但文字高度相似（如不同区域的同名街道），或地理位置很近但文字差异大。单靠文本对齐难以捕捉这种空间关系。

**核心思想**：从流形学习角度出发，图像嵌入空间应与地理坐标空间保持一致——地理邻近的图像在特征空间中也应接近。

**地理距离矩阵**：对batch内所有图像的UTM坐标做min-max归一化后计算曼哈顿距离：

$$D_{ij}^U = \|\hat{U}_i - \hat{U}_j\|_1, \quad \hat{U}_i = \frac{U_i - \min(U_i)}{\max(U_i) - \min(U_i)}$$

**特征相似度矩阵**：

$$D_{ij}^V = \frac{V_i \cdot V_j}{\|V_i\| \cdot \|V_j\|}$$

**图像-地理匹配损失**：

$$\mathcal{L}_{geography} = \frac{1}{N^2}\sum_{i=1}^{N}\sum_{j=1}^{N}(D_{ij}^V - D_{ij}^U)^2$$

该损失使特征相似度矩阵逼近地理距离矩阵，从而在特征空间中保持地理空间的拓扑结构。

### 损失函数 / 训练策略

**总目标函数**：

$$\mathcal{L}_{total} = \alpha \mathcal{L}_{address} + \beta \mathcal{L}_{caption} + \gamma \mathcal{L}_{geography}$$

其中 $\alpha=1, \beta=0.2, \gamma=0.8$。

**训练细节**：
- 基于 OpenAI CLIP（默认 ViT-B/16）
- 不引入额外参数，直接微调CLIP的图像和文本编码器
- Adam 优化器（$\beta_1=0.9, \beta_2=0.98$），cosine学习率从 2.4e-5 到 2.4e-8
- 输入图像 224×224，batch size 32/GPU，8块 V100 训练 100 epochs
- BLIP-Caption-Large 生成场景描述（prompt: "A street view of"，长度10-30词）

## 实验关键数据

### 数据集

作者基于 Pitts-250K 和 SF-XL 数据集构建了三个 IAL 数据集：

| 数据集 | 规模 | 训练/验证 | 测试 | 图像尺寸 | 覆盖面积 | 子街道数(训/测) |
|---------|------|-----------|------|----------|----------|----------------|
| Pitts-IAL | 6.7GB / 234K | 234K | 19K | 480×640 | 20 km² | 428/327 |
| SF-IAL-Base | 6.8GB / 184K | 184K | 21K | 512×512 | 6 km² | 400/369 |
| SF-IAL-Large | 121GB / 1.96M | 1.96M | 280K | 512×512 | 170 km² | 3616/3406 |

### 主实验

评估指标包括 SSA（子街道级准确率）和 SA（街道级准确率），报告 Top-1 和 Top-5。

| 方法 | Pitts-IAL SSA-1 | Pitts-IAL SA-1 | SF-IAL-Base SSA-1 | SF-IAL-Base SA-1 | SF-IAL-Large SSA-1 | SF-IAL-Large SA-1 |
|------|----------------|----------------|-------------------|-----------------|--------------------|--------------------|
| Zero-shot CLIP | 0.85 | 1.28 | 1.25 | 2.80 | 0.26 | 0.50 |
| CLIP + address | 77.66 | 80.86 | 83.66 | 85.76 | 81.84 | 84.56 |
| CLIP + CoOp | 67.91 | 71.19 | 77.77 | 79.90 | 74.84 | 78.23 |
| CLIP + CoCoOp | 69.04 | 73.28 | 79.19 | 81.15 | 76.92 | 79.85 |
| CLIP + MaPLe | 72.98 | 76.04 | 81.46 | 83.69 | 79.63 | 82.34 |
| **AddressCLIP** | **80.39** | **82.62** | **86.32** | **87.44** | **85.92** | **88.10** |

AddressCLIP 在三个数据集上分别比最佳 prompt learning 方法（MaPLe）提升 **7.41%、4.86%、6.29%**（SSA-1）。

### 消融实验

**关键组件消融**（Pitts-IAL / SF-IAL-Base，SSA-1）：

| $\mathcal{L}_{address}$ | $\mathcal{L}_{caption}$ | $\mathcal{L}_{geography}$ | Pitts-IAL SSA-1 | SF-IAL-Base SSA-1 |
|:---:|:---:|:---:|:---:|:---:|
| ✔ | | | 77.66 | 83.66 |
| | ✔ | | 69.27 | 75.85 |
| ✔ | ✔ | | 79.20 | 84.86 |
| ✔ | | ✔ | 79.27 | 85.54 |
| ✔ | ✔ | ✔ | **80.39** | **86.32** |

- 单独 $\mathcal{L}_{caption}$ 远弱于 $\mathcal{L}_{address}$，说明地址信息的直接监督更关键
- $\mathcal{L}_{caption}$ 和 $\mathcal{L}_{geography}$ 各自贡献约 +1.5-1.9%，组合带来 +2.7% 提升，两者互补

**编码器冻结策略**：仅解冻图像编码器带来约30%增益，仅解冻文本编码器增益有限；两者同时解冻效果最佳。

**与检索方法的对比**（Pitts-IAL，ResNet50）：

| 方法 | 存储 | 推理+检索时间 | SSA-1 |
|------|------|-------------|-------|
| SALAD | 2.34 GB | 2.53 ms | 75.17 |
| AnyLoc | - | - | 74.83 |
| AddressCLIP | **0.34 GB** | **3.46 ms** | **77.01** |

AddressCLIP 无需检索数据库，存储仅 0.34GB，推理内存仅 0.64MB，同时 SSA-1 超出检索方法 1.84%。

### 关键发现

1. **SF-IAL-Base 性能优于 Pitts-IAL**：旧金山街道更规整、街景图像采集更密集
2. **SA 指标高于 SSA**：子街道级标签学习能进一步提升主街道识别能力
3. **在 SF-IAL-Large（170km²）上仍达 85.92%**：证明方法可扩展到大规模城市场景
4. **多城市混合训练**：Pitts+SF 联合训练性能仅下降 <0.8%，展现跨城市扩展潜力
5. **地理覆盖度实验**：仅 12.5% 的位置覆盖度即可保留约 75% 的原始性能

## 亮点与洞察

1. **问题定义创新**：首次将图像地址定位定义为独立问题（IAL），从"预测坐标"转向"预测可读地址"，更贴合人类使用习惯
2. **语义地址分割策略**：通过在交叉口处分割街道创建子街段，既缓解了长尾分布，又消除了交叉口歧义，设计简洁有效
3. **流形学习约束**：图像-地理匹配损失从拓扑保持角度约束特征空间，是对比学习之外的有效补充
4. **端到端优势明显**：对比两阶段流程，AddressCLIP 消除了GPS转地址的累积误差和歧义，同时大幅降低存储开销

## 局限与展望

1. **判别式模型局限**：推理受限于候选地址集合，无法预测训练中未见过的地址
2. **城市级限制**：目前仅在城市级别验证，扩展到更大地理范围（跨城市/国家）需要更多研究
3. **生成式方向探索**：作者尝试用 LLaVA 进行指令微调（LLaVA-IAL），初步展示了生成式模型在 IAL 任务上的可行性
4. **场景描述的边际收益有限**：BLIP-Base 和 BLIP-Large 生成的描述带来的性能差异很小，提示场景描述的改进空间有限

## 相关工作与启发

- **StreetCLIP / GeoCLIP**：利用 CLIP 进行地理定位的先驱工作，但仍输出坐标而非地址
- **CoOp / CoCoOp / MaPLe**：通用 prompt learning 方法在 IAL 任务上效果有限，说明通用迁移不如任务特定设计
- **启发**：流形学习约束可推广到其他需要保持空间结构的视觉-语言任务

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|:-----------:|------|
| 创新性 | 8 | 新问题定义 + 语义地址分割 + 地理流形约束的组合 |
| 技术深度 | 7 | 方法整洁但各组件较直接，无复杂架构设计 |
| 实验充分度 | 9 | 三个数据集、多维度消融、定性可视化、与检索方法对比 |
| 写作质量 | 8 | 结构清晰，问题动机阐述到位 |
| 实用价值 | 8 | 端到端地址预测在社交媒体/导航等场景有直接应用 |
| **总分** | **8.0** | 问题定义新颖，方法有效，实验全面 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Omniview-Tuning: Boosting Viewpoint Invariance of Vision-Language Pre-training Models](omniview-tuning_boosting_viewpoint_invariance_of_vision-language_pre-training_mo.md)
- [\[ECCV 2024\] Towards Real-World Adverse Weather Image Restoration: Enhancing Clearness and Semantics with Vision-Language Models](towards_real-world_adverse_weather_image_restoration_enhancing_clearness_and_sem.md)
- [\[ECCV 2024\] Attention Prompting on Image for Large Vision-Language Models](attention_prompting_on_image_for_large_visionlanguage_models.md)
- [\[CVPR 2026\] Mechanisms of Object Localization in Vision-Language Models](../../CVPR2026/multimodal_vlm/mechanisms_of_object_localization_in_vision-language_models.md)
- [\[ECCV 2024\] Merlin: Empowering Multimodal LLMs with Foresight Minds](merlin_empowering_multimodal_llms_with_foresight_minds.md)

</div>

<!-- RELATED:END -->
