---
title: >-
  [论文解读] SOS: Segment Object System for Open-World Instance Segmentation With Object Priors
description: >-
  [ECCV 2024][语义分割][图像分割] 提出 SOS 方法，通过用 DINO 自注意力图作为物体先验生成聚焦于物体的 SAM 提示点，从而产出高质量伪标注来训练标准实例分割系统，在 COCO/LVIS/ADE20k 跨类别/跨数据集设置下大幅超越 SOTA，精度提升高达 81.6%。 领域现状：开放世界实例分割（Op…
tags:
  - "ECCV 2024"
  - "语义分割"
  - "图像分割"
  - "pseudo annotation"
  - "提示学习"
  - "object prior"
  - "注意力机制"
---

# SOS: Segment Object System for Open-World Instance Segmentation With Object Priors

**会议**: ECCV 2024  
**arXiv**: [2409.14627](https://arxiv.org/abs/2409.14627)  
**代码**: [https://github.com/chwilms/SOS](https://github.com/chwilms/SOS)  
**领域**: 分割  
**关键词**: open-world instance segmentation, pseudo annotation, SAM prompting, object prior, DINO self-attention

## 一句话总结

提出 SOS 方法，通过用 DINO 自注意力图作为物体先验生成聚焦于物体的 SAM 提示点，从而产出高质量伪标注来训练标准实例分割系统，在 COCO/LVIS/ADE20k 跨类别/跨数据集设置下大幅超越 SOTA，精度提升高达 81.6%。

## 研究背景与动机

**领域现状**：开放世界实例分割（Open-World Instance Segmentation, OWIS）要求系统在仅学习少量已知类别的基础上，能够分割图像中任意未知类别的物体实例。这一任务打破了标准实例分割的封闭世界假设，在真实场景中具有重要价值——譬如训练时没有见过冲浪板或网球拍，测试时仍需要检测它们。

**现有痛点**：现有 OWIS 方法主要分两条路线：

**定位分数替代分类**（如 OLN、SWORD、OpenInst）：用 IoU 等定位质量分数替代前景/背景二分类，避免将未知物体归为背景。但这类方法精度（precision）普遍偏低。

**伪标注增强**（如 GGN、UDOS、LDET）：为未标注物体生成伪标注来扩充训练集。但现有伪标注方法生成的标注**噪声极大**——经常包含大量背景区域，导致精度低下。

**核心矛盾**：提升召回率（检测更多未知物体）与保持精度（不把背景误判为物体）之间存在根本矛盾。现有方法普遍在召回率上有所改善，但精度很低。而简单地使用 SAM "segment anything" 也不行，因为 SAM 会同时分割物体和非物体区域（如天空、地面等 stuff 区域）。

**本文切入角度**：关键在于如何让 SAM 只关注物体而非任意区域。通过系统研究各种"物体先验"（object prior）来生成聚焦于物体的提示点，引导 SAM 只生成高质量的物体分割，从根本上提升伪标注质量。

**核心 idea**：用自监督 ViT（DINO）的自注意力图作为物体位置先验来采样提示点，将 SAM 从 "Segment Anything" 转变为 "Segment Objects"，再用生成的高质量伪标注训练标准实例分割系统。

## 方法详解

### 整体框架

SOS 由三个模块串联组成，形成"先验定位 → 伪标注生成 → 实例分割训练"的流水线：
1. **OLM（Object Localization Module）**：从物体先验中采样物体聚焦的点提示
2. **PAC（Pseudo Annotation Creator）**：用采样的点提示引导 SAM 生成分割，经后处理得到伪标注
3. **实例分割训练**：将伪标注与原始标注合并，训练标准 Mask R-CNN

测试时只需运行训练好的 Mask R-CNN，无需 DINO 或 SAM。

### 关键设计

#### 1. 物体定位模块（OLM）

- **功能**：从输入图像中粗略定位所有物体，输出一组聚焦于物体的点坐标作为 SAM 的提示。
- **核心思路**：
    - 使用 DINO（自监督 ViT-S）最后一层的 6 个自注意力头的注意力图，通过逐元素取最大值聚合为一张场景布局图
    - 将该图归一化为概率质量函数（最小值归零、总和归一），作为物体先验 $P(x,y)$
    - 从物体先验中随机采样 $S=50$ 个点坐标，每采样一个点后将其周围 $N=20$ 像素范围内的先验值置零并重新归一化，以确保采样点分散到不同物体上
- **设计动机**：论文系统比较了 8 类物体先验（网格/超像素/轮廓密度/VOCUS2显著性/DeepGaze/CAM/DINO/U-Net），发现 DINO 的自注意力图是最佳选择——它能精准聚焦于物体的判别性区域（如北极熊的面部），而不像 VOCUS2 或 U-Net 那样也高亮背景区域。DINO 比 Grid 基线 F1 提升 7.5 分。

#### 2. 伪标注生成器（PAC）

- **功能**：用 OLM 输出的点提示驱动预训练 SAM 生成分割候选，经筛选后与原始标注合并。
- **核心思路**：
    - 对每个点提示，SAM 生成 3 个候选分割（处理歧义情况）
    - **置信度过滤**：移除 SAM 置信度低于 $\tau_{conf}=0.9$ 的分割
    - **NMS 去重**：IoU 阈值 $\tau_{NMS}=0.95$ 去除近重复分割
    - **与原始标注合并**：如果伪标注与任何原始标注的 IoU 超过 $\tau_{NMS}$，则抑制该伪标注（只保留覆盖未知物体的伪标注）
    - **数量限制**：每张图最多保留 $P=10$ 个伪标注（实际平均 7.8 个）
- **设计动机**：仅靠点提示驱动 SAM 仍会产生噪声分割（物体部件、stuff 区域等），因此需要严格的后处理。$\tau_{conf}=0.9$ 的高阈值确保只保留 SAM 高度确信的分割，$\tau_{NMS}=0.95$ 去除几乎完全重叠的冗余分割。与 GGN 仅用 3 个伪标注相比，SOS 可用 10 个而不引入过多噪声，正是因为伪标注质量更高。

#### 3. 实例分割训练

- **功能**：用合并后的标注（原始标注 + 高质量伪标注）训练类别无关（class-agnostic）的 Mask R-CNN (ResNet-50 + FPN)。
- **设计动机**：选择标准实例分割系统是为了（a）最大灵活性，系统可替换为任意实例分割网络；（b）验证性能提升来自伪标注质量而非模型改进。类别无关训练使系统不假设物体类别数量，天然适合开放世界场景。

### 损失函数 / 训练策略

- 使用 Mask R-CNN 的默认训练配置（类别无关模式）
- DINO（ViT-S）在 ImageNet 上自监督预训练，不做微调
- SAM（ViT-H）在 SA-1B 上预训练，不做微调
- 伪标注仅在训练时需要，推理时只运行 Mask R-CNN
- 无额外训练技巧或损失修改

## 实验关键数据

### 主实验

**COCO (VOC) → COCO (non-VOC) 跨类别评估**：

| 方法 | AP | AR100 | F1 |
|------|-----|-------|-----|
| Mask R-CNN | 1.0 | 8.2 | 1.8 |
| SAM（Grid 提示） | 3.6 | 48.1 | 6.7 |
| OLN | 4.2 | 28.4 | 7.3 |
| GGN | 4.9 | 28.3 | 8.4 |
| SWORD | 4.8 | 30.2 | 8.3 |
| **SOS（本文）** | **8.9** | **39.3** | **14.5** |

vs GGN（第二名）：F1 **+6.1**，精度 **+81.6%**（4.9→8.9），召回 **+38.9%**（28.3→39.3）。

**跨数据集评估**：

| 设置 | SOS F1 | 第二名 F1 | 提升 |
|------|--------|----------|------|
| COCO → LVIS | **13.3** | GGN/LDET 10.5 | +2.8 |
| COCO → ADE20k | **17.0** | GGN 13.3 | +3.7 |
| COCO → UVO | 28.0 | LDET **28.5** | -0.5 |

在 UVO 上略低于 LDET，因为 UVO 包含 ImageNet 以外的物体类别，而 DINO 是在 ImageNet 上训练的，可能遗漏这些类别。

### 消融实验

**物体先验比较**（COCO 跨类别设置）：

| 物体先验 | AP | AR100 | F1 | 说明 |
|---------|-----|-------|-----|------|
| Grid（SAM 默认） | 3.8 | 36.5 | 6.9 | 包含大量 stuff |
| Dist（训练集分布） | 3.4 | 27.4 | 6.0 | 忽略图像内容 |
| Spx（超像素） | 5.6 | 34.8 | 9.6 | 简单有效 |
| Contour（轮廓密度） | 5.6 | 36.6 | 9.7 | |
| VOCUS2（显著性） | 6.1 | 37.7 | 10.5 | 高亮背景高对比区域 |
| CAM | 5.4 | 36.7 | 9.4 | 类激活图 |
| DeepGaze | 5.4 | 35.9 | 9.4 | 眼动数据 |
| U-Net | 7.3 | 37.3 | 12.2 | 学习像素级物体位置 |
| **DINO** | **8.9** | **38.1** | **14.4** | 自注意力场景布局 |
| GT（上界） | 18.1 | 42.5 | 25.4 | 使用 GT 物体中心 |

**组件消融**：

| 配置 | AP | AR100 | F1 |
|------|-----|-------|-----|
| Mask R-CNN（无伪标注） | 1.2 | 10.8 | 2.2 |
| + Grid 伪标注（无后处理） | 3.4 | 35.2 | 6.2 |
| + DINO 伪标注（无后处理） | 8.9 | 37.1 | 14.3 |
| + DINO + 后处理（完整 SOS） | **8.9** | **38.1** | **14.4** |

**伪标注数量**：

| 数量 | AP | AR100 | F1 |
|------|-----|-------|-----|
| 3 | 8.3 | 34.5 | 13.4 |
| 5 | 8.8 | 36.2 | 14.2 |
| **10** | **8.9** | **38.1** | **14.4** |
| 20 | 8.8 | 38.1 | 14.3 |

### 伪标注质量量化

在 COCO 训练集上评估伪标注覆盖非 VOC 类物体的能力：

| 方法 | 精度 | 召回 | F1 |
|------|------|------|-----|
| GGN（3 个） | 7.3 | 12.1 | 9.1 |
| SOS（3 个） | **19.0** | 26.4 | **22.1** |
| SOS（10 个） | 15.5 | **41.7** | 22.6 |

SOS 的伪标注精度是 GGN 的 **2.6 倍**，验证了 DINO 聚焦物体的有效性。

### 关键发现

- **物体先验选择是核心**：DINO 相比 Grid 基线 F1 翻倍（14.4 vs 6.9），主要通过大幅提升精度实现（8.9 vs 3.8）。
- **精度提升是最大亮点**：vs GGN 精度提升 81.6%，打破了 OWIS 领域"高召回低精度"的困局。
- **组合 SAM + DINO 形成互补**：DINO 知道物体在哪（定位），SAM 知道物体长什么样（分割），两者结合产生高质量伪标注。
- **10 个伪标注为最佳**：比 GGN 的 3 个更多，但因为质量高不会引入过多噪声。

## 亮点与洞察

- **"物体先验 + 基础模型提示"的范式**：提供了一种通用方法论——如何将通用基础模型（SAM）转化为任务特定工具。通过设计合理的提示策略（而非微调），可最大化利用基础模型的能力。这一思路可迁移到其他 SAM 下游应用（如弱监督分割、零样本检测等）。
- **物体先验的系统研究**：论文对 8 种物体先验的比较本身就是重要贡献，为任何需要类别无关物体定位的任务提供了实验指导。
- **高质量伪标注 > 数量**：SOS 用 10 个高质量伪标注远超 GGN 的 3 个低质量伪标注，说明伪标注的"质量优先"原则。
- **推理无额外开销**：DINO 和 SAM 仅用于生成训练伪标注，推理时直接用 Mask R-CNN，不增加部署成本。

## 局限与展望

1. **DINO 依赖 ImageNet**：DINO 在 ImageNet 上预训练，对 ImageNet 以外的物体类别（如 UVO 中的特殊物体）可能遗漏，导致 COCO→UVO 上表现略低于 LDET。可通过在更多样化无标签数据上训练 DINO 来缓解。
2. **伪标注仍有噪声**：精度最高也只有 19%（IoU=0.5 标准），说明大部分伪标注仍不完美。未来可引入更强的过滤机制或迭代精炼。
3. **固定的提示策略**：OLM 使用固定的采样策略（50 个点，N=20 像素排斥区），未探索自适应策略。
4. **仅评估 Mask R-CNN**：虽然声称可替换为任意实例分割系统，但未实际验证其他架构（如 query-based 方法）的表现。
5. **类别信息未利用**：SOS 是完全类别无关的，未探索如何利用已知类别的语义信息来进一步提升未知类别的检测。

## 相关工作与启发

- **vs GGN**：GGN 通过学习像素亲和力生成伪标注，但结果噪声大（包含大量背景片段）。SOS 利用 DINO+SAM 生成聚焦物体的高质量伪标注，精度是 GGN 的 2.6 倍。
- **vs OLN/SWORD**：这些方法通过修改分类头来改善泛化，但不增加训练数据。SOS 通过增加高质量伪标注从数据侧解决问题，两种思路可能互补。
- **vs 直接使用 SAM**：vanilla SAM 的 Grid 提示方式 F1 仅 6.7（精度 3.6），因为它不区分物体和 stuff。SOS 通过 DINO 先验将 SAM 的精度从 3.6 提升到 8.9，验证了"提示设计"而非"微调"的重要性。

## 评分

- 新颖性: ⭐⭐⭐⭐ 物体先验 + SAM 提示的组合思路新颖，物体先验的系统研究有开创性意义
- 实验充分度: ⭐⭐⭐⭐⭐ 4 个评估设置、8 种先验对比、组件消融、伪标注质量量化，极为全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，物体先验研究部分尤其出色，定量+定性分析充分
- 价值: ⭐⭐⭐⭐ 提供了利用基础模型解决开放世界任务的通用范式，精度的大幅提升解决了该领域的关键瓶颈

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] ActionVOS: Actions as Prompts for Video Object Segmentation](actionvos_actions_as_prompts_for_video_object_segmentation.md)
- [\[ECCV 2024\] VISA: Reasoning Video Object Segmentation via Large Language Models](visa_reasoning_video_object_segmentation_via_large_language_models.md)
- [\[ECCV 2024\] Learning Camouflaged Object Detection from Noisy Pseudo Label](learning_camouflaged_object_detection_from_noisy_pseudo_label.md)
- [\[ECCV 2024\] Unsupervised Moving Object Segmentation with Atmospheric Turbulence](unsupervised_moving_object_segmentation_with_atmospheric_turbulence.md)
- [\[CVPR 2025\] V-CLR: View-Consistent Learning for Open-World Instance Segmentation](../../CVPR2025/segmentation/v-clr_view-consistent_learning_for_open-world_instance_segmentation.md)

</div>

<!-- RELATED:END -->
