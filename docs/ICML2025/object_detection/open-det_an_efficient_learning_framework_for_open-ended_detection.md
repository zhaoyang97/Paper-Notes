---
title: >-
  [论文解读] Open-Det: An Efficient Learning Framework for Open-Ended Detection
description: >-
  [ICML 2025][目标检测][Open-Ended Detection] Open-Det 提出了一个高效的开放端目标检测（OED）框架，通过重构目标检测器（解耦 one-to-many/one-to-one 匹配）、引入 VL-prompts 蒸馏模块桥接视觉-语言语义鸿沟、LoRa Head + Text Denoising 加速 LLM 训练、以及 Masked Alignment Loss 消除矛盾监督，仅用 GenerateU 1.5% 的训练数据和 20.8% 的训练 epoch 就取得了更高的检测性能（APr +1.0%）。
tags:
  - "ICML 2025"
  - "目标检测"
  - "Open-Ended Detection"
  - "视觉语言"
  - "知识蒸馏"
  - "LoRa Head"
  - "Masked Alignment Loss"
---

# Open-Det: An Efficient Learning Framework for Open-Ended Detection

**会议**: ICML 2025  
**arXiv**: [2505.20639](https://arxiv.org/abs/2505.20639)  
**代码**: [https://github.com/Med-Process/Open-Det](https://github.com/Med-Process/Open-Det)  
**领域**: 目标检测 / 开放词汇检测 / 多模态VLM  
**关键词**: Open-Ended Detection, Vision-Language Alignment, knowledge distillation, LoRa Head, Masked Alignment Loss

## 一句话总结

Open-Det 提出了一个高效的开放端目标检测（OED）框架，通过重构目标检测器（解耦 one-to-many/one-to-one 匹配）、引入 VL-prompts 蒸馏模块桥接视觉-语言语义鸿沟、LoRa Head + Text Denoising 加速 LLM 训练、以及 Masked Alignment Loss 消除矛盾监督，仅用 GenerateU 1.5% 的训练数据和 20.8% 的训练 epoch 就取得了更高的检测性能（APr +1.0%）。

## 研究背景与动机

**领域现状**：开放词汇检测（OVD）已将检测能力从闭集扩展到开集，但推理时仍需依赖额外的类别词汇表作为输入。Open-Ended Detection（OED）作为更通用的范式，无需预定义类别即可检测并生成目标名称，GenerateU 是该方向的开创性工作。

**现有痛点**：GenerateU 存在三大核心问题——(a) 需要大规模数据集（5.077M）和大量 GPU 资源（16 张 A100）；(b) 训练收敛慢（149 epochs）；(c) 检测性能有限。

**核心矛盾**：作者剖析了三个根本原因：
   - **语义鸿沟**：直接将视觉 query 送入 LLM，高维特征空间中视觉和语言模态间的对齐不充分
   - **矛盾监督**：未考虑图像内类别间关系，产生相互矛盾的损失和梯度
   - **重头 LLM Head + 噪声对齐**：T5 的 32K 词表对应 24.7M 参数的 head 层，训练初期视觉 query 对齐质量差，噪声监督破坏预训练权重

**本文目标** 设计一个高效 OED 框架——加速训练收敛、提升训练效率和性能，同时摆脱对大规模数据的依赖。

**切入角度**：从检测器和 LLM 两端同时加速——检测器用解耦匹配加速 box 训练，LLM 用 LoRa Head + VL-prompts 蒸馏加速名称生成训练。

**核心 idea**：通过四个协作模块（高效检测器 + VL 蒸馏 + 去噪 LLM + 双向对齐），以极少数据和资源实现超越 GenerateU 的 OED 性能。

## 方法详解

### 整体框架

Open-Det 包含四个协作组件：**(1) Object Detector (ODR)** 负责生成 bounding box 和视觉 query；**(2) Prompts Distiller** 通过 VLD-M 将 VLM（CLIP）的视觉-语言对齐知识蒸馏到 VL-prompts 中；**(3) Object Name Generator** 基于 T5 语言模型，用 VL-prompts 作为输入生成目标名称；**(4) Vision-Language Aligner** 通过 BVLA-M 增强视觉 query 和文本嵌入的双向对齐。

输入流程：图像 $I \in \mathbb{R}^{H \times W \times 3}$ 送入检测器得到 decoder query $Q_d$，目标名称 $T$ 送入 VLM 得到文本嵌入 $T_e$。BVLA-M 计算对齐分数和匹配索引，VLD-M 将 VLM 知识蒸馏到 VL-prompts $P_{vl}$ 中，最后送入 T5 生成目标名称。推理阶段不需要 Vision-Language Aligner，实现了 vocabulary-free 检测。

### 关键设计

1. **Object Detector (ODR) — 解耦 One-to-Many/One-to-One 匹配**:

    - 功能：加速 box 检测器的训练收敛
    - 核心思路：前 4 层 decoder 使用 one-to-many 匹配 + cross-attention 增强定位能力，后 2 层使用 one-to-one 匹配 + self-attention 消除重复检测。不需要额外的分支或 head，直接在 decoder 内部解耦
    - 设计动机：借鉴 DINO 的 anchor box 去噪和 one-to-many 加速收敛的思想，但避免了现有方法需要额外分支增加复杂性的问题
    - 此外引入了基于阈值的 query 选择方法：$Q_{id} = \{e_t \in E | \sigma(\text{Linear}(e_t)) > \lambda\}$（$\lambda$ 默认 0.05），实现灵活数量的目标检测

2. **Bidirectional Vision-Language Alignment Module (BVLA-M)**:

    - 功能：增强视觉 query 和文本嵌入之间的对齐质量
    - 核心思路：从 V-to-L 和 L-to-V 两个方向计算对齐分数：$S_{align} = \cos(Q_d \times M_{VL}, T_e) + \cos(Q_d, T_e \times M_{LV})$，其中 $M_{VL}$ 和 $M_{LV}$ 分别是两个方向的变换矩阵
    - 设计动机：视觉 query 通道维度（256）远小于文本嵌入（768），直接将低维映射到高维进行单向对齐效果不佳。双向对齐让两个模态在各自的空间中都进行匹配

3. **Vision-to-Language Distillation Module (VLD-M)**:

    - 功能：将 VLM 的视觉-语言对齐知识蒸馏到 VL-prompts 中，桥接视觉 query 和图像级文本嵌入之间的语义鸿沟
    - 核心思路：通过可变形交叉注意力（deformable cross-attention）让 query $Q_d$ 与 backbone 特征 $B$ 和 encoder 特征 $E$ 交互，自适应采样目标周围的背景信息，将区域嵌入转化为类图像表示。再通过 FFN 重新加权、MLP 投影、线性融合生成 VL-prompts $P_{vl}$
    - 设计动机：区别于 RegionCLIP 生成伪标签、DVDet 裁切扩展框，VLD-M 直接在特征层面丰富背景信息并蒸馏知识，更高效
    - 使用 Cosine Similarity Loss 在 $P_{vl}$ 和 $T_e$ 之间进行监督蒸馏

4. **Object Name Generator — LoRa Head + Text Denoising**:

    - 功能：加速 LLM（T5）的训练并防止噪声对齐破坏预训练权重
    - LoRa Head：训练初期冻结 T5 的 heavy-weight head（24.7M 参数），引入 LoRa 适配器作为轻量 head 加速训练，大幅减少可训练参数
    - Text Denoising：在文本嵌入 $T_e$ 上添加高斯噪声 $\mathcal{N}(0, \sigma^2)$（$\sigma$ 为 $T_e$ 的标准差）后送入 T5 重建文本，增强模型对噪声输入的鲁棒性
    - 设计动机：训练初期 query 未充分训练，对齐质量差属于噪声监督，直接训练 LLM 全部权重会破坏预训练

### 损失函数 / 训练策略

- **Masked Alignment Loss (MAL)**：通过文本嵌入相似度矩阵 $M = T_e \times T_e^\top$ 生成二值 mask（阈值 $\tau = 0.99$），对同类目标标记为 1 避免产生矛盾负约束。与 ScaleDet 不同，MAL 通过相似度二值化的 BCE 更新解决 query-text 匹配冲突，而非统一多数据集标签
- **Joint Loss**：将二分类分数 $p_i$、对齐分数 $s_i$、IoU 分数 $u_i$ 联合优化：$\mathcal{L}_{JL} = -\frac{1}{N}\sum_{i=1}^{N}[(\sqrt{p_i^\alpha s_i^\alpha u_i^{1-2\alpha}} - p_i)^2 y_i \log(p_i) + p_i^2(1-\sqrt{p_i^\alpha s_i^\alpha u_i^{1-2\alpha}})(1-y_i)\log(1-p_i)]$，其中 $\alpha = 0.25$。平方根操作防止三分数乘积过小导致数值不稳定

## 实验关键数据

### 主实验 — LVIS MiniVal 零样本迁移

| 方法 | Backbone | 训练数据量 | Vocab-Free | Epochs | APr | APc | APf | AP |
|------|----------|-----------|------------|--------|-----|-----|-----|-----|
| GLIP(A) | Swin-T | 0.660M | ✗ | - | 14.2 | 13.9 | 23.4 | 18.5 |
| GLIP(C) | Swin-T | 5.456M | ✗ | - | 20.8 | 21.4 | 31.0 | 26.0 |
| Grounding-DINO | Swin-T | 5.460M | ✗ | - | 18.1 | 23.3 | 32.7 | 27.4 |
| GenerateU | Swin-T | 0.077M | ✓ | 149 | 17.4 | 22.4 | 29.6 | 25.4 |
| GenerateU | Swin-T | 5.077M | ✓ | - | 20.0 | 24.9 | 29.8 | 26.8 |
| **Open-Det** | **Swin-T** | **0.077M** | **✓** | **31** | **21.0** | **24.8** | **30.1** | **27.0** |
| **Open-Det** | **Swin-T** | **0.077M** | **✓** | **50** | **21.9** | **25.1** | **30.4** | **27.4** |
| GenerateU | Swin-L | 5.077M | ✓ | - | 22.3 | 25.2 | 31.4 | 27.9 |
| **Open-Det** | **Swin-S** | **0.077M** | **✓** | **31** | **26.0** | **28.6** | **32.8** | **30.4** |
| **Open-Det** | **Swin-L** | **0.077M** | **✓** | **31** | **31.2** | **32.1** | **34.3** | **33.1** |

### COCO & Objects365 零样本评估

| 方法 | Backbone | 训练数据 | COCO AP | Objects365 AP |
|------|----------|---------|---------|---------------|
| GenerateU | Swin-L | VG | 33.0 | 10.1 |
| GenerateU | Swin-L | VG+GRIT | 33.6 | 10.5 |
| **Open-Det** | **Swin-L** | **VG** | **35.8 (+2.2)** | **13.8 (+3.3)** |

### 消融实验 — 各组件贡献

| ODR | BVLA-M | VLD-M | ONG | Losses | APr | APc | APf | AP |
|-----|--------|-------|-----|--------|-----|-----|-----|-----|
| ✗ | ✗ | ✗ | ✗ | ✗ | 10.2 | 17.4 | 23.2 | 19.6 |
| ✓ | ✗ | ✗ | ✗ | ✗ | 13.9 | 19.8 | 27.6 | 23.1 |
| ✓ | ✓ | ✗ | ✗ | ✗ | 14.7 | 20.3 | 27.9 | 23.5 |
| ✓ | ✓ | ✓ | ✗ | ✗ | 16.3 | 24.2 | 29.9 | 26.3 |
| ✓ | ✓ | ✓ | ✓ | ✗ | 16.9 | 24.5 | 29.7 | 26.3 |
| ✓ | ✓ | ✓ | ✓ | ✓ | **21.0** | **24.8** | **30.1** | **27.0** |

### 关键发现

- **VLD-M 贡献最大**：引入 VLD-M 后 APr 从 14.7 跳到 16.3（+1.6），AP 从 23.5 到 26.3（+2.8），证明了视觉-语言蒸馏对于弥合模态鸿沟的关键作用
- **Losses 对稀有类增益巨大**：MAL + Joint Loss 将 APr 从 16.9 提升到 21.0（+4.1），其中 Joint Loss 单独贡献 +2.6 APr，Masked Alignment Loss 贡献 +1.7 APr
- **LoRa Head + Text Denoising 协同效果显著**：单独使用 LoRa Head 提升 APr +1.2，单独使用 Text Denoising 提升 +4.6，组合后 APr 提升 +5.7（15.3→21.0）
- **Backbone 扩展性好**：从 Swin-T 到 Swin-L，APr 从 21.0 大幅提升到 31.2，显示框架可以充分利用更大 backbone 的能力
- **VL 对齐分数**：Open-Det 在 5 万+ 目标实例上取得 0.555±0.074 的对齐分数，远超 GenerateU 的 0.448±0.026

## 亮点与洞察

- **极致的训练效率**：用 1.5% 数据、20.8% epoch、4×V100（vs 16×A100）超越 GenerateU，这种"小数据+简单设备"的范式对于资源受限的研究组非常有价值
- **解耦 decoder 设计**：将 one-to-many 和 one-to-one 匹配在 decoder 层内解耦（前4层/后2层），不需要额外分支，比现有 Co-DETR 等方法更简洁，这个 trick 可迁移到其他 DETR-like 框架
- **LoRa Head 冻结策略**：训练初期冻结重头并用 LoRa 替代——这个思路不限于 OED，任何"先对齐再微调"的多模态训练都可以采用类似策略防止噪声梯度破坏预训练权重
- **Masked Alignment Loss 消除矛盾监督**：用文本嵌入自相似矩阵来识别同类目标并避免错误负约束，这是一个优雅且低成本的方案
- **VLD-M 桥接区域-图像级特征鸿沟**：通过可变形注意力自适应采样背景信息将区域特征"包装"成图像级表示，思路可迁移到任何涉及区域-全局语义对齐的任务

## 局限与展望

- **跨模态语义差异仍然存在**：作者坦承视觉区域和文本嵌入之间的语义鸿沟并未完全消除，受限于 backbone、检测器、VLM、LLM 之间的交互质量
- **仅在 VG 上训练**：虽然展示了数据效率，但未验证在更多样数据集上的表现，也未尝试结合 GRIT5M 数据——这可能进一步提升上限
- **LLM 选择受限**：使用 FlanT5-base（较小），未探索更大的 LLM 如 LLaMA、DeepSeek 等，这些模型可能带来显著提升
- **缺少分割能力**：仅输出 bounding box，作者也建议未来加入分割 decoder 构建统一检测-分割框架
- **推理速度未报告**：虽然强调训练效率，但未给出推理 FPS/延迟数据，无法评估其实际部署能力

## 相关工作与启发

- **vs GenerateU**: 同为 OED 框架，GenerateU 直接将视觉 query 送入 LLM，Open-Det 引入 VL-prompts 蒸馏和双向对齐作为中间桥梁，在极少资源下实现更优性能
- **vs GLIP/Grounding-DINO**: 这些 OVD 方法在推理时仍需类别词汇表输入，Open-Det 完全实现 vocabulary-free，且仅用 1.4%~11.7% 的训练数据就超越它们
- **vs RegionCLIP/DVDet**: 在区域-文本对齐方面，RegionCLIP 生成伪标签，DVDet 裁切扩展框，Open-Det 的 VLD-M 直接在特征层面蒸馏，更加端到端
- **vs Co-DETR**: 在 one-to-many 匹配方面，Co-DETR 用额外分支，Open-Det 在 decoder 内解耦更简洁

## 评分

- 新颖性: ⭐⭐⭐⭐ 多个模块的组合创新（VLD-M、BVLA-M、MAL、LoRa Head）各自有创新点，但单个技术突破性一般
- 实验充分度: ⭐⭐⭐⭐ 在 LVIS、COCO、Objects365 上验证，消融实验详尽，但缺少推理速度对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题分析到位，公式推导完整，图表丰富
- 价值: ⭐⭐⭐⭐ 对资源受限场景下的 OED 有显著实用价值，但 OED 本身还是较新的方向，影响面待观察

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](../../AAAI2026/object_detection/vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)
- [\[NeurIPS 2025\] DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection](../../NeurIPS2025/object_detection/dithub_a_modular_framework_for_incremental_openvocabulary_ob.md)
- [\[CVPR 2026\] SRA-Det: Learning Omni-Grained Open-Vocabulary Detection Beyond Category Names](../../CVPR2026/object_detection/sra-det_learning_omni-grained_open-vocabulary_detection_beyond_category_names.md)
- [\[CVPR 2026\] Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection](../../CVPR2026/object_detection/parameter-efficient_semantic_augmentation_for_enhancing_open-vocabulary_object_d.md)
- [\[CVPR 2025\] Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection](../../CVPR2025/object_detection/distribution_prototype_diffusion_learning_for_open-set_supervised_anomaly_detect.md)

</div>

<!-- RELATED:END -->
