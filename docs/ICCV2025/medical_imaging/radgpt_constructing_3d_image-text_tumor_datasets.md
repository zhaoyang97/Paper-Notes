---
title: >-
  [论文解读] RadGPT: Constructing 3D Image-Text Tumor Datasets
description: >-
  [ICCV 2025][医学图像][CT报告生成] 本文提出 RadGPT——一个解剖感知的 VL AI 管线，通过将放射科医师修订的肿瘤分割 mask 经由确定性算法转化为结构化报告、再由 LLM 适配为叙述性报告，构建了首个大规模公开腹部 CT 图文肿瘤数据集 AbdomenAtlas 3.0（9,262 例 CT、每体素标注 + 报告），并证明分割辅助可显著提升 AI 报告中的肿瘤检测率。
tags:
  - "ICCV 2025"
  - "医学图像"
  - "CT报告生成"
  - "肿瘤数据集"
  - "分割辅助报告"
  - "腹部CT"
  - "结构化报告"
---

# RadGPT: Constructing 3D Image-Text Tumor Datasets

**会议**: ICCV 2025  
**arXiv**: [2501.04678](https://arxiv.org/abs/2501.04678)  
**代码**: [https://github.com/MrGiovanni/RadGPT](https://github.com/MrGiovanni/RadGPT)  
**领域**: 医学图像 / 报告生成  
**关键词**: CT报告生成, 肿瘤数据集, 分割辅助报告, 腹部CT, 结构化报告

## 一句话总结
本文提出 RadGPT——一个解剖感知的 VL AI 管线，通过将放射科医师修订的肿瘤分割 mask 经由确定性算法转化为结构化报告、再由 LLM 适配为叙述性报告，构建了首个大规模公开腹部 CT 图文肿瘤数据集 AbdomenAtlas 3.0（9,262 例 CT、每体素标注 + 报告），并证明分割辅助可显著提升 AI 报告中的肿瘤检测率。

## 研究背景与动机

**领域现状**：美国每年超过 8500 万次 CT 扫描且年增 6%，远超放射科医师 0.7% 的年增率。AI 辅助报告生成有巨大需求，但缺少训练数据——目前没有公开的腹部 CT 数据集同时包含放射学报告和体素级标注。

**现有痛点**：(a) 公开 CT 数据集要么只有分割 mask 没有报告，要么只有文字标注没有体素级标注；(b) 现有腹部 CT 报告生成模型（M3D、Merlin）训练数据有限，在肿瘤检测上表现极差，尤其对 ≤2cm 的小肿瘤几乎完全漏检；(c) 传统报告生成评价指标（BLEU、ROUGE）容易被写作风格差异干扰，无法真正衡量诊断准确性。

**核心矛盾**：肿瘤在 CT 中可能仅占全体积的 0.0001%，纯视觉 VLM 极难定位如此微小的病变。而分割模型恰好擅长体素级定位，但无法输出结构化文本报告。

**本文目标** (a) 构建首个大规模 CT-mask-报告三元组数据集；(b) 设计从分割到报告的确定性可解释管线；(c) 证明分割辅助能显著改善报告中的肿瘤检出率。

**切入角度**：不让 VLM 直接从图像生成报告（容易漏检小肿瘤），而是先用分割模型定位肿瘤，再用确定性规则提取属性填充模板生成结构化报告，最后用 LLM 转化为叙述风格。这种"分割→属性提取→模板填充→风格适配"的管线确保了报告与分割 mask 的完全一致性和可解释性。

**核心 idea**：用确定性算法将分割 mask 翻译为放射学报告，既保证了报告的可解释性和准确性，又克服了 VLM 对微小肿瘤的漏检问题。

## 方法详解

### 整体框架
RadGPT 是三阶段管线：Stage I 分割（DiffTumor + nnU-Net 分割 26 个解剖结构 → 放射科医师修订）→ Stage II 结构化报告生成（确定性算法从分割中提取属性 → 填充模板）→ Stage III 风格适配（LLM 将结构化报告转化为目标医院的叙述风格）。在构建 AbdomenAtlas 3.0 时加入人工修订；作为全自动模型时跳过修订。

### 关键设计

1. **器官子分割定位肿瘤（Stage II-a）**:

    - 功能：将肝脏分割为 Couinaud 标准的 8 个子段、胰腺分割为头/体/尾，用于在结构化报告中精确描述肿瘤位置
    - 核心思路：肝脏子分割——将肝脏区域 HU 偏移 200 作为输入，在 LiTS 131 例上微调 nnU-Net；胰腺子分割——利用肠系膜上动脉（SMA）作为解剖标志物，确定性算法切分头/体/尾（无公开标注，本文是首个提供的数据集）
    - 设计动机：肿瘤位置对手术规划至关重要（如可否切除），放射科医师的报告必须指明肿瘤所在子段

2. **类放射科医师的肿瘤测量（Stage II-b）**:

    - 功能：从分割 mask 中提取 WHO 标准的肿瘤最长径 $D$ 及其垂直径 $d$、体积、HU 衰减值
    - 核心思路：确定性算法实现 WHO 测量标准——在任意轴位平面找肿瘤最长径及同平面垂直径。结构化报告还包含器官体积（诊断器官增大）和平均 HU（诊断脂肪肝<40 HU、脂肪胰腺<0.7 胰/脾比值）
    - 设计动机：标准化测量确保报告间的可比性，确定性规则确保报告与 mask 完全一致

3. **胰腺癌分期（Stage II-c）**:

    - 功能：从肿瘤和血管的分割中自动完成 PDAC T 分期（T1-T4）
    - 核心思路：先分割 5 条关键血管（SMA、CHA、CA、SMV、PV），测量肿瘤-血管接触角度。接触角 >180° 表示肿瘤不可切除（T4 期）。确定性算法忠实实现放射科指南
    - 设计动机：PDAC 是致命率极高的肿瘤，分期决定手术方案。本数据集是首个提供公开 PDAC T 分期标签的

4. **LLM 风格适配（Stage III）**:

    - 功能：将填充好的结构化报告转化为目标医院的叙述风格
    - 核心思路：使用 in-context learning，优先选择诊断相似的示例报告作为 few-shot 示例，要求 LLM 保留医学信息并做一致性自检。还支持将结构化报告与已有人工报告融合（enhanced 模式）
    - 设计动机：不同医院的报告风格差异大，LLM 适配使数据集有更广的适用性

### 损失函数 / 训练策略
RadGPT 本身无需端到端训练，但用到的分割模型（DiffTumor 肿瘤分割 + nnU-Net 器官/血管分割）分别在不同数据集上训练。报告质量评估使用新提出的诊断指标：用 LLM 从 AI 和人工报告中提取标签（肿瘤有/无），再计算敏感度和特异度。LLM 标签提取的零样本准确率经放射科医师验证为 96%。

## 实验关键数据

### 主实验
AbdomenAtlas 3.0 内部验证（test split）+ UCSF 外部验证，肿瘤检测的敏感度和特异度：

| 模型 | 胰腺 Sen.(≤2cm) | 胰腺 Sen.(>2cm) | 胰腺 Spec. | 肾脏 Sen.(≤2cm) | 肝脏 Sen.(≤2cm) |
|------|--------|--------|--------|--------|--------|
| CT-CHAT | 66.7 | 51.9 | 61.2 | 31.1 | 5.7 |
| CT2Rep | 0.0 | 0.0 | 92.5 | 36.5 | 35.8 |
| M3D | 0.0 | 7.4 | 97.2 | 8.1 | 9.4 |
| Merlin | 33.3 | 51.9 | 71.8 | 28.4 | 30.2 |
| RadFM | 0.0 | 0.0 | 99.9 | 3.7 | 3.3 |
| **RadGPT** | **66.7** | **81.5** | **93.2** | **54.8** | **39.6** |

### 消融 / 数据集统计

| 指标 | 数值 |
|------|------|
| CT 扫描总数 | 9,262 |
| 包含肿瘤的 CT | 3,955 |
| 新标注的肿瘤 | 3,011（将源数据集的标注扩大 4.2×） |
| 报告 token 总数 | 1,843,262 |
| 小肿瘤（≤2cm） | 7,003 |
| 肝脏肿瘤 | 5,582 | 
| 肾脏肿瘤 | 4,424 |
| 胰腺肿瘤 | 368 |

### 关键发现
- **分割辅助戏剧性提升肿瘤检出**：RadGPT 胰腺肿瘤>2cm 敏感度 81.5%，远超所有纯 VLM（最高仅 51.9%）。对 ≤2cm 小肿瘤差距更大
- **纯 VLM 几乎无法检测小肿瘤**：CT2Rep、M3D、RadFM 对胰腺和肝脏小肿瘤的敏感度接近 0%，说明 VLM 面对体积占比极小的病变严重不足
- **外部验证性能保持**：在 UCSF 外部数据集上，RadGPT 仍显著优于所有对比方法
- **传统文本指标不可靠**：Merlin 的 BLEU/ROUGE 分数与 RadGPT 相当，但诊断敏感度差距巨大，进一步证实了使用诊断指标评估的必要性

## 亮点与洞察
- **确定性管线的可解释性**：关键创新在于 Stage II 完全使用确定性算法（无 ML），确保结构化报告与分割 mask 100% 一致。放射科医师只需审核分割即可信任报告，无需额外审核文本
- **数据集的综合价值**：AbdomenAtlas 3.0 同时提供 CT、体素级标注和报告三元组，填补了腹部 CT 数据集的关键空白，可推动分割辅助报告生成这一新范式
- **诊断导向的评估指标**：提出用 LLM 从报告中提取诊断标签再计算敏感度/特异度的评估方案，比 BLEU/ROUGE 更贴近临床需求
- **胰腺癌自动分期**：首次实现从分割 mask 到 T 分期的全自动管线，对高致死率的 PDAC 有重要临床意义

## 局限与展望
- **依赖分割质量**：全自动模式下分割错误会直接传导到报告中，尤其小肿瘤的假阳/假阴影响较大
- **仅覆盖三种器官肿瘤**：当前仅处理肝/肾/胰腺肿瘤，肺、结肠等其他常见肿瘤尚未覆盖
- **叙述报告的风格单一**：LLM 风格适配基于单个医院的示例报告，跨机构泛化性待验证
- **改进方向**：可探索将分割信息以 visual prompt 的形式注入 VLM，让模型端到端地利用位置信息；可扩展到不确定性估计——分割置信度低时报告中加注提示

## 相关工作与启发
- **vs CT2Rep [ECCV'24]**：CT2Rep 纯视觉方法，对肿瘤附近的细微差异缺乏感知能力，肿瘤检测敏感度极低
- **vs Merlin [MICCAI'24]**：Merlin 在 BLEU 指标上表现不错但诊断敏感度差距大，说明文本相似度≠诊断准确度
- **vs 传统 2D X-ray 报告生成**：X-ray 中病变区域占比大（5-10%），而 CT 中肿瘤可能仅占 0.0001%，直接迁移 2D 方法不可行

## 评分
- 新颖性: ⭐⭐⭐⭐ "分割→报告"的确定性管线思路新颖且实用，但方法偏工程化
- 实验充分度: ⭐⭐⭐⭐⭐ 6 个模型对比、内外部验证、新评估指标经放射科医师验证
- 写作质量: ⭐⭐⭐⭐ 内容详实，数据集描述清晰，但论文篇幅较长
- 价值: ⭐⭐⭐⭐⭐ 数据集贡献巨大，代码和数据公开，对社区有分水岭意义

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MultiverSeg: Scalable Interactive Segmentation of Biomedical Imaging Datasets with In-Context Guidance](multiverseg_scalable_interactive_segmentation_of_biomedical_imaging_datasets_wit.md)
- [\[CVPR 2026\] VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation](../../CVPR2026/medical_imaging/voxtell_free-text_promptable_universal_3d_medical_image_segmentation.md)
- [\[ICCV 2025\] Scaling Tumor Segmentation: Best Lessons from Real and Synthetic Data](scaling_tumor_segmentation_best_lessons_from_real_and_synthetic_data.md)
- [\[ICCV 2025\] M-Net: MRI Brain Tumor Sequential Segmentation Network via Mesh-Cast](m-net_mri_brain_tumor_sequential_segmentation_network_via_mesh-cast.md)
- [\[ICCV 2025\] UKBOB: One Billion MRI Labeled Masks for Generalizable 3D Medical Image Segmentation](ukbob_one_billion_mri_labeled_masks_for_generalizable_3d_medical_image_segmentat.md)

</div>

<!-- RELATED:END -->
