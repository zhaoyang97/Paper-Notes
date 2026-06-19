---
title: >-
  [论文解读] Pose Priors from Language Models
description: >-
  [CVPR 2025][人体理解][3D姿态估计] 提出 ProsePose 框架，利用大型多模态模型 (LMM, 如 GPT-4V) 作为接触先验，从图像中提取身体部位接触约束并转化为可优化的损失函数，在无需人工接触标注的情况下改善双人交互和自接触场景的 3D 姿态估计。 语言天然地编码了丰富的物理和社会交互信息——世代以…
tags:
  - "CVPR 2025"
  - "人体理解"
  - "3D姿态估计"
  - "大语言模型"
  - "接触预测"
  - "姿态优化"
  - "物理交互"
---

# Pose Priors from Language Models

**会议**: CVPR 2025  
**arXiv**: [2405.03689](https://arxiv.org/abs/2405.03689)  
**代码**: [GitHub](https://prosepose.github.io)  
**领域**: 人体姿态估计 (Human Pose Estimation)  
**关键词**: 3D姿态估计, 大语言模型, 接触预测, 姿态优化, 物理交互

## 一句话总结

提出 ProsePose 框架，利用大型多模态模型 (LMM, 如 GPT-4V) 作为接触先验，从图像中提取身体部位接触约束并转化为可优化的损失函数，在无需人工接触标注的情况下改善双人交互和自接触场景的 3D 姿态估计。

## 研究背景与动机

语言天然地编码了丰富的物理和社会交互信息——世代以来发展出了描述拥抱、握手、运动姿势等细微差别的词汇。然而，大多数 3D 人体姿态估计方法忽略了这一信息源。

接触场景（双人交互或自接触）对姿态估计尤为挑战：
1. **遮挡严重**：接触处身体部位被遮挡，仅依赖 2D 关键点无法推断接触信息
2. **数据获取昂贵**：现有方法如 BUDDI、REMIPS 依赖手动标注的接触图或动捕数据训练，此类数据集仅有几万张图像
3. **泛化受限**：在特定接触数据集上训练的模型难以泛化到新的交互类型

核心 insight：LMM 在海量图文对上预训练，隐式学习了丰富的人体姿态和交互知识。如果能让 LMM 说出哪些身体部位在接触，就可以将语言描述转化为 3D 姿态优化的约束。这提供了一条**无需任何接触标注数据**的可扩展替代路径。

## 方法详解

### 整体框架

ProsePose 框架分为三个阶段：
1. **初始姿态估计**：使用姿态回归器（双人用 BEV，单人用 HMR2）获得粗略 3D 姿态
2. **LMM 约束生成**：向 LMM 发送图像和指令，提取身体部位接触对列表
3. **约束姿态优化**：将接触约束转化为损失函数，与关键点损失、先验损失联合优化姿态参数

### 关键设计

1. **LMM 接触约束生成**:
    - 功能：从图像中自动推断身体部位间的接触关系
    - 核心思路：裁剪并分割目标人物图像，送入 LMM (GPT-4V)。Prompt 指定粗粒度身体区域（arm, shoulder, back, waist 等），要求列出所有接触的区域对。使用 chain-of-thought 先描述姿态再列约束。不指定左右手（LMM 在 chirality 上不可靠），后续枚举所有左右分配取最小损失。采样 N=20 次响应取频率过滤
    - 设计动机：粗粒度区域匹配 LMM 的语言粒度；多次采样+频率过滤缓解幻觉

2. **约束到损失函数的转换**:
    - 功能：将语言约束转化为可微的优化目标
    - 核心思路：每个约束 c=(R_a,R_b) 对应两组 mesh 顶点，损失=两组间最小距离。对 N 个采样的约束集取平均 loss（类似 self-consistency）。对双人场景取两种人员映射中的最小 loss。空约束集超阈值时回退到基线
    - 设计动机：平均多次采样的损失有效降低单次幻觉的影响

3. **联合姿态优化**:
    - 功能：综合多种损失约束优化 SMPL-X 参数
    - 核心思路：分两阶段——阶段一优化 θ(姿态)+β(体型)+t(平移)，阶段二固定 β 仅优化 θ 和 t。总损失 = λ_LMM·L_LMM + λ_GMM·L_GMM + λ_β·L_β + λ_θ·L_θ + λ_2D·L_2D + λ_P·L_P
    - 设计动机：多源损失相互制衡——L_LMM 引导接触、L_2D 保持投影准确、L_θ 防止偏离初始化、L_P 防止互穿

### 损失函数 / 训练策略

- **L_LMM**: LMM 接触约束损失（核心贡献）
- **L_GMM**: 高斯混合姿态先验
- **L_β**: 体型正则 ||β||²₂
- **L_θ**: 初始姿态偏差惩罚 ||θ'-θ||²₂
- **L_2D**: 2D 关键点重投影损失（OpenPose + ViTPose）
- **L_P**: 互穿惩罚损失（基于 winding numbers）

## 实验关键数据

### 主实验

双人交互（Joint PA-MPJPE↓ / PCC↑）：

| 方法 | Hi4D PM↓ | FlickrCI3D PM↓ | FlickrCI3D PCC↑ | CHI3D PM↓ | CHI3D PCC↑ |
|------|----------|----------------|-----------------|-----------|------------|
| BEV (初始化) | 144 | 106 | 64.8 | 96 | 71.4 |
| Heuristic | 116 | 67 | 77.8 | 105 | 74.1 |
| **ProsePose** | **93** | **58** | **79.9** | 100 | **75.8** |
| BUDDI (有监督) | 89 | 66 | 81.9 | 68 | 78.0 |

单人瑜伽（MOYO 数据集）：

| 方法 | PA-MPJPE↓ | PCC↑ | PCC@5mm↑ | PCC@15mm↑ |
|------|-----------|------|----------|-----------|
| HMR2 | 84 | 83.0 | 34.2 | 69.5 |
| HMR2+opt | 81 | 85.2 | 47.7 | 74.6 |
| **ProsePose** | 82 | **87.8** | **54.2** | **81.4** |

### 消融实验

| 配置 | PA-MPJPE (Hi4D val) |
|------|---------------------|
| 完整模型 | 81 |
| 去掉 L_LMM | 138 |
| 去掉 L_2D | 130 |
| 去掉 L_GMM | 85 |
| 去掉 L_β | 91 |
| 去掉 L_θ | 84 |
| 去掉 L_P | 78 |

采样数影响：N 从 1 增到 20，PA-MPJPE 持续下降、PCC 持续上升

### 关键发现

1. **无监督方法中最佳**: ProsePose 在所有不使用接触监督的方法中表现最好
2. **缩小与监督方法的差距**: 在 Hi4D 上消除了 Heuristic 和 BUDDI 之间 85% 的 PA-MPJPE 差距
3. **L_LMM 和 L_2D 贡献最大**: 去掉这两个损失导致 PA-MPJPE 分别飙升至 138 和 130
4. **多次采样是关键**: 单次 LMM 预测噪声大，20 次采样取平均显著提升性能
5. **Chirality 是 LMM 的主要弱点**: LMM 难以区分左右肢体

## 亮点与洞察

1. **LMM 作为 3D 理解工具的新范式**: 将 LMM 的隐式物理知识提取为显式约束，桥接语言和 3D 几何
2. **零标注的可扩展方案**: 不需要任何接触标注数据，直接利用预训练 LMM
3. **优雅的鲁棒性策略**: 多采样+频率过滤+损失平均 = 类似 self-consistency 的集成方法
4. **统一框架**: 同一框架处理双人交互和单人自接触两种场景
5. **Oracle 实验**: Ground-truth 接触可将 PA-MPJPE 从 93 降至 81，说明更好的 LMM 有巨大改进空间

## 局限与展望

1. **LMM 的 chirality 困难**: 无法可靠区分左右肢体，限制了约束精确度
2. **粗粒度区域**: 目前使用的身体区域较粗，更精细的约束可能进一步改善
3. **回退比例高**: 在 CHI3D 上 224/431 例子回退到基线
4. **LMM 调用成本**: GPT-4V 20 次采样的 API 成本不低
5. 未来随着 LMM 能力提升，该框架效果将自然改善

## 相关工作与启发

- **vs BUDDI**: BUDDI 使用学习的扩散先验需要接触标注数据训练；ProsePose 无需任何接触标注
- **vs PoseScript/PoseFix**: 这些方法也使用语言和姿态但需要大量 text-pose 配对训练数据；ProsePose 直接利用预训练 LMM 的零样本能力
- **vs PoseGPT**: PoseGPT 用语言作为训练数据但未超越纯回归方法；ProsePose 证明了 LMM 先验在接触场景中的有效性
- **vs CloseInt**: CloseInt 训练物理引导扩散模型需要双人动捕数据；ProsePose 不需要任何此类数据

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 LMM 作为 3D 姿态的接触先验是新颖且有启发性的范式
- 实验充分度: ⭐⭐⭐⭐ 三个双人数据集+一个单人数据集，详细消融，LMM 分析深入
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法形式化严谨，实验分析透彻
- 价值: ⭐⭐⭐⭐ 开创了 LMM→3D 姿态优化的新方向，随着 LMM 进步潜力巨大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ChatGarment: Garment Estimation, Generation and Editing via Large Language Models](chatgarment_garment_estimation_generation_and_editing_via_large_language_models.md)
- [\[CVPR 2026\] Bridging Facial Understanding and Animation via Language Models](../../CVPR2026/human_understanding/bridging_facial_understanding_and_animation_via_language_models.md)
- [\[NeurIPS 2025\] VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models](../../NeurIPS2025/human_understanding/vimorag_video-based_retrieval-augmented_3d_motion_generation_for_motion_language.md)
- [\[CVPR 2025\] UniPose: A Unified Multimodal Framework for Human Pose Comprehension, Generation and Editing](unipose_a_unified_multimodal_framework_for_human_pose_comprehension_generation_a.md)
- [\[CVPR 2025\] Two is Better than One: Efficient Ensemble Defense for Robust and Compact Models](two_is_better_than_one_efficient_ensemble_defense_for_robust_and_compact_models.md)

</div>

<!-- RELATED:END -->
