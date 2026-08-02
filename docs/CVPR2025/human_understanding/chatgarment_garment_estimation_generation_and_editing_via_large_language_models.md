---
title: "ChatGarment: Garment Estimation, Generation and Editing via Large Language Models"
authors: "Siyuan Bian, Chenghao Xu, Yuliang Xiu, Michael J. Black, Cewu Lu, Gerard Pons-Moll"
affiliations: "MPI for Informatics, Shanghai Jiao Tong University, EPFL, ETH Zurich"
venue: "CVPR 2025"
date: 2024-12-23
tags: ["garment generation", "3D clothing", "VLM", "LLaVA", "sewing pattern", "GarmentCode"]
arxiv: "2412.17811"
code: "https://chatgarment.github.io/"
---

# ChatGarment: Garment Estimation, Generation and Editing via Large Language Models

## 研究背景与动机

3D 服装建模在虚拟试衣、数字人、游戏和电影制作中具有重要应用价值。传统的3D服装建模依赖于专业的CAD软件和服装设计师，门槛极高。近年来，基于深度学习的方法尝试从图像直接预测3D服装形状，但面临以下问题：

**几何质量**：端到端回归的网格（mesh）往往缺乏细节，尤其是在褶皱和缝合线处

**可编辑性**：预测的3D网格难以进行后续的参数化编辑（如修改袖长、领口形状）

**物理合理性**：直接回归的形状不保证物理模拟的可行性，无法用于布料仿真

**语言交互**：用户难以通过自然语言描述来控制服装的生成和编辑

**GarmentCode** 提供了一种参数化的服装表示方案：每件服装由一组JSON参数定义，涵盖衣型、尺寸、版型细节等，可以确定性地生成缝纫图样（sewing pattern），进而通过物理仿真得到高质量的3D服装。然而，GarmentCode 的参数空间复杂（原始约900个token），难以直接由神经网络预测。

本文提出 ChatGarment，将视觉语言模型（VLM）与GarmentCode结合，实现基于图像或文本输入的3D服装生成和编辑。

## 方法详解

### GarmentCodeRC：压缩参数表示

原始 GarmentCode 的 JSON 参数约包含900个token，对于LLM来说预测长度过大。本文提出 **GarmentCodeRC**（Reduced Code），通过以下策略将参数量压缩至约350个token：

| 压缩策略 | 说明 | token减少量 |
|---------|------|-----------|
| 冗余参数移除 | 删除可由其他参数推导的冗余项 | ~200 tokens |
| 数值精度截断 | 将浮点数精度从6位降至3位 | ~150 tokens |
| 键名缩写 | 使用短键名替代长键名 | ~100 tokens |
| 默认值省略 | 省略等于默认值的参数 | ~100 tokens |

总压缩比：$900 \to 350$ tokens（约61%压缩率），同时保持生成质量无损。

### LLaVA 微调

ChatGarment 基于 LLaVA 架构，进行以下微调：

#### 输入模态
- **图像输入**：服装照片或渲染图
- **文本输入**：自然语言描述（如"一件V领短袖连衣裙"）
- **混合输入**：图像 + 编辑指令（如"把袖子改长一些"）

#### \<ENDS\> Token

对于JSON中的数值参数，标准的文本生成方式（逐digit预测）效率低且误差累积。本文引入特殊的 **\<ENDS\>** token 来标记数值的结束：

```
"sleeve_length": 0.65<ENDS>
```

\<ENDS\> token 的作用：
1. 明确数值边界，避免多余digit的生成
2. 作为解码时的终止信号，提高数值预测的确定性
3. 减少数值回归任务中的累积误差

### 数据构建

| 数据类型 | 数量 | 用途 |
|---------|------|------|
| 3D服装模型 | 40,000 件 | GarmentCode生成的参数化服装 |
| 多视角渲染图 | 1,000,000+ 张 | 每件服装渲染25+视角 |
| 文本描述 | 40,000 条 | 自动生成 + 人工校验 |
| 编辑指令对 | 200,000+ 对 | (原始参数, 编辑指令, 目标参数) 三元组 |

### 训练策略

训练分为三个阶段：
1. **阶段1**：冻结视觉编码器，训练投影层和LLM，使模型理解服装图像
2. **阶段2**：端到端微调，学习从图像/文本到GarmentCodeRC的映射
3. **阶段3**：编辑指令微调，学习基于编辑指令的参数修改

## 实验结果

### 服装重建精度

| 方法 | Dress4D CD↓ (mm) | CAPE CD↓ (mm) | 参数准确率↑ |
|------|-----------------|--------------|-----------|
| SewFormer | 27.06 | 23.45 | 62.3% |
| NeuralTailor | 19.82 | 17.31 | 71.5% |
| DressCode | 8.47 | 7.92 | 83.1% |
| **ChatGarment** | **3.12** | **3.85** | **94.7%** |

ChatGarment 在 Dress4D 数据集上的 Chamfer Distance 仅为 3.12mm，相比 SewFormer 的 27.06mm 降低了 88.5%。

### 文本到服装生成

| 评估维度 | ChatGarment | 基线方法 |
|---------|-------------|---------|
| 文本一致性 (CLIP Score) | 0.312 | 0.247 |
| 几何质量 (FID-3D) | 23.7 | 45.2 |
| 用户偏好率 | 78.3% | 21.7% |

### 服装编辑

ChatGarment 支持多种编辑操作：
- 局部编辑：修改袖长、领口、裙摆等
- 风格迁移：将一件衣服的风格应用到另一件
- 语义编辑：通过自然语言描述修改（"加个蝴蝶结"、"收腰一点"）

编辑后的参数变化精确对应用户意图，保持未编辑部分不变。

## 总结与展望

ChatGarment 创新性地将 VLM 与参数化服装表示结合，通过 GarmentCodeRC 压缩（900→350 tokens）、\<ENDS\> 数值终止token 和大规模数据构建，实现了高质量的3D服装生成。在 Dress4D 上 CD 3.12 vs SewFormer 27.06，展示了巨大的精度优势。该系统使非专业用户也能通过自然语言交互创建和编辑3D服装。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Pose Priors from Language Models](pose_priors_from_language_models.md)
- [\[NeurIPS 2025\] VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models](../../NeurIPS2025/human_understanding/vimorag_video-based_retrieval-augmented_3d_motion_generation_for_motion_language.md)
- [\[ECCV 2024\] CoMo: Controllable Motion Generation Through Language Guided Pose Code Editing](../../ECCV2024/human_understanding/como_controllable_motion_generation_through_language_guided_pose_code_editing.md)
- [\[CVPR 2025\] UniPose: A Unified Multimodal Framework for Human Pose Comprehension, Generation and Editing](unipose_a_unified_multimodal_framework_for_human_pose_comprehension_generation_a.md)
- [\[ICML 2025\] Scaling Large Motion Models with Million-Level Human Motions](../../ICML2025/human_understanding/scaling_large_motion_models_with_million-level_human_motions.md)

</div>

<!-- RELATED:END -->
