---
title: >-
  [论文解读] 4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models
description: >-
  [CVPR 2025][多模态VLM][4D语言场] 提出4D LangSplat，通过多模态大语言模型生成逐物体视频caption来构建4D语言场，结合状态可变形网络建模语义的时间连续演变，首次实现动态场景中时间敏感和时间无关的开放词汇查询。 在3D场景中建立语言场（language field）来支持开放词汇查询已成为机…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "4D语言场"
  - "高斯泼溅"
  - "多模态大语言模型"
  - "开放词汇查询"
  - "动态场景理解"
---

# 4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models

**会议**: CVPR 2025  
**arXiv**: [2503.10437](https://arxiv.org/abs/2503.10437)  
**代码**: [https://github.com/4d-langsplat/4d-langsplat](https://github.com/4d-langsplat/4d-langsplat)  
**领域**: 多模态VLM  
**关键词**: 4D语言场, 高斯泼溅, 多模态大语言模型, 开放词汇查询, 动态场景理解

## 一句话总结
提出4D LangSplat，通过多模态大语言模型生成逐物体视频caption来构建4D语言场，结合状态可变形网络建模语义的时间连续演变，首次实现动态场景中时间敏感和时间无关的开放词汇查询。

## 研究背景与动机

在3D场景中建立语言场（language field）来支持开放词汇查询已成为机器人导航、场景编辑等应用的基础能力。LangSplat通过将CLIP特征植入3D高斯表示，在静态场景中实现了精确高效的3D语言场。然而真实世界本质上是**动态**的。

**核心痛点**：将LangSplat扩展到4D场景面临两大挑战：

**CLIP无法捕获时间动态**：CLIP设计用于静态图像-文本对齐，无法理解状态变化、动作和时间上下文。例如"正在跑的狗"vs"坐着的狗"在CLIP特征中难以区分。

**缺乏像素对齐的物体级视频特征**：现有视觉模型（如VideoCLIP）主要提取全局视频级特征，裁剪物体则混入背景信息或无法区分物体/相机运动。

**核心矛盾**：构建精确的4D语言场需要逐像素、逐物体、时间变化的特征监督，但现有视觉模型无法提供。

**本文切入角度**：绕开视觉特征的局限，转而利用MLLM将视频转化为逐物体的文本描述，再用LLM编码为句子嵌入作为监督信号。核心idea：**用文本语义替代视觉特征来监督时间变化的4D语言场。**

## 方法详解

### 整体框架
1. 先用4D-GS重建动态RGB场景（可变形高斯场）
2. 为每个高斯点增加**两个语言场**：
    - **时间无关语义场**：用CLIP特征学习不随时间变化的语义（如"人"、"杯子"）
    - **时间变化语义场**：用MLLM生成的caption特征学习动态语义（如"正在倒咖啡"）
3. 查询时按需组合两个语义场

### 关键设计

1. **多模态逐物体视频提示（Multimodal Object-Wise Video Prompting）**:

    - 功能：引导MLLM为视频中每个物体生成时间一致、高质量的逐帧caption
    - 核心思路：分两步——①视觉提示：用SAM+DEVA跟踪获得时间一致的物体mask，然后对目标物体构建视觉提示 $\mathcal{P}_{i,t} = \text{Contour}(M_{i,t}) \cup \text{Gray}(M_{i,t}) \cup \text{Blur}(M_{i,t})$（红色轮廓高亮+背景灰度化+背景模糊），保留场景上下文同时聚焦目标物体；②文本提示：先用整个视频生成视频级运动描述 $\mathcal{D}_i$，再以此为上下文为每帧生成具体caption $C_{i,t}$。最后用LLM（e5-mistral-7b）编码caption为句子嵌入 $\mathbf{e}_{i,t}$ 作为逐像素监督。
    - 设计动机：直接裁剪物体提取视觉特征会混入背景或失去运动参考；MLLM能理解视频中的动态语义，且文本嵌入天然与自然语言查询共享空间。视觉提示的三重处理（轮廓+灰度+模糊）确保MLLM关注目标物体而不忽略场景上下文。

2. **状态可变形网络（Status Deformable Network）**:

    - 功能：建模每个高斯点语义特征的时间演变，约束其在有限状态原型间平滑过渡
    - 核心思路：为每个高斯点 $i$ 定义 $K$ 个状态原型特征 $\{\mathbf{S}_{i,1}, ..., \mathbf{S}_{i,K}\}$，时刻 $t$ 的语义特征为状态的线性组合：
    $\mathbf{f}_{i,t} = \sum_{k=1}^K w_{i,t,k} \mathbf{S}_{i,k}, \quad \sum_{k=1}^K w_{i,t,k} = 1$
   权重 $w_{i,t,k}$ 由MLP解码器 $\phi$ 从HexPlane时空特征预测。状态原型和MLP联合训练。
    - 设计动机：直接学习任意的语义变形 $\Delta \mathbf{f}$ 允许特征变到任意语义状态，增加学习难度且破坏时间一致性。现实中物体通常只有有限的语义状态（如站立→行走→跑步），用加权组合约束更合理。实验表明 $K=3$ 最优。

3. **双语义场联合查询**:

    - 功能：支持时间无关和时间敏感两类开放词汇查询
    - 核心思路：时间无关查询只用CLIP语义场计算相关性得到分割mask；时间敏感查询先用时间无关场定位空间区域，再用时间变化场在该区域计算余弦相似度，以均值为阈值筛选时间段。
    - 设计动机：两场分工明确——CLIP适合捕获稳定实体语义，MLLM caption适合捕获动态状态语义。

### 损失函数 / 训练策略
- 时间无关场：在三个SAM层级上用CLIP特征做feature splatting监督
- 时间变化场：用LLM句子嵌入做逐像素回归
- CLIP和文本特征分别用自编码器压缩至3维和6维
- MLLM使用Qwen2-VL-7B，句子嵌入使用e5-mistral-7b

## 实验关键数据

### 主实验：时间敏感查询（HyperNeRF数据集）

| 方法 | 平均Acc(%) | 平均vIoU(%) |
|------|-----------|------------|
| LangSplat | 54.01 | 22.65 |
| Deformable CLIP | 61.80 | 44.72 |
| Non-Status Field | 87.58 | 68.57 |
| **Ours** | **90.83** | **72.26** |

### 时间无关查询

| 方法 | HyperNeRF mIoU | HyperNeRF mAcc | Neu3D mIoU | Neu3D mAcc |
|------|---------------|----------------|-----------|-----------|
| Feature-3DGS | 36.63 | 74.02 | 34.96 | 87.12 |
| Gaussian Grouping | 50.49 | 80.92 | 49.93 | 95.05 |
| LangSplat | 74.92 | 97.72 | 61.49 | 91.89 |
| **Ours** | **82.48** | **98.01** | **85.11** | **98.32** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅Blur视觉提示 | $\Delta_{\text{sim}}$=0.33 | 单一提示不足以聚焦物体 |
| Blur+Gray | $\Delta_{\text{sim}}$=2.15 | 灰度化背景有帮助 |
| Blur+Gray+Contour | $\Delta_{\text{sim}}$=3.32 | 三重视觉提示效果最佳 |
| 仅视频文本提示 | $\Delta_{\text{sim}}$=0.14 | 无图像级提示效果差 |
| 视频+图像提示 | $\Delta_{\text{sim}}$=3.32 | 二者结合最优 |
| K=2 状态数 | Acc=94.56 | 状态数不足 |
| **K=3** | **Acc=97.82** | **最优状态数** |
| K=6 | Acc=94.56 | 过多状态反而性能下降 |

### 关键发现
- CLIP无法理解动态语义：Deformable CLIP在时间敏感查询上比本方法低29.03% Acc
- MLLM文本监督远优于视觉特征监督，尤其在状态转换边界（如饼干刚裂开、咖啡刚开始滴落）
- 状态可变形网络相比直接学习变形场提升3.25% Acc和4.19% vIoU
- 在时间无关查询上也因动态场景建模更准确而超越LangSplat

## 亮点与洞察
- **文本替代视觉的范式转换**：绕过视觉特征直接用MLLM文本描述做监督，巧妙解决了动态语义特征获取的困难
- 视觉提示设计（轮廓+灰度+模糊）在引导MLLM关注目标物体的同时保留场景上下文，效果显著
- Status Deformable Network用有限状态原型建模语义演变，是一个简洁有效的结构化约束
- 时间无关+时间变化双语义场的分工设计，优雅地利用了CLIP和MLLM各自的优势

## 局限与展望
- 依赖MLLM生成caption的质量，对复杂动态场景MLLM可能描述不准确
- SAM+DEVA跟踪在遮挡严重时可能丢失物体
- 状态数 $K$ 需要手动设定，自适应确定更理想
- 计算开销较大：需要逐物体逐帧调用MLLM生成caption
- 在未标注的动态场景数据集上评估仍受限于手工标注

## 相关工作与启发
- LangSplat证明SAM mask + CLIP特征在3D静态语言场中的有效性，本文将其扩展到4D
- 4D-GS的可变形高斯场提供了动态场景重建的基础
- MLLM（如Qwen2-VL）的视频理解能力是本方法的关键使能技术
- 核心启发：**当视觉特征不满足需求时，可以转化为文本域利用LLM的强大语义理解来弥补**

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次用MLLM文本监督构建4D语言场，范式创新
- 实验充分度: ⭐⭐⭐⭐ 两个数据集+多个消融，但动态场景标注数据有限
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，方法描述完整
- 价值: ⭐⭐⭐⭐ 开辟了4D动态语言场的新方向，对机器人和场景理解有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] On the Out-of-Distribution Generalization of Multimodal Large Language Models](on_the_out-of-distribution_generalization_of_large_multimodal_models.md)
- [\[CVPR 2025\] Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models](img-diff_contrastive_data_synthesis_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] 4DP-QA: Scalable QA for 4D Perception in Vision Language Models](../../CVPR2026/multimodal_vlm/4dp-qa_scalable_qa_for_4d_perception_in_vision_language_models.md)
- [\[CVPR 2025\] Period-LLM: Extending the Periodic Capability of Multimodal Large Language Model](period-llm_extending_the_periodic_capability_of_multimodal_large_language_model.md)
- [\[CVPR 2025\] Cross-modal Information Flow in Multimodal Large Language Models](cross-modal_information_flow_in_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
