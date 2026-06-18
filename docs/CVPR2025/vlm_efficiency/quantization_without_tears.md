---
title: >-
  [论文解读] Quantization without Tears
description: >-
  [CVPR 2025][多模态VLM][网络量化] 提出 QwT（Quantization without Tears）方法，通过在量化网络的每个 block 后添加一个轻量级线性补偿层来弥补量化信息损失，该补偿层参数可通过闭式解在2分钟内求得，在视觉、语言、多模态等多种任务上均显著提升了 PTQ 精度。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "网络量化"
  - "后训练量化"
  - "补偿模块"
  - "闭式解"
  - "模型压缩"
---

# Quantization without Tears

**会议**: CVPR 2025  
**arXiv**: [2411.13918](https://arxiv.org/abs/2411.13918)  
**代码**: [https://github.com/wujx2001/QwT](https://github.com/wujx2001/QwT)  
**领域**: 多模态VLM  
**关键词**: 网络量化, 后训练量化, 补偿模块, 闭式解, 模型压缩

## 一句话总结

提出 QwT（Quantization without Tears）方法，通过在量化网络的每个 block 后添加一个轻量级线性补偿层来弥补量化信息损失，该补偿层参数可通过闭式解在2分钟内求得，在视觉、语言、多模态等多种任务上均显著提升了 PTQ 精度。

## 研究背景与动机

现有网络量化方法存在"速度-精度困境"：PTQ 速度快但精度差，QAT 精度高但需要大量训练（如200个epoch）。此外，量化方法通常非常复杂，包含大量需要针对特定任务调优的超参数，且缺乏跨模型和任务的通用性。作者认为，问题的根源在于现有方法强制要求量化后的网络结构 $S^{\mathbb{Z}}$ 必须与原始结构 $S$ 完全一致。如果允许在量化网络中添加少量额外模块 $S_c$ 来补偿量化损失，就可以同时实现速度、精度、简洁性和通用性。

## 方法详解

### 整体框架

QwT 的核心范式转变：允许量化后的网络结构从 $S^{\mathbb{Z}} = S$ 变为 $S^{\mathbb{Z}} = S \cup S_c$，其中 $S_c$ 是由一组轻量线性层组成的补偿模块。流程为：(1) 使用任意 PTQ 方法量化模型；(2) 在每个 block 后添加线性补偿层 $c_i$；(3) 用闭式解设置补偿层参数。整个过程约2分钟完成，零超参数。

### 关键设计

1. **线性补偿层（Compensation Module）**:
    - 功能：弥补每个 block 量化后输出 $y^{\mathbb{Z}}$ 与原始输出 $y$ 之间的信息损失
    - 核心思路：定义 $c(x) = Wx + b$，量化后每个 block 的输出变为 $y^{\text{QwT}} = l^{\mathbb{Z}}(x^{\mathbb{Z}}) + c(x^{\mathbb{Z}})$，其中 $W \in \mathbb{R}^{d_{out} \times d_{in}}$, $b \in \mathbb{R}^{d_{out}}$
    - 设计动机：虽然单个线性层无法精确补偿非线性信息损失，但每个 block 都进行线性修正，整体补偿效果是非线性的；且线性层保证了闭式解的存在

2. **闭式解求解（Closed-form Solution）**:
    - 功能：无需反向传播即可直接求得补偿层的最优参数
    - 核心思路：信息损失度量为 $\|Y - Y^{\mathbb{Z}}\|^2$，这是经典线性回归问题，闭式解为 $W^* = (Y - Y^{\mathbb{Z}}) X^{\mathbb{Z}\top} (X^{\mathbb{Z}} X^{\mathbb{Z}\top})^{-1}$，其中 $b$ 被吸收进 $W$ 中（在 $X^{\mathbb{Z}}$ 上拼接全1行向量）
    - 设计动机：闭式解保证了极快的初始化速度（~2分钟），且零超参数，避免了 QAT 方法的大量训练开销

3. **$R^2$ 过滤机制**:
    - 功能：自动跳过补偿效果差的 block，防止精度退化
    - 核心思路：计算每个补偿模块的决定系数 $R^2$，仅当 $R^2 > 0$ 时应用闭式解初始化，否则将 $W$ 和 $b$ 设为零（即不改变该 block）
    - 设计动机：少数（<5%）block 的线性回归拟合度很低，强制补偿反而会损害精度；设零保证"至少不变差"

### 损失函数 / 训练策略

- **无需训练（PTQ模式）**：直接用512张校准图像的闭式解初始化，无反向传播
- **可选微调（QwT*模式）**：仅需1个epoch微调补偿层和分类头，即可逼近QAT精度（QAT通常需200个epoch）
- **对QAT的扩展**：QAT模型因已充分收敛，闭式解不再有效，改为零初始化后微调

## 实验关键数据

### 主实验

| 数据集/任务 | 模型 | 比特数 | PTQ基线 Top-1 | +QwT Top-1 | +QwT* Top-1 |
|------------|------|--------|--------------|-----------|-------------|
| ImageNet 分类 | ViT-B | W4A4 | 68.5 | 76.3 | 78.5 |
| ImageNet 分类 | Swin-T | W4A4 | 73.0 | 75.5 | 79.3 |
| ImageNet 分类 | DeiT-T | W4A4 | 58.2 | 61.4 | 64.8 |
| ImageNet 分类 | ResNet-50 | W4A4 | 62.3 | 68.5 | 72.5 |
| COCO 检测 | Swin-S+MaskRCNN | W4A4 | 42.6 AP | 43.1 AP | - |
| CLIP 零样本 | ViT-B/32 (V+T) | W6A6 | 29.8 | 43.5 | - |
| DiT 生成 | DiT-XL/2 | W4A8 | 6.75 FID | 6.06 FID | - |

### 消融实验

| 配置 | ViT-B W4A4 Top-1 | 说明 |
|------|-----------------|------|
| PTQ4ViT 基线 | 30.7 | 原始PTQ精度极低 |
| PTQ4ViT + QwT | 70.0 | 闭式解补偿提升~40% |
| RepQ-ViT 基线 | 68.5 | 较好的PTQ基线 |
| RepQ-ViT + QwT | 76.3 | 仍有7.8%提升 |
| Percentile 基线 (W6A6) | 56.7 | - |
| Percentile + QwT (W6A6) | 79.8 | 提升23.1% |

### 关键发现

- QwT 的额外开销极小：推理延迟仅增加约3%，模型大小增加约3%
- 在低比特（4-bit）场景下效果尤为显著，平均提升约5%
- QwT 对 CLIP 双编码器量化效果突出：V+T 同时量化到6-bit时，PTQ精度从29.8%提升至43.5%（+13.7%）
- LLaMA3-8B 的 INT4 量化中，QwT 将 WikiText2 困惑度从6.65降至6.63，常识QA平均准确率从64.90%提升至65.18%

## 亮点与洞察

- **范式创新**：打破了"量化网络结构必须与原始网络相同"的隐含假设，开辟了"量化+补偿"的新范式
- **极致简洁**：零超参数 + 闭式解 + ~2分钟完成，是目前最简单的量化精度提升方法
- **黑盒兼容**：可作为插件叠加到任意PTQ方法之上（RepQ-ViT, PTQ4ViT, GPTQ等），无需了解底层量化细节
- **跨任务通用**：同一方法在CNN、ViT、CLIP、DiT、LLaMA上均有效，覆盖分类/检测/分割/生成/NLU

## 局限与展望

- 补偿层是全连接线性层，当 $d_{in}$ 很大时参数量和计算量不可忽略（ResNet中用分组卷积缓解）
- 对QAT模型的闭式解无效，只能零初始化后微调
- CLIP双编码器量化时精度恢复虽大但仍有明显gap（43.5% vs 63.4%全精度）
- 未在更大的LLM（如70B）上验证效果

## 相关工作与启发

- 与 BRECQ、QDrop 等基于block重建的PTQ方法相比，QwT不修改量化过程本身而是"事后补偿"
- 与 GPTQ 的二阶信息加补偿矩阵方式有哲学上的相似性，但GPTQ在权重空间做，QwT在输出空间做
- 补偿模块的思想可推广到其他压缩手段（剪枝后的精度恢复、知识蒸馏中的残差学习等）

## 评分

- 新颖性: ⭐⭐⭐⭐ 量化范式的转变（允许结构变化）是重要创新，但线性补偿层本身是简单想法
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖CNN/ViT/CLIP/DiT/LLaMA五类模型和分类/检测/分割/生成/NLU五类任务，极其全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，动机论述有说服力，公式推导简洁
- 价值: ⭐⭐⭐⭐ 作为量化工具箱的通用插件有很高实用价值，但LLM上的提升较为有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MLLM-as-a-Judge for Image Safety without Human Labeling](mllm-as-a-judge_for_image_safety_without_human_labeling.md)
- [\[CVPR 2025\] MBQ: Modality-Balanced Quantization for Large Vision-Language Models](mbq_modality-balanced_quantization_for_large_vision-language_models.md)
- [\[CVPR 2025\] It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data](its_a_blind_match_towards_vision-language_correspondence_without_parallel_data.md)
- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](../../CVPR2026/multimodal_vlm/fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)
- [\[ICML 2025\] Ranked from Within: Ranking Large Multimodal Models Without Labels](../../ICML2025/multimodal_vlm/ranked_from_within_ranking_large_multimodal_models_without_labels.md)

</div>

<!-- RELATED:END -->
