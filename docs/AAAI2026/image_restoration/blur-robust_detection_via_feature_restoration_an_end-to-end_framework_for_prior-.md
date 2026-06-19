---
title: >-
  [论文解读] Blur-Robust Detection via Feature Restoration: An End-to-End Framework for Prior-Guided Infrared UAV Target Detection
description: >-
  [AAAI 2026][图像恢复][运动模糊] 提出 JFD3 端到端双分支框架，在特征域而非图像域进行去模糊，并利用频率结构先验引导检测网络，实现运动模糊条件下红外无人机目标的高精度实时检测。 - 红外无人机检测的实际需求：红外无人机目标 (IRUT) 检测在全天候监控和侦察任务中至关重要，但传感器的快速运动常导致严重的运…
tags:
  - "AAAI 2026"
  - "图像恢复"
  - "运动模糊"
  - "红外无人机目标检测"
  - "特征域去模糊"
  - "频率结构引导"
  - "端到端联合框架"
---

# Blur-Robust Detection via Feature Restoration: An End-to-End Framework for Prior-Guided Infrared UAV Target Detection

**会议**: AAAI 2026  
**arXiv**: [2511.14371](https://arxiv.org/abs/2511.14371)  
**代码**: [IVPLaboratory/JFD3](https://github.com/IVPLaboratory/JFD3)  
**领域**: 图像复原  
**关键词**: 运动模糊, 红外无人机目标检测, 特征域去模糊, 频率结构引导, 端到端联合框架  

## 一句话总结
提出 JFD3 端到端双分支框架，在特征域而非图像域进行去模糊，并利用频率结构先验引导检测网络，实现运动模糊条件下红外无人机目标的高精度实时检测。

## 研究背景与动机

- **红外无人机检测的实际需求**：红外无人机目标 (IRUT) 检测在全天候监控和侦察任务中至关重要，但传感器的快速运动常导致严重的运动模糊
- **运动模糊对检测的破坏性影响**：IRUT 本身特征微弱、嵌于复杂背景中，运动模糊进一步削弱目标与背景间的对比度，使判别性特征提取更加困难
- **直接检测策略的缺陷**：在模糊图像上直接运行检测器会导致频繁的漏检和误检，因为模糊严重降低了目标的可区分性
- **分离式流水线的瓶颈**：先去模糊再检测的两阶段方案存在三个问题：(1) 深度去模糊网络计算复杂、延迟高；(2) 去模糊以视觉质量为优化目标，不面向检测任务；(3) 可能引入对检测有害的噪声
- **已有联合方法的局限**：现有联合低级与高级视觉任务的研究主要针对可见光场景的雾天退化，红外运动模糊检测问题几乎未被探索
- **核心洞察**：应在特征域而非图像域进行去模糊，让恢复过程直接服务于检测目标，同时利用频率域的结构先验弥补模糊导致的边缘细节丢失

## 方法详解

### 整体框架
JFD3 采用训练时双分支、推理时单分支的架构。训练阶段包含共享权重的清晰分支和模糊分支，清晰分支为模糊分支提供特征级监督；推理阶段仅保留模糊分支以实现高效推理。基础检测器采用 DEIM (D-FINE-N)。

### 关键设计一：Feature-Domain Deblurring (FDD) 网络
- **功能**：在特征域而非图像域对模糊特征进行恢复，输出语义增强的特征表示
- **核心思路**：基于 MIMO-UNet 构建轻量编码器-解码器结构（基础通道数降至 2，每阶段仅 2 个残差块），用清晰分支的特征作为监督目标，对模糊分支的编码器用 L1 损失对齐特征，对解码器用 SSIM 损失保持结构一致性
- **设计动机**：传统图像域去模糊计算开销大且产生冗余视觉细节，特征域恢复直接调整特征分布、减少域偏移，仅增加 0.02M 参数，适合实时部署

### 关键设计二：Frequency Structure Guidance Module (FSGM)
- **功能**：从去模糊网络提取高频结构先验，注入到检测 backbone 的 stem 与 stage1 之间，补偿模糊导致的结构细节缺失
- **核心思路**：由两个子模块组成——(1) FFRB：对结构先验做 FFT，用可学习高通滤波器提取高频成分，再通过空间感知通道注意力 (SCA) 和通道感知空间注意力 (CSA) 进行精炼；(2) SPIB：用交叉注意力将精炼后的结构先验融合到检测特征图中，采用多尺度动态卷积核 (5×5 和 7×7) 实现层次化结构引导
- **设计动机**：模糊红外图像中小目标的边界结构退化严重，频率域高频成分恰好包含边缘和纹理信息；部分压缩式注意力避免全局压缩丢失小目标关键线索

### 关键设计三：Feature Consistency Self-Supervised (FCSS) Loss
- **功能**：在双分支检测 backbone 的 4 个 stage 之间施加特征一致性约束
- **核心思路**：使用余弦相似度度量清晰分支与模糊分支中间特征的对齐程度，$\mathcal{L}_{\text{FCSS}} = \frac{1}{4}\sum_{i=1}^{4}(1 - \cos(\mathbf{F}_C^{(i)}, \mathbf{F}_B^{(i)}))$
- **设计动机**：驱动模糊分支逼近清晰分支的特征表示，弥合退化输入与干净输入之间的表示差距，提升模糊分支自身的特征提取能力

## 损失函数与训练策略

总损失为三项加权求和：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + 0.4\mathcal{L}_{\text{deb}} + 0.2\mathcal{L}_{\text{FCSS}}$。其中 $\lambda_2$ 在 20 epoch 后退火至 0.01，使模型在后期集中优化检测精度。训练 150 epochs，采用 AdamW 优化器，RTX 3090 GPU。

## 实验关键数据

### 数据集
构建 IRBlurUAV 基准：30,000 对合成模糊/清晰图像 (IRBlurUAV-syn，8:1:1 划分) + 4,118 张真实模糊图像 (IRBlurUAV-real，仅测试)，图像尺寸 640×512。

### IRBlurUAV-syn 主实验 (Table 1)

| 方法 | 策略 | AP50 | AR50 | AP | AR | Params(M) | FPS |
|------|------|------|------|----|----|-----------|-----|
| RT-DETR | Direct | 0.716 | 0.811 | 0.369 | 0.400 | 19.0 | 50.2 |
| D-FINE | Direct | 0.722 | 0.795 | 0.347 | 0.382 | 3.5 | 45.8 |
| DeepRFT+RT-DETR | Separate | 0.673 | 0.749 | 0.284 | 0.329 | 36.1 | 9.6 |
| DREB-Net | Joint | 0.710 | 0.754 | 0.300 | 0.357 | 34.6 | 10.9 |
| **JFD3 (Ours)** | **Joint** | **0.767** | **0.850** | **0.428** | **0.458** | **3.5** | **25.7** |

JFD3 在 AP 上超过最强基线 RT-DETR 达 +5.9%，同时参数量仅 3.5M，FPS 达 25.7 满足实时需求。

### IRBlurUAV-real 泛化性实验 (Table 2)

| 方法 | AP50 | AR50 | AP | AR |
|------|------|------|----|----|
| D-FINE | 0.514 | 0.693 | 0.151 | 0.190 |
| DREB-Net | 0.520 | 0.619 | 0.143 | 0.196 |
| **JFD3 (Ours)** | **0.623** | **0.730** | **0.251** | **0.291** |

在真实模糊场景中 JFD3 的 AP 领先第二名 +10.0%，证明了强泛化能力。

### 消融实验 (Table 3)

| FDD | FSGM | AP50 | AP |
|-----|------|------|----|
| ✗ | ✗ | 0.654 | 0.290 |
| ✓ | ✗ | 0.763 | 0.390 |
| ✓ | ✓ | 0.765 | 0.420 |

FDD 贡献最大 (+10.0% AP)，FSGM 进一步提升 AP +3.0%，两者互补。

## 亮点

- **首个针对红外运动模糊的端到端联合检测框架**，填补了该领域空白
- **特征域去模糊**的设计范式非常巧妙：仅 0.02M 额外参数即可大幅提升检测精度，避免了图像域去模糊的高计算成本
- 频率结构引导模块利用 FFT 高通滤波 + 双注意力机制精炼结构先验，针对小目标边缘退化问题对症下药
- 双分支训练、单分支推理的策略兼顾了训练时知识迁移和推理时效率
- 构建了包含合成与真实场景的 IRBlurUAV 基准数据集（34K+ 图像），为社区提供了评测工具

## 局限与展望

- 合成数据采用简单线性运动轨迹模拟模糊，与真实非均匀运动模糊可能存在域差
- 仅在自建数据集上验证，缺乏在其他公开红外数据集或可见光模糊数据集上的跨域评估
- 推理 FPS 为 25.7 (RTX 3090)，在嵌入式平台上能否达到实时仍有疑问
- 双分支共享权重的设计对清晰分支特征质量有强依赖，若训练数据清晰图像质量不佳可能影响效果
- 未探讨与图像域去模糊联合使用时的最优策略（Table 4 显示两者互补，但未充分挖掘）

## 相关工作

- **图像去模糊**：DeepRFT (CNN)、MDT (Transformer)、EVSSM (Mamba)、MaIR 等方法追求视觉质量但计算开销大，且未面向检测任务优化
- **红外 UAV 目标检测**：MSHNet、PConv 等针对清晰图像设计，UniCD 考虑了非均匀性退化但未处理运动模糊
- **联合去模糊与检测**：DREB-Net 提出双流融合架构但面向可见光车辆检测，未考虑红外小目标的纹理稀缺和结构退化挑战

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首次将特征域去模糊与红外 UAV 检测联合端到端优化，频率引导模块设计新颖
- 实验充分度: ⭐⭐⭐⭐ — 合成+真实双数据集验证，多策略对比全面，消融实验完整
- 写作质量: ⭐⭐⭐⭐ — 结构清晰、三种策略的对比图直观，公式推导详细
- 价值: ⭐⭐⭐⭐ — 填补红外运动模糊检测空白，轻量设计有工程部署价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Generic Event Boundary Detection via Denoising Diffusion (DiffGEBD)](../../ICCV2025/image_restoration/generic_event_boundary_detection_via_denoising_diffusion.md)
- [\[CVPR 2026\] Similarity-Consistent Likelihood Diffusion enables Hidden Person Detection from Wall Reflections](../../CVPR2026/image_restoration/similarity-consistent_likelihood_diffusion_enables_hidden_person_detection_from_.md)
- [\[CVPR 2025\] A Physics-Informed Blur Learning Framework for Imaging Systems](../../CVPR2025/image_restoration/a_physics-informed_blur_learning_framework_for_imaging_systems.md)
- [\[CVPR 2026\] From Events to Clarity: The Event-Guided Diffusion Framework for Dehazing](../../CVPR2026/image_restoration/from_events_to_clarity_the_event-guided_diffusion_framework_for_dehazing.md)
- [\[CVPR 2026\] Zero-Shot Image Denoising via Hybrid Prior-Guided Pseudo Sample Generation](../../CVPR2026/image_restoration/zero-shot_image_denoising_via_hybrid_prior-guided_pseudo_sample_generation.md)

</div>

<!-- RELATED:END -->
