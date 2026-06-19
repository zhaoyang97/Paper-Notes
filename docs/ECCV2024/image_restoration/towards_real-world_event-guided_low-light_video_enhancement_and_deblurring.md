---
title: >-
  [论文解读] Towards Real-world Event-guided Low-light Video Enhancement and Deblurring
description: >-
  [ECCV 2024][图像恢复][事件相机] 本文首次提出事件相机引导的低光视频增强与去模糊联合任务，构建了基于分光棱镜的真实世界数据集 RELED，并设计了包含事件引导可变形时序对齐 (ED-TFA) 和频谱滤波跨模态增强 (SFCM-FE) 两个核心模块的端到端框架，在 PSNR 上比此前最佳方法提升 1.2dB 以上。
tags:
  - "ECCV 2024"
  - "图像恢复"
  - "事件相机"
  - "低光增强"
  - "运动去模糊"
  - "跨模态融合"
  - "时序对齐"
---

# Towards Real-world Event-guided Low-light Video Enhancement and Deblurring

**会议**: ECCV 2024  
**arXiv**: [2408.14916](https://arxiv.org/abs/2408.14916)  
**代码**: [https://github.com/intelpro/ELEDNet](https://github.com/intelpro/ELEDNet)  
**领域**: 图像修复 / 低光视频增强与去模糊  
**关键词**: 事件相机, 低光增强, 运动去模糊, 跨模态融合, 时序对齐

## 一句话总结

本文首次提出事件相机引导的低光视频增强与去模糊联合任务，构建了基于分光棱镜的真实世界数据集 RELED，并设计了包含事件引导可变形时序对齐 (ED-TFA) 和频谱滤波跨模态增强 (SFCM-FE) 两个核心模块的端到端框架，在 PSNR 上比此前最佳方法提升 1.2dB 以上。

## 研究背景与动机

在低光环境下，帧相机为获得足够亮度需要延长曝光时间，这带来两个同时出现的退化：**低可见度**和**运动模糊**。这两个问题在现有研究中通常被分开处理——低光增强和运动去模糊各有专门的方法和数据集，但级联处理两个任务往往产生次优结果。

**核心矛盾**：同时解决低光和模糊的联合问题极具挑战性：
1. 仅靠帧信息，当运动模糊严重或光照极低时，帧中几乎没有可用的运动和结构信息；
2. 现有低光+去模糊联合方法 (如 LEDNet) 依赖合成数据，无法泛化到真实场景；
3. 缺乏包含真实低光模糊图像、正常光照清晰图像和事件流的同步数据集。

**事件相机的优势**为联合解决两个问题提供了可能：其**高动态范围** (HDR) 特性使其在低光下仍能捕获场景细节；其**高时间分辨率**使其能在长曝光期间精确记录运动信息。

但**事件相机本身的问题**也不容忽视：在极低光照下，事件也会产生大量噪声。因此需要一种能有效利用事件优势、同时抑制噪声的跨模态融合方法——这正是 SFCM-FE 模块的设计出发点。

## 方法详解

### 整体框架

整体流程为：输入连续三帧低光模糊视频 $\{B^{t-1}, B^t, B^{t+1}\}$ 和对应的事件体素网格 $\{E^{t-1}, E^t, E^{t+1}\}$。首先通过卷积层分别提取帧特征和事件特征，然后：
1. **ED-TFA 模块**：利用事件引导进行多尺度可变形时序对齐，生成对齐特征；
2. **SFCM-FE 模块**：在频域利用低通滤波器增强跨模态特征的主要结构信息，抑制噪声；
3. **UNet 解码器**：从增强的多尺度特征金字塔生成最终的正常光照清晰输出。

### 关键设计

1. **RELED 数据集与三轴分光棱镜相机系统**:

    - 功能：构建首个包含真实低光模糊图像、正常光照清晰图像和事件流的同步三模态数据集。
    - 核心思路：使用两个分光棱镜将入射光分为三路：第一路相机短曝光 (2ms) 捕获正常光照清晰图像，第二路加 1/32 ND 滤光片 + 长曝光 (16ms) 模拟真实低光模糊，第三路连接事件相机。通过微控制器实现硬件级同步，通过单应性矩阵校准多相机间的微小偏移。
    - 设计动机：现有数据集要么使用 gamma 校正/ZeroDCE 合成低光（与真实场景差距大），要么使用合成模糊（如 GoPro 的帧平均），缺乏同步的真实事件流。RELED 数据集的分辨率为 1024×768，涵盖 42 个城市场景，质量远超此前的合成数据集和低分辨率 DAVIS 数据。

2. **事件引导可变形时序特征对齐 (ED-TFA)**:

    - 功能：利用事件信息引导多帧间的时序对齐，从相邻帧中提取有用的运动信息。
    - 核心思路：首先用 transposed attention Transformer 编码器分别提取帧特征金字塔 $\{\mathcal{F}(B^k)_s\}$ 和事件特征金字塔 $\{\mathcal{F}(E^k)_s\}$。然后在多个尺度上，将帧和事件特征拼接生成模板特征，通过可变形卷积 (DCN) 进行对齐：
    $\mathcal{F(T)}_s^{t+1 \to t} = \mathcal{D}(\mathcal{F(T)}_s^{t+1 \to t}, \mathcal{O}_s^{t+1 \to t}, \mathcal{M}_s^{t+1 \to t})$
      采用粗到细策略：低分辨率下估计粗略偏移，逐步上采样传递到高分辨率进行精细调整。前向和后向两个方向分别对齐后合并。
    - 设计动机：在低光模糊条件下，仅用帧信息找帧间对应关系极度 ill-posed——帧中充斥噪声且结构不清。事件相机的高动态范围和高时序分辨率恰好弥补了这一缺陷。多尺度粗到细设计确保了鲁棒性——低分辨率时亚像素偏移小容易对齐，高分辨率时依赖之前的偏移估计逐步精细化。

3. **频谱滤波跨模态特征增强 (SFCM-FE)**:

    - 功能：通过频域低通滤波增强跨模态特征的结构信息，同时抑制低光噪声。
    - 核心思路：将对齐帧特征、事件特征和上一尺度输出拼接后，分为两个分支：
        - **(a) 低通滤波分支**：对特征做 FFT 变换到频域，施加高斯低通滤波器 $\mathcal{P}(x,y,\sigma) = \exp\left(-\frac{(x-x_c)^2 + (y-y_c)^2}{2\sigma^2}\right)$ 提取低频结构信息，再通过 FFC (Fast Fourier Convolution) 块进一步频率选择，最后 IFFT 回空间域。低频滤波后的特征再经过**逐像素空间动态滤波器**增强空间变化的主要结构。
        - **(b) 原始分支**：保留未经频域滤波的特征。
        - 最终通过**空间注意力**加权融合两分支：$\mathcal{G}(X)^{(c)} = \mathcal{G}(\bar{X})_L^{(a)} \odot \sigma(\text{Conv}(\cdot)) + \mathcal{G}(\tilde{X})^{(b)} \odot \sigma(\text{Conv}(\cdot))$。
    - 设计动机：低光场景下帧和事件**都有严重噪声**，直接做跨模态特征融合效果不佳（消融实验中 EFNet 融合甚至降低 0.23dB）。噪声主要集中在高频，而场景的主要结构在低频。低通滤波天然地抑制高频噪声、保留结构信息。再辅以逐像素动态滤波器适应空间变化的结构差异，最后残差连接保留原始信息。

### 损失函数 / 训练策略

- 采用多尺度输出 $\{S_s^t\}, s \in \{0, 1, 2\}$ 的监督训练。
- 所有方法在 RELED 数据集上训练 200 个 epoch。
- 提供轻量版 Ours-s (5.3MB) 和标准版 Ours (12.8MB)。

## 实验关键数据

### 主实验（RELED 数据集）

| 方法类别 | 方法 | PSNR | SSIM | 参数量 (MB) |
|---------|------|------|------|------------|
| 帧-低光增强 | LLFormer | 26.62 | 0.862 | 13.15 |
| 帧-去模糊 | DSTNet | 29.59 | 0.903 | 7.53 |
| 帧-联合方法 | LEDNet | 26.47 | 0.856 | 7.41 |
| 事件-去模糊 | REFID | 30.10 | 0.913 | 15.9 |
| 事件-去模糊 | UEVD | 29.93 | 0.905 | 27.88 |
| **本文 Ours-s** | **ELEDNet-s** | **30.98** | **0.919** | **5.3** |
| **本文 Ours** | **ELEDNet** | **31.30** | **0.925** | **12.8** |

关键对比：
- 比最佳事件引导方法 REFID 提升 **+1.20 dB PSNR**
- 比唯一联合方法 LEDNet 提升 **+4.83 dB PSNR**
- 轻量版 Ours-s 仅 5.3MB 参数即超越所有其他方法

### 消融实验

| 配置 | PSNR | 贡献 |
|------|------|------|
| Baseline (无 ED-TFA, 无 SFCM-FE) | 29.59 | - |
| + ED-TFA | 30.78 | +1.19 dB |
| + SFCM-FE | 30.40 | +0.81 dB |
| + ED-TFA + SFCM-FE (Full) | **31.30** | +1.71 dB |

SFCM-FE 模块内部消融：

| 组件 | PSNR | 说明 |
|------|------|------|
| 仅 CABs | 30.81 | +0.03 dB，几乎无效 |
| CABs + SA | 30.79 | +0.01 dB，单纯堆叠无效 |
| SA + LPF 分支 | 31.22 | +0.44 dB，低通滤波是核心 |
| CABs + SA + LPF | **31.30** | +0.52 dB |

跨模态融合方式对比：

| 融合方式 | PSNR | 变化 |
|----------|------|------|
| 无融合 | 30.78 | - |
| EFNet 融合 | 30.55 | -0.23 dB（反而下降） |
| REFID 融合 | 30.86 | +0.08 dB |
| SFCM-FE (本文) | **31.30** | +0.52 dB |

### 关键发现

- 事件引导方法整体大幅优于纯帧方法，验证了事件相机在低光退化下的核心优势。
- EFNet 的跨模态融合在低光场景中反而降低性能，说明简单融合无法处理双模态都含大量噪声的情况。
- 低通滤波分支是 SFCM-FE 性能提升的主要来源（+0.44 dB），验证了频域噪声抑制策略的有效性。
- ED-TFA 和 SFCM-FE 对最终性能的贡献是互补的。

## 亮点与洞察

- **开创性任务定义**: 首次将事件引导的低光增强和去模糊统一为联合任务，填补了研究空白。
- **真实世界数据集**: RELED 使用分光棱镜实现真正同步的三模态采集，质量和规模远超合成方法。
- **频域噪声抑制的精巧设计**: 在低光条件下帧和事件都有强噪声，SFCM-FE 用低通滤波提取结构信息是非常有针对性的解决方案。
- **轻量版也强**: 5.3MB 的 Ours-s 即超越所有对比方法，说明方法设计本身的有效性而非单纯堆参数。

## 局限与展望

- RELED 数据集规模有限（42 场景，29 训练 + 13 测试），泛化能力待验证。
- 分光棱镜系统成本高、体积大，限制了数据采集的便利性。
- 仅处理 3 帧输入，更长序列的时序建模可能进一步提升质量。
- 高斯低通滤波的标准差 σ 为固定值，自适应频率选择可能更优。
- 未探索事件表征方式的影响（除 voxel grid 外还有 event frame、time surface 等）。

## 相关工作与启发

- 与 EFNet、REFID 等事件引导去模糊方法不同，本文解决的是更复杂的联合低光+去模糊问题。
- SFCM-FE 中的频域滤波思路可推广到其他跨模态融合场景，特别是当多个模态都含大量噪声时。
- 分光棱镜数据采集系统的设计思路对构建其他多模态配对数据集有参考价值。
- 事件引导的粗到细可变形对齐策略对一般视频修复任务也有启发。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次定义联合任务，真实数据集和频域跨模态融合都有新意
- 实验充分度: ⭐⭐⭐⭐ 消融全面深入，但数据集规模和场景多样性有限
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述逻辑完整
- 价值: ⭐⭐⭐⭐ 开创性任务+真实数据集，对事件视觉社区有较大推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Event-Illumination Collaborative Low-light Image Enhancement with a High-resolution Real-world Dataset](../../CVPR2026/image_restoration/event-illumination_collaborative_low-light_image_enhancement_with_a_high-resolut.md)
- [\[ECCV 2024\] Unrolled Decomposed Unpaired Learning for Controllable Low-Light Video Enhancement](unrolled_decomposed_unpaired_learning_for_controllable_low-light_video_enhanceme.md)
- [\[ICCV 2025\] Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)](../../ICCV2025/image_restoration/low-light_image_enhancement_using_event-based_illumination_estimation.md)
- [\[ECCV 2024\] Domain-Adaptive Video Deblurring via Test-Time Blurring](domain-adaptive_video_deblurring_via_test-time_blurring.md)
- [\[CVPR 2026\] TextOVSR: Text-Guided Real-World Opera Video Super-Resolution](../../CVPR2026/image_restoration/textovsr_text-guided_real-world_opera_video_super-resolution.md)

</div>

<!-- RELATED:END -->
