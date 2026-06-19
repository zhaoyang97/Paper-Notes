---
title: >-
  [论文解读] Image Demoiréing in RAW and sRGB Domains
description: >-
  [ECCV 2024][图像恢复][图像去摩尔纹] 提出RRID框架联合利用RAW和sRGB双域数据进行图像去摩尔纹，设计了带GFM（门控反馈）和FSM（频域选择）的SCDM去摩尔纹模块，以及RGISP实现设备相关ISP学习辅助颜色恢复，在PSNR上超越SOTA 0.62dB。 领域现状：用智能手机拍摄屏幕内容已成为日常操作…
tags:
  - "ECCV 2024"
  - "图像恢复"
  - "图像去摩尔纹"
  - "RAW域处理"
  - "ISP学习"
  - "频域滤波"
  - "双域融合"
---

# Image Demoiréing in RAW and sRGB Domains

**会议**: ECCV 2024  
**arXiv**: [2312.09063](https://arxiv.org/abs/2312.09063)  
**代码**: [https://github.com/rebeccaeexu/RRID](https://github.com/rebeccaeexu/RRID)  
**领域**: 其他  
**关键词**: 图像去摩尔纹, RAW域处理, ISP学习, 频域滤波, 双域融合

## 一句话总结
提出RRID框架联合利用RAW和sRGB双域数据进行图像去摩尔纹，设计了带GFM（门控反馈）和FSM（频域选择）的SCDM去摩尔纹模块，以及RGISP实现设备相关ISP学习辅助颜色恢复，在PSNR上超越SOTA 0.62dB。

## 研究背景与动机

**领域现状**：用智能手机拍摄屏幕内容已成为日常操作，但相机CFA（颜色滤波阵列）与屏幕LCD子像素之间的频率混叠会产生摩尔纹（moiré patterns），严重影响画质。现有去摩尔纹方法主要在sRGB域操作，代表方法包括DMCNN（多分辨率CNN）、MopNet（边缘引导+模式属性）、WDNet（小波变换双分支）、FHDMi（两阶段方法）、ESDNet（超高清轻量模型）等。

**现有痛点**：在sRGB域做去摩尔纹的效果有限，原因在于ISP中的非线性操作（如demosaicing）会进一步恶化RAW域中原本就存在的摩尔纹。因此一些研究（如RDNet、RawVDmoiré）主张在RAW域进行去摩尔纹。然而仅依赖RAW数据存在一个严重问题——RAW到sRGB的转换（ISP过程）存在不确定性，仅用RAW数据无法获取准确的颜色校正信息，导致输出图像出现明显的**色偏（color cast）**。

**核心矛盾**：RAW域去摩尔纹效果好但会产生色偏；sRGB域保留了颜色信息但摩尔纹更难去除。单独使用任一域都有先天缺陷——这是一个"鱼和熊掌不可兼得"的问题。

**本文目标** 如何同时利用RAW域和sRGB域的互补优势：（1）RAW域提供更原始、摩尔纹更弱的信号用于去纹；（2）sRGB域提供设备ISP生成的颜色参考用于色彩恢复；（3）学习设备相关的ISP完成RAW到sRGB的准确转换。

**切入角度**：现代智能手机和DSLR相机可以同时输出RAW和sRGB图像（如iPhone 15 Pro、华为P60 Pro），因此RAW-sRGB配对数据在实际场景中是可获取的。作者利用这个实际条件，提出同时以RAW和sRGB作为输入，让模型学习一个设备相关的ISP来辅助颜色恢复。

**核心 idea**：联合RAW（去纹更优）和sRGB（颜色更准）双域数据，通过针对性设计的去摩尔纹模块和可学习ISP，同时实现纹路去除和颜色校正。

## 方法详解

### 整体框架
RRID的输入是配对的RAW图像 $\mathbf{I}_{raw} \in \mathbb{R}^{H/2 \times W/2 \times 4}$（packed RGGB）和sRGB图像 $\mathbf{I}_{rgb} \in \mathbb{R}^{H \times W \times 3}$。系统包含三个核心阶段：（1）浅层特征提取——分别用卷积+DCAB从RAW和sRGB提取特征，sRGB经过下采样对齐分辨率；（2）SCDM去摩尔纹——分别对RAW特征（配GFM）和sRGB特征（配FSM）进行多尺度去摩尔纹，获得预去纹特征 $\mathbf{D}_{raw}$ 和 $\mathbf{D}_{rgb}$；（3）RGISP完成RAW→sRGB的颜色转换——利用 $\mathbf{D}_{rgb}$ 的颜色信息指导 $\mathbf{D}_{raw}$ 的颜色空间转换；（4）4个RSTB（Residual Swin Transformer Block）做全局色调映射和细节精修，输出最终的去摩尔纹sRGB图像。

### 关键设计

1. **Skip-Connection-based Demoiréing Module (SCDM) + GFM/FSM**:

    - 功能：分别在RAW和sRGB分支进行针对性的多尺度去摩尔纹
    - 核心思路：SCDM基于多尺度U-Net架构，以DCAB（Dilated Channel Attention Block）为基本构建单元——DCAB使用多个扩张卷积层扩大感受野，结合通道注意力自适应缩放特征。核心创新在于skip connection中嵌入了针对性的去纹模块：**GFM（Gated Feedback Module）** 用于RAW分支，通过特征门控机制自适应区分纹理细节和摩尔纹。具体地，将中间特征沿通道维度分为 $\mathbf{F}_{gate}$ 和 $\mathbf{F}_{content}$，通过GELU激活的gate对content做点乘选择性保留。**FSM（Frequency Selection Module）** 用于sRGB分支，利用可学习带阻滤波器在频域抑制摩尔纹——在8×8 Block DCT域中，用卷积层学习自适应的频率选择性衰减
    - 设计动机：将去纹模块放在skip connection而非主干网络中有两大好处：（1）效率——FSM中的Block DCT计算量大，放在主干会导致推理时间暴增至4.6秒，放在skip connection仅0.089秒；（2）信息流更好——skip connection传递的是编码器的多尺度特征，在这里做去纹可以同时保留高频细节和去除摩尔纹

2. **RGB Guided ISP (RGISP)**:

    - 功能：利用sRGB预去纹特征的颜色信息，学习设备相关的ISP完成RAW特征到sRGB域的颜色转换
    - 核心思路：借鉴传统ISP中矩阵变换进行颜色空间转换的原理，RGISP使用转置交叉注意力（transposed cross-attention）实现。从RAW特征 $\mathbf{D}_{raw}$ 生成Query $\mathbf{Q}$ 和Key $\mathbf{K}$，从sRGB特征 $\mathbf{D}_{rgb}$ 生成Value $\mathbf{V}$。计算变换矩阵 $\mathbf{M} = \text{Softmax}(\mathbf{Q} \cdot \mathbf{K}^T / \lambda) \in \mathbb{R}^{C \times C}$，然后 $\mathbf{D}_{out} = \mathbf{M} \cdot \mathbf{V}$。$\mathbf{M}$ 实际上学习了一个通道级的颜色转换矩阵，全局共享（类比ISP中全局的白平衡和色彩空间设置），辅以深度卷积和点卷积做局部细节修正
    - 设计动机：相比直接拼接RAW和sRGB特征或使用自注意力，交叉注意力机制能更好地让sRGB的颜色信息"指导"RAW特征的颜色转换。实验中RGISP比自注意力和传统的RRM方法高出0.3-0.5dB

3. **DCAB（Dilated Channel Attention Block）**:

    - 功能：SCDM的基础构建块，提供大感受野特征编解码
    - 核心思路：DCAB由一系列扩张卷积层（不同扩张率）+ ReLU激活 + 通道注意力机制组成，并配有残差连接。扩张卷积在不增加参数量的情况下扩大感受野，这对检测和去除不同尺度的摩尔纹至关重要。通道注意力自适应重标定各通道的重要性
    - 设计动机：摩尔纹具有多尺度特性——不同频率干涉产生不同尺度的纹路。扩张卷积是比简单堆叠卷积层更高效的扩大感受野方式

### 损失函数 / 训练策略
总损失为RAW域和sRGB域L1损失的加权和：$\mathcal{L} = \alpha \|\hat{\mathbf{Y}}_{raw} - \mathbf{Y}_{raw}\|_1 + \|\hat{\mathbf{Y}}_{rgb} - \mathbf{Y}_{rgb}\|_1$，其中 $\alpha=0.5$。使用AdamW优化器（$\beta_1=0.9, \beta_2=0.999$），多步学习率调度，初始学习率 $2 \times 10^{-4}$，训练500 epochs，batch size 80，4块RTX 3090。

## 实验关键数据

### 主实验（TMM22数据集）

| 方法 | 输入类型 | PSNR | SSIM | LPIPS | 参数量(M) | 推理时间(s) |
|------|---------|------|------|-------|-----------|-----------|
| DMCNN | sRGB | 23.54 | 0.885 | 0.154 | 1.55 | 0.052 |
| ESDNet | sRGB | 26.77 | 0.927 | 0.089 | 5.93 | 0.115 |
| RDNet | RAW | 26.16 | 0.921 | 0.091 | 6.04 | 1.094 |
| RawVDmoiré | RAW | 27.26 | 0.935 | 0.075 | 5.33 | 0.182 |
| **RRID（本文）** | **sRGB+RAW** | **27.88** | **0.938** | **0.079** | **2.38** | **0.089** |

### 消融实验

| 配置 | PSNR | SSIM | 说明 |
|------|------|------|------|
| B6: Full RRID | **27.88** | **0.938** | 完整模型 |
| B1: w/o RAW输入和RAW分支 | 25.79 | 0.915 | PSNR暴降2dB |
| B2: w/o sRGB输入和sRGB分支 | 27.24 | 0.929 | 产生明显色偏 |
| B4: w/o RGISP | 27.38 | 0.932 | 颜色校正能力下降 |
| S1: w/o GFM & FSM | 27.00 | 0.926 | 去纹能力显著下降 |
| S5: 全用FSM替代GFM | 27.32 | 0.930 | 域特定设计优于统一设计 |

### 关键发现
- RAW输入的贡献极大——去掉RAW分支PSNR暴降超过2dB，证实了RAW域去摩尔纹的优越性
- 仅用RAW（B2）虽然去纹效果好但色偏严重，加入sRGB（完整模型）同时解决两个问题
- GFM和FSM的domain-specific设计（GFM给RAW、FSM给sRGB）优于互换或统一使用——交换后PSNR下降0.3-0.5dB
- 将去纹模块放在skip connection（0.089s）比放在主干（4.6s）快50倍且效果更好
- RGISP的交叉注意力比自注意力和传统RRM分别高出0.34dB和0.67dB
- 在纯sRGB数据集FHDMi上（去掉RAW分支），RRID仍取得次优成绩，展示了泛化能力

## 亮点与洞察
- **双域互补的设计理念**：RAW的物理优势（12-14bit、无ISP非线性处理、摩尔纹更弱）和sRGB的颜色优势完美互补，通过RGISP的交叉注意力实现了"最优组合"。这种"各取所长"的多输入设计思路可迁移到其他RAW+sRGB联合任务
- **skip connection做去纹的效率设计**：将计算密集的FSM（Block DCT）放在skip connection而非主路径，推理时间从4.6秒降到0.089秒——这种"重模块放在支路"的架构设计trick非常实用
- **可学习带阻滤波器替代频率先验**：摩尔纹在频域有特定的频率特征但难以手工提取。用卷积层在Block DCT域学习自适应的带阻滤波器，既利用了频域处理的优势又避免了手工设计的局限

## 局限与展望
- 训练数据TMM22分辨率仅256×256，限制了模型学习全局颜色映射的能力——作者承认在严重色偏场景下颜色恢复仍不完美
- 当前仅有TMM22一个RAW-sRGB配对去摩尔纹数据集，更大规模更高分辨率的数据集是亟需的
- FSM中Block DCT的块大小固定为8×8，更自适应的块大小选择可能对不同频率的摩尔纹更有效
- RGISP学习的是通道级颜色转换，对局部色偏（如不同屏幕区域颜色温度不同）的处理能力有限
- 未与最新的DSDNet（2025年提出的RAW域去摩尔纹方法）进行对比

## 相关工作与启发
- **vs RDNet**: RAW域去摩尔纹的开创之作，仅使用RAW输入+预训练ISP做颜色转换，本文证明了联合sRGB域输入的显著优势
- **vs RawVDmoiré**: RAW域视频去摩尔纹，取得次优性能。RRID通过加入sRGB分支同时超越其PSNR并保持更快的推理速度
- **vs ESDNet**: 专为超高清sRGB图像设计的轻量去摩尔纹，参数量是RRID的2.5倍但PSNR低1.1dB
- **vs CR3Net**: 同样使用RAW+sRGB配对数据，但用于去反射而非去摩尔纹。CR3Net在去摩尔纹任务上表现不佳（23.75dB），而RRID为27.88dB

## 评分
- 新颖性: ⭐⭐⭐⭐ RAW+sRGB双域联合去摩尔纹是首创，RGISP学习设备ISP的设计有洞察
- 实验充分度: ⭐⭐⭐⭐⭐ 三个层面的消融（架构/输入、SCDM、RGISP），跨数据集泛化验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，每个模块的设计动机交代充分
- 价值: ⭐⭐⭐⭐ 开辟了RAW+sRGB联合图像复原的新方向，实际应用前景明确

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Edit-aware RAW Reconstruction](../../CVPR2026/image_restoration/edit-aware_raw_reconstruction.md)
- [\[CVPR 2026\] PNG: Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning](../../CVPR2026/image_restoration/diffusion-based_srgb_real_noise_generation_via_prompt-driven_noise_representatio.md)
- [\[CVPR 2025\] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction with Application to Strong Lens Discovery](../../CVPR2025/image_restoration/polishing_the_sky_wide-field_and_high-dynamic_range_interferometric_image_recons.md)
- [\[CVPR 2026\] RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution](../../CVPR2026/image_restoration/rawdomain_degradation_models_smartphone_sr.md)
- [\[CVPR 2026\] RawMetaDiff: Unlocking Extreme Darkness from Dual-Exposure RAW with Meta-Guided Diffusion](../../CVPR2026/image_restoration/rawmetadiff_unlocking_extreme_darkness_from_dual-exposure_raw_with_meta-guided_d.md)

</div>

<!-- RELATED:END -->
