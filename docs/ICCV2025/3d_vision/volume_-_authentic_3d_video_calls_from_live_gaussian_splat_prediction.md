---
title: >-
  [论文解读] VoluMe: Authentic 3D Video Calls from Live Gaussian Splat Prediction
description: >-
  [ICCV 2025][3D视觉][3D视频通话] 微软提出首个从单目2D摄像头实时预测3D高斯泼溅重建的方法，实现真实感、保真性、实时性和时序稳定性四项要求的统一，使任何人仅用标准笔记本摄像头即可进行体积3D视频通话。 领域现状 领域现状：3D虚拟会议有望提升远程沟通的共同在场感和参与度，但现有3D表示方案各有硬伤： 复杂…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D视频通话"
  - "Gaussian Splatting"
  - "单目重建"
  - "实时3D重建"
  - "数字人"
  - "视频会议"
---

# VoluMe: Authentic 3D Video Calls from Live Gaussian Splat Prediction

**会议**: ICCV 2025  
**arXiv**: [2507.21311](https://arxiv.org/abs/2507.21311)  
**领域**: 3D视觉  
**关键词**: 3D视频通话, Gaussian Splatting, 单目重建, 实时3D重建, 数字人, 视频会议

## 一句话总结

微软提出首个从单目2D摄像头实时预测3D高斯泼溅重建的方法，实现真实感、保真性、实时性和时序稳定性四项要求的统一，使任何人仅用标准笔记本摄像头即可进行体积3D视频通话。

## 研究背景与动机

### 领域现状

**领域现状**：3D虚拟会议有望提升远程沟通的共同在场感和参与度，但现有3D表示方案各有硬伤：

**复杂硬件方案**（如Google Project Starline）：需要多相机、专用传感器、强算力，成本极高

**Avatar方案**（如Meta Codec Avatars）：需要预先注册（enrollment）建模用户外观，无法实时适应当前时刻的穿着、发型、眼镜等变化

**生成模型方案**（如Live 3D Portrait / TriPlaneNet）：基于EG3D生成模型反演，受限于模型训练数据，纹理和几何都不够真实

**视频通话的四个核心需求**：

### 现有痛点

**现有痛点**：保真性(Authenticity)**：从原始相机视角渲染时必须忠实还原输入视频（包括眼镜、配饰、发型等每个细节）

### 核心矛盾

**核心矛盾**：真实感(Realism)**：在新视角下生成合理逼真的3D重建

### 解决思路

**解决思路**：实时性(Live)**：在消费级设备上实时运行

### 补充说明

**补充说明**：稳定性(Stability)**：时间序列和视角变换时无闪烁

现有方法无一同时满足这四项要求。特别是保真性——avatar方法使用过去注册的固定外观，无法反映"此刻"的状态。

## 方法详解

### 整体框架

基于Splatter Image架构，输入单张2D图像，通过U-Net回归每个像素对应的3D高斯参数，直接输出整个3D高斯泼溅场景表示。相比原版Splatter Image，引入四项关键改进。

### 关键设计1：平面单应性校正

视频通话中人脸常出现在画面边缘，透视畸变严重。方法使用平面单应变换将人脸对齐到图像中心正面位置再输入网络，消除透视畸变对重建质量的影响。

### 关键设计2：距离自适应缩放

用户与摄像头的距离变化很大。在训练时引入缩放步骤，使用人脸检测框的大小估计距离，对最终高斯的深度和尺度做相应调整，确保远近都能正确重建。

### 关键设计3：双高斯预测

每个像素输出两个3D高斯核（而非一个），显著提升重建质量。第一个负责主要表面，第二个可以捕捉次表面细节（如头发的半透明效果、镜片反射等）。

### 关键设计4：稳定性损失

逐帧独立预测会导致时序闪烁。引入稳定性损失：对连续两帧的3D高斯重建，在多个视角下渲染并计算帧间差异，惩罚不必要的时序变化。这使得网络在保持保真性的同时，学会输出时间上一致的重建。

### 损失函数

- **重建损失**：在多个训练视角下渲染高斯并与GT图像计算 $\ell_1$ + SSIM + LPIPS 损失
- **稳定性损失**：连续帧的渲染差异最小化
- **仅使用合成数据训练**：完全在合成渲染的人脸数据上训练，但通过前馈网络的skip connection机制可以很好地泛化到真实图像

### 网络架构

轻量化U-Net：比原始SongUNet更小的backbone，每分辨率仅两层卷积，五个分辨率级别维持大感受野，去除自注意力层。保持实时推理能力（30FPS）。

## 实验关键数据

### 视觉质量

论文报告在Ava256和Cafca数据集上达到SOTA的PSNR和LPIPS指标：
- 在PSNR和LPIPS上超越所有现有方法
- 在Ava256的时序抖动(jitter)指标上取得最优
- 从原始相机视角渲染时保真性（authenticity）远超avatar类方法

### 保真性对比

与Live 3D Portrait和TriPlaneNet（基于EG3D）的定性对比中，本文方法忠实还原了输入画面的所有细节（眼镜、帽子、多样发型等），而基于生成模型的方法受限于训练数据分布，常出现眼睛追踪相机、纹理失真等问题。

### 实时系统演示

展示了完整的一对一3D视频通话系统：
- 输入：标准2D RGB摄像头
- 输出：带运动视差的3D显示
- 帧率：30 FPS
- 设备：消费级PC
- 视角范围：输入正面±40°

### 关键发现

- 纯合成数据训练的模型可以很好泛化到真实图像——前馈网络的skip connection是关键
- 双高斯比单高斯显著提升质量（并发工作也独立验证了这点）
- 稳定性损失对视频序列至关重要，显著降低时序闪烁
- 不需要360°全方位重建——视频通话场景只需正面±55°即可

## 亮点与洞察

1. **"保真性"概念的提出**：将authenticity作为3D视频通话的核心需求形式化，与生成式方法形成鲜明对比——用户期望的是"此刻的我"而非"建模后的我"
2. **极致实用主义**：每个设计选择都服务于实际部署（轻量U-Net、省略自注意力、合成数据训练避免隐私问题）
3. **合成数据泛化的成功范例**：通过前馈架构的skip connection和直接采样机制，输入信息可直接流向输出表示，避免了生成模型的重建偏差
4. **问题定义的精准**：不追求360°重建，准确定义了视频通话场景的实际需求范围

## 局限与展望

- 仅支持±40°视角（视频通话足够，但不适合更大范围的3D展示）
- 背面/侧面区域的重建质量依赖geometry prior，可能不够真实
- 缓存全文不完整，部分定量对比数据未获取
- 合成数据训练的泛化性在极端光照/遮挡条件下可能退化
- 多人视频通话场景未讨论

## 相关工作

- **Splatter Image**：本文的基础架构，U-Net直接预测高斯泼溅，但原始版本不针对人脸且不处理视频稳定性
- **Project Starline (Google)**：高端多相机3D通信方案，视角范围55°，需要专用硬件
- **Codec Avatars (Meta)**：photoreal avatar，需要预先enrollment，外观固定
- **Live 3D Portrait**：基于EG3D生成模型的实时3D人脸，受限于生成模型的表达能力
- **TriPlaneNet**：将单图映射为EG3D隐码，重建受限于EG3D训练分布
- **GS-LRM / Xu et al. / Zou et al.**：更重的transformer架构或多视图扩散，无法实时

## 评分

- **创新性**: ⭐⭐⭐⭐ — 关键改进（单应校正、双高斯、稳定性损失）实用但偏增量
- **技术深度**: ⭐⭐⭐⭐ — 系统设计全面但单个模块深度一般
- **实验充分性**: ⭐⭐⭐☆ — 缓存不完整；从可见内容看数据集和指标覆盖合理但缺少用户研究
- **实用价值**: ⭐⭐⭐⭐⭐ — 直接可部署的3D视频通话系统，商业前景明确
- **总体推荐**: ⭐⭐⭐⭐ — 工程导向的扎实工作，"保真性"视角对3D通信领域有启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] FluidNexus: 3D Fluid Reconstruction and Prediction from a Single Video](../../CVPR2025/3d_vision/fluidnexus_3d_fluid_reconstruction_and_prediction_from_a_single_video.md)
- [\[ICCV 2025\] Boosting Multi-View Indoor 3D Object Detection via Adaptive 3D Volume Construction](boosting_multiview_indoor_3d_object_detection_via_adaptive_3.md)
- [\[ICCV 2025\] Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs](compression_of_3d_gaussian_splatting_with_optimized_feature_planes_and_standard_.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[ICCV 2025\] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)

</div>

<!-- RELATED:END -->
