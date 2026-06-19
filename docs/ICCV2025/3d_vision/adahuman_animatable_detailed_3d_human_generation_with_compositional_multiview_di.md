---
title: >-
  [论文解读] AdaHuman: Animatable Detailed 3D Human Generation with Compositional Multiview Diffusion
description: >-
  [ICCV 2025][3D视觉][3D人体重建] 提出AdaHuman框架，通过姿态条件化的3D联合扩散模型和组合式3DGS细化模块，从单张图片生成高精度、可动画化的3D人体虚拟人。 从单张图片生成高质量可动画化的3D人体虚拟人对游戏、动画和VR有重要应用价值。现有方法面临两个主要挑战：（1）自遮挡问题：——虚拟人通常以输…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D人体重建"
  - "3D高斯溅射"
  - "多视角扩散模型"
  - "姿态条件生成"
  - "可动画化虚拟人"
---

# AdaHuman: Animatable Detailed 3D Human Generation with Compositional Multiview Diffusion

**会议**: ICCV 2025  
**arXiv**: [2505.24877](https://arxiv.org/abs/2505.24877)  
**代码**: [https://nvlabs.github.io/AdaHuman](https://nvlabs.github.io/AdaHuman)（代码和模型将公开）  
**领域**: 3D视觉  
**关键词**: 3D人体重建, 3D高斯溅射, 多视角扩散模型, 姿态条件生成, 可动画化虚拟人

## 一句话总结

提出AdaHuman框架，通过姿态条件化的3D联合扩散模型和组合式3DGS细化模块，从单张图片生成高精度、可动画化的3D人体虚拟人。

## 研究背景与动机

从单张图片生成高质量可动画化的3D人体虚拟人对游戏、动画和VR有重要应用价值。现有方法面临两个主要挑战：（1）**自遮挡问题**——虚拟人通常以输入图片的相同姿态生成，复杂姿态下的遮挡区域难以补全，影响绑定和动画；（2）**细节模糊**——前馈式3D重建模型受限于固定输出分辨率（如256×256），无法捕捉面部、衣服纹理等精细细节。

SDS方法虽灵活但存在过饱和伪影和慢速生成。多视角生成+重建的流水线提高了速度和真实感，但仍未解决上述两个核心问题。因此需要一种既能处理姿态变换又能增强局部细节的新框架。

## 方法详解

### 整体框架

AdaHuman包含两个核心模块：（1）**姿态条件化3D联合扩散**——在扩散过程中同步进行多视角图像合成和3DGS重建，支持任意姿态条件下的虚拟人生成；（2）**组合式3DGS细化**——通过局部身体部位的图像到图像细化和裁剪感知相机光线图，将精细化的局部3DGS无缝整合为完整虚拟人。

### 关键设计

1. **姿态条件化3D联合扩散模型**:

    - 输入全身图像后，生成不同身体部位（头部、上半身、下半身）的局部视图，与输入一起构成输入视角集合 $\mathcal{I}_{i=1}^V$
    - 每个视角由三元组 $\{x_i, p_i, c_i\}$ 表示——RGB图像、2D语义姿态图（从SMPL渲染）、相机光线图
    - 将单图像LDM的2D自注意力替换为3D注意力，实现多视角一致生成
    - 在每个去噪步骤 $t$，通过3DGS生成器 $\mathbf{G}$ 从预测的"干净"图像 $x_j^{t\to 0}$ 生成3DGS $\mathcal{G}^t$，再渲染回3D一致的图像用于下一步去噪
    - **关键优势**：姿态条件化使模型无需标准姿态训练数据即可将虚拟人生成为A-pose，最小化自遮挡，自然支持绑定和动画
    - 重建模式选择同帧的3个90°间隔目标视角；重新姿态模式选择不同帧的4个目标视角

2. **组合式3DGS细化模块**:

    - **局部身体部位细化**：从粗糙3DGS $\mathcal{G}_\text{coarse}$ 渲染4个90°分隔的标准视角，针对3个局部部位（头、上身、下身）分别渲染放大视图
    - 通过image-to-image扩散（类似SDEdit）细化局部渲染，显著增强细节
    - **裁剪感知相机光线图**：局部视图的像素坐标 $(u,v)$ 通过裁剪框参数映射回全局视图坐标 $(i,j)$，计算对应的相机光线嵌入 $\mathcal{R}(i,j) = (\mathbf{o}(i,j), \mathbf{o}(i,j) \times \mathbf{d}(i,j))$
    - 设计动机：建立局部和全局视图之间的精确3D坐标对应关系，使3DGS生成器能在全局空间中统一处理不同尺度的输入

3. **可见性感知3DGS组合**:

    - 两个关键标准决定保留哪些3D高斯点：
        - **视角覆盖度**：统计每个高斯点被多少个输入视角覆盖，低覆盖度的点缺乏多视角共识，可能不可靠
        - **可见性显著度**：测量所有渲染视角中alpha通道的梯度幅度，低显著度的点对外观贡献小，可能是噪声
    - 当某个高斯点同时被更精细部位（如头部）的视角良好覆盖时，删除来自粗糙部位的冗余点
    - 设计动机：直接合并所有局部3DGS会产生浮动伪影，此策略通过智能筛选确保只保留最可靠和视觉最重要的高斯点

### 训练策略

- 在MVHumanNet（6209个主体）和CustomHuman（589个网格）数据上联合训练
- LDM和3DGS生成器均从预训练权重初始化
- 先训练重建30K步，再微调重新姿态10K步
- LDM用MSE损失监督图像latent，3DGS用MSE+LPIPS渲染损失+表面正则化
- 除目标视角外额外采样12个视角为3DGS提供密集监督

## 实验关键数据

### 主实验（虚拟人重建）

| 模型 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | CD(cm)↓ |
|------|-------|-------|--------|------|---------|
| LGM | 18.99 | 0.8445 | 0.1664 | 122.3 | 2.175 |
| SiTH | 20.77 | 0.8727 | 0.1277 | 42.9 | 1.389 |
| SIFU | 20.59 | 0.8853 | 0.1359 | 92.6 | 2.009 |
| Human3Diffusion | 21.08 | 0.8728 | 0.1364 | 35.3 | 1.230 |
| **AdaHuman** | **21.46** | **0.8925** | **0.1087** | **27.3** | **0.962** |

用户研究中，AdaHuman相比SiTH、SIFU、Human3Diffusion、粗糙3DGS分别获得88.3%、99.2%、79.7%、93.8%的偏好率。

### 消融实验

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | 说明 |
|------|-------|-------|--------|------|------|
| 粗糙3DGS（无细化） | 20.84 | 0.8789 | 0.1296 | 31.9 | 缺乏面部等精细细节 |
| 直接组合（无过滤） | 20.41 | 0.8700 | 0.1350 | 36.2 | 产生大量浮动伪影 |
| 可学习组合（网络预测） | 20.87 | 0.8788 | 0.1270 | 28.0 | 轻微改善但仍有伪影 |
| 无联合扩散 | 20.79 | 0.8762 | 0.1283 | 27.6 | 视角不一致 |
| **完整方法** | **21.46** | **0.8925** | **0.1087** | **27.3** | 最佳 |
| +真值姿态条件 | 23.00 | 0.9028 | 0.1086 | 27.0 | 更好的姿态对齐 |

### 关键发现

- 重新姿态任务上PSNR达24.64（对比SiTH的21.21），LPIPS降至0.0863，大幅领先
- 在野外图片上对复杂宽松服装的泛化能力强
- 虽然未在标准姿态数据上训练，但利用MVHumanNet中的多样姿态分布成功泛化到A-pose
- 推理耗时约70秒（A100 GPU）

## 亮点与洞察

- **姿态解耦设计**：将虚拟人重建与姿态变换统一在一个扩散框架中，无需标准姿态的训练数据即可生成动画就绪的虚拟人
- **局部-全局组合策略**：裁剪感知光线图巧妙地在不改变3DGS生成器架构的情况下引入多尺度输入
- **两种动画模式对比**：直接重新姿态（更真实的衣服变形但慢）vs. LBS动画（实时但衣服变形有限），为不同应用场景提供选择

## 局限与展望

- 局部细化策略在手部和手臂等遮挡/低覆盖区域可能产生伪影
- 动画仍依赖SMPL模型和蒙皮权重，面部表情、手势和服装变形精度受限
- 未来可探索视频扩散模型提升动画质量和时序一致性
- 基于模拟的方法有望改善宽松服装的物理正确变形

## 相关工作与启发

- 延续Human3Diffusion的联合扩散+重建范式，核心创新在于姿态条件化和组合式细化
- 与IDOL和LHM等同期工作相比，基于扩散模型能利用更强的生成先验
- 裁剪感知光线图的设计思路可推广到其他需要多尺度3DGS重建的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 姿态条件联合扩散和可见性感知组合方案设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 定量+定性+用户研究+全面消融
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示直观
- 价值: ⭐⭐⭐⭐ 解决单图生成可动画化虚拟人的实际需求，效果显著领先

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] REPARO: Compositional 3D Assets Generation with Differentiable 3D Layout Alignment](reparo_compositional_3d_assets_generation_with_differentiable_3d_layout_alignmen.md)
- [\[ICCV 2025\] FiffDepth: Feed-forward Transformation of Diffusion-Based Generators for Detailed Depth Estimation](fiffdepth_feed-forward_transformation_of_diffusion-based_generators_for_detailed.md)
- [\[ICCV 2025\] PHD: Personalized 3D Human Body Fitting with Point Diffusion](phd_personalized_3d_human_body_fitting_with_point_diffusion.md)
- [\[ICCV 2025\] HairCUP: Hair Compositional Universal Prior for 3D Gaussian Avatars](haircup_hair_compositional_universal_prior_for_3d_gaussian_avatars.md)
- [\[ICCV 2025\] GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields](geoprog3d_compositional_visual_reasoning_for_city-scale_3d_language_fields.md)

</div>

<!-- RELATED:END -->
