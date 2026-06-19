---
title: >-
  [论文解读] Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars
description: >-
  [ICCV 2025][人体理解][3D高斯] 提出Avat3r——首个可动画的大型3D重建模型(LRM)，仅需4张输入图像即可在前馈方式下回归出高质量可驱动的3D高斯头部头像，通过整合DUSt3R位置图和Sapiens语义特征作为先验、并用简单的cross-attention建模表情动画，在Ava256和NeRSemble数据集上大幅超越现有方法。
tags:
  - "ICCV 2025"
  - "人体理解"
  - "3D高斯"
  - "头部头像重建"
  - "面部动画"
  - "大重建模型"
  - "前馈推理"
---

# Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars

**会议**: ICCV 2025  
**arXiv**: [2502.20220](https://arxiv.org/abs/2502.20220)  
**代码**: 无公开代码  
**领域**: 人体理解  
**关键词**: 3D高斯, 头部头像重建, 面部动画, 大重建模型, 前馈推理

## 一句话总结
提出Avat3r——首个可动画的大型3D重建模型(LRM)，仅需4张输入图像即可在前馈方式下回归出高质量可驱动的3D高斯头部头像，通过整合DUSt3R位置图和Sapiens语义特征作为先验、并用简单的cross-attention建模表情动画，在Ava256和NeRSemble数据集上大幅超越现有方法。

## 研究背景与动机
创建照片级3D头部头像在远程呈现、电影制作、个性化游戏等领域需求旺盛，但现有方法各有严重局限：
- **Studio级优化方法**（如URAvatar）需要多视角录制+昂贵的测试时优化（8×A100训练3小时），不适合消费级场景
- **单目视频重建**（如FlashAvatar）会过拟合训练视角，新视角外推能力差
- **3D感知肖像动画**（如GPAvatar、GAGAvatar）主要聚焦正面渲染，牺牲3D一致性换取图像质量
- **照片级3D人脸模型**（如GPHM、HeadGAP）受限于训练数据身份数量（仅几百人），难以学习人脸外观的完整分布

**核心观察**：3D人脸数据在身份轴上有限（仅几百人），但在表情轴上数据充足（每人数千帧不同表情）。因此可以设计一个条件于身份（由输入图像提供）、仅在表情轴上泛化的系统，避免学习人脸外观的全分布。这一"只在数据充足的轴泛化"的设计哲学是本文的核心创新思路。

## 方法详解

### 整体框架
输入4张带相机参数的图像 + 目标表情编码 $z_{exp}$ → DUSt3R生成位置图 $I^{pos}$ + Sapiens提取语义特征图 $I^{feat}$ → 图像/位置/射线拼接后patchify → Vision Transformer主干（self-attention做跨视角匹配 + cross-attention注入表情信息）→ 上采样为每像素高斯属性图 $M$ → 位置/颜色skip connection → 置信度阈值过滤 → 输出3D高斯集合 $\mathcal{G}$，可从任意视角渲染。整个流程是纯前馈推理，无需测试时优化。

### 关键设计
1. **基础模型先验注入（DUSt3R + Sapiens）**: DUSt3R预测稠密位置图作为每个高斯的粗略3D位置初始化，通过skip connection加到最终位置预测上（$M^{pos} \leftarrow M^{pos}+I^{pos}$）。Sapiens 2b模型提取低分辨率语义特征图，通过GridSample对齐分辨率后与token拼接，简化后续Transformer的跨视角匹配任务。两者均离线预计算以节省训练开销。关键发现：DUSt3R在输入不一致（不同表情）时仍能产生合理的位置图。

2. **可动画的大型重建模型架构**: 采用GRM风格的Vision Transformer。输入图像 $I$、位置图 $I^{pos}$、Plücker射线坐标 $I^{pluck}$ 拼接后patchify为token。核心由8层self-attention（跨视角匹配）+ 8层cross-attention（表情注入）组成。表情编码通过MLP投影为长度 $S=4$ 的token序列 $f_{exp} \in \mathbb{R}^{S \times D}$，cross-attention让每个图像token关注表情序列。令人惊讶的发现是：仅用简单的cross-attention就足以建模复杂的面部动画，无需显式变形场或模板mesh。

3. **置信度过滤与自适应高斯数量**: 利用DUSt3R的置信度图（阈值 $\tau=0.5$）过滤低置信度像素，自然决定高斯数量。蓬松头发的人会生成更多高斯，光头的人更少——类似前景分割的自适应效果。颜色也有skip connection（$M^{rgb} \leftarrow M^{rgb}+I$），提供"目标颜色应接近输入像素颜色"的归纳偏置。

4. **不一致输入训练策略**: 训练时4张输入图像采样自不同时间步（不同表情），而非传统LRM要求的同表情多视角。DUSt3R在输入不一致时仍能产生合理结果。这一策略不仅允许在更大的单目视频数据集上训练和推理，还使模型对手机拍摄时的意外移动更鲁棒。

### 损失函数 / 训练策略
- **损失函数**：$\mathcal{L} = 0.8 \mathcal{L}_{l1} + 0.2 \mathcal{L}_{ssim} + 0.01 \mathcal{L}_{lpips}$，LPIPS在3M步后才引入（避免过早关注高频细节）
- **训练数据**：Ava256数据集，244人训练/12人测试，80个相机视角，每人约5000帧，512×512裁剪
- **训练配置**：Adam, lr=5e-5, batch size=1/GPU × 8×A100, 共3.5M步约4天
- **监督策略**：每个batch=4张随机表情输入 + 8张目标表情监督视角
- **视角采样**：k-farthest viewpoint sampling确保输入视角分布合理且多样

## 实验关键数据

### 主实验

**Few-shot(4张输入) 头部头像创建**:

| 数据集 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | AKD↓ | CSIM↑ |
|--------|------|-------|-------|--------|------|-------|
| Ava256 | HeadNeRF | 9.1 | 0.64 | 0.52 | 6.9 | 0.11 |
| Ava256 | InvertAvatar | 14.2 | 0.36 | 0.55 | 15.8 | 0.29 |
| Ava256 | GPAvatar | 19.4 | 0.69 | 0.34 | 5.3 | 0.31 |
| Ava256 | **Avat3r** | **20.7** | **0.71** | **0.33** | **4.8** | **0.59** |
| NeRSemble | HeadNeRF | 9.8 | 0.69 | 0.47 | 4.9 | 0.22 |
| NeRSemble | GPAvatar† | 17.6 | 0.67 | 0.40 | 5.7 | 0.07 |
| NeRSemble | **Avat3r** | **20.5** | **0.75** | **0.33** | **3.7** | **0.50** |

CSIM（身份相似度）从GPAvatar的0.31提升到0.59，说明Avat3r生成的头像在身份保持上有质的飞跃。在未见过的NeRSemble上泛化能力同样强劲。

**运行时分析**:

| 方法 | 创建时间(s)↓ | 驱动速度(fps)↑ |
|------|-------------|---------------|
| HeadNeRF | 6511 | 1 |
| GPAvatar | 0.2 | 9.5 |
| **Avat3r(4-shot)** | **12.3** | **7.9** |
| Avat3r(1-shot) | 1.15 | 53 |

### 消融实验

| 配置 | PSNR↑ | AKD↓ | 说明 |
|------|-------|------|------|
| w/o DUSt3R | 21.1 | 8.31 | 几何保真度下降，多视角预测对齐困难 |
| w/o Sapiens | 20.9 | 8.08 | 清晰度下降，尤其头发区域 |
| w/o 随机时间步训练 | 21.2 | 8.86 | 图像略清晰但对不一致输入脆弱 |
| w/o 位置skip | 21.39 | - | 对齐问题和模糊 |
| w/o 颜色skip | 21.76 | - | 色偏 |
| **完整模型** | **22.05** | **8.08** | 所有组件协同最优 |

### 关键发现
- DUSt3R和Sapiens分别贡献几何先验和语义先验，两者互补不可替代
- 不一致输入训练虽微降图像清晰度（PSNR 22.05→21.2），但对鲁棒性至关重要（AKD 8.08→8.86，即面部关键点距离增大）
- 增加984个仅中性表情的身份仅增0.08%数据量即可改善跨身份泛化
- 单图推理（1-shot）速度53fps远快于4-shot的7.9fps，因为高斯数量少3/4

## 亮点与洞察
- **首个可动画的大型3D重建模型**：将LRM范式首次扩展到可驱动的3D头部头像领域
- **极简的动画机制**：仅用cross-attention到表情编码序列就实现了复杂的面部动画，无需模板mesh或显式变形场
- **不固定高斯数量**：每像素预测高斯+置信度过滤，自适应调节不同人的高斯密度
- **"只在某一轴泛化"的设计哲学**值得借鉴：数据在身份轴不足但表情轴充足时，条件化于身份、泛化于表情
- 强大的泛化能力：在未见过的NeRSemble数据集，甚至能动画化AI生成图像和古代雕像

## 局限与展望
- 单图推理依赖3D GAN做3D lifting，引入误差累积
- 需要相机位姿作为输入——错误的位姿估计会导致重建偏差
- 光照效果被烘焙到重建中，无法重光照，限制虚拟环境中的应用
- 训练数据仅244人，存在身份过拟合风险
- 创建时间12.3s主要受限于DUSt3R推理（而非Transformer本身），可通过轻量化DUSt3R加速

## 相关工作与启发
- **vs GPAvatar**: 在NeRF上预测canonical TriPlane再用FLAME驱动，表情被FLAME空间约束；Avat3r用cross-attention直接学习表情映射，CSIM 0.31→0.59
- **vs FlashAvatar**: 需要完整视频且严重过拟合训练视角；Avat3r仅需4张图即远超（PSNR 15.0→20.5 on NeRSemble）
- **vs HeadGAP/GPHM**: 学习人脸完整分布但受限于训练身份数量；Avat3r条件于输入图像绕过身份泛化问题
- **vs URAvatar**: 8×A100训练3小时 vs Avat3r 12.3秒前馈推理，实用性天壤之别
- cross-attention建模动态的极简方案可能适用于手势、身体动作等其他动态重建任务
- DUSt3R对不一致输入的鲁棒性这一发现可推广到其他3D重建场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个将LRM扩展为可动画头部重建模型，设计简洁但是已有组件的巧妙组合
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、丰富消融、单图/多图/应用场景全覆盖
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，动机阐述充分，每个设计选择都有实验支撑
- 价值: ⭐⭐⭐⭐ 实用价值高（手机拍4张→分钟级可驱动头像），对数字人领域有推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] ImHead: A Large-scale Implicit Morphable Model for Localized Head Modeling](imhead_a_large-scale_implicit_morphable_model_for_localized_head_modeling.md)
- [\[CVPR 2025\] RGBAvatar: Reduced Gaussian Blendshapes for Online Modeling of Head Avatars](../../CVPR2025/human_understanding/rgbavatar_reduced_gaussian_blendshapes_for_online_modeling_of_head_avatars.md)
- [\[NeurIPS 2025\] VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image](../../NeurIPS2025/human_understanding/vasa-3d_lifelike_audio-driven_gaussian_head_avatars_from_a_single_image.md)
- [\[CVPR 2026\] 4DSurf: High-Fidelity Dynamic Scene Surface Reconstruction](../../CVPR2026/human_understanding/textit4dsurf_high-fidelity_dynamic_scene_surface_reconstruction.md)
- [\[ICCV 2025\] GGTalker: Talking Head Synthesis with Generalizable Gaussian Priors and Identity-Specific Adaptation](ggtalker_talking_head_systhesis_with_generalizable_gaussian_priors_and_identity-.md)

</div>

<!-- RELATED:END -->
