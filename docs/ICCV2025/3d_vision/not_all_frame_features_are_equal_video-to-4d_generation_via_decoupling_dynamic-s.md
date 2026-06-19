---
title: >-
  [论文解读] Not All Frame Features Are Equal: Video-to-4D Generation via Decoupling Dynamic-Static Features
description: >-
  [ICCV 2025][3D视觉][4D生成] DS4D 首次提出在video-to-4D生成中沿时间轴和空间轴解耦动静态特征，通过动静态特征解耦模块（DSFD）获取动态表征，并通过时空相似性融合模块（TSSF）跨视角自适应聚合动态信息，在Consistent4D和Objaverse数据集上达到SOTA。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "4D生成"
  - "动态3D高斯"
  - "动静态特征解耦"
  - "视频到4D"
  - "时空相似性融合"
---

# Not All Frame Features Are Equal: Video-to-4D Generation via Decoupling Dynamic-Static Features

**会议**: ICCV 2025  
**arXiv**: [2502.08377](https://arxiv.org/abs/2502.08377)  
**代码**: 即将公开  
**领域**: 3D视觉  
**关键词**: 4D生成, 动态3D高斯, 动静态特征解耦, 视频到4D, 时空相似性融合

## 一句话总结

DS4D 首次提出在video-to-4D生成中沿时间轴和空间轴解耦动静态特征，通过动静态特征解耦模块（DSFD）获取动态表征，并通过时空相似性融合模块（TSSF）跨视角自适应聚合动态信息，在Consistent4D和Objaverse数据集上达到SOTA。

## 研究背景与动机

从视频生成动态3D内容（即4D生成）是计算机视觉和图形学的重要课题，在虚拟现实、游戏、影视制作中有广泛应用。然而，从少量视角准确预测运动轨迹同时保证高质量生成仍是巨大挑战。

**核心痛点**：现有方法（无论是推理型还是优化型）都直接使用整帧的全部信息来建模时空相关性，完全忽视了帧内动态区域和静态区域的区别。当静态区域占比很大时（这在现实中极为常见，如一个人在固定背景中运动），模型会严重偏向于拟合静态区域，导致动态区域的纹理细节模糊、运动信息被忽略。

**直观示例**：以一个人走路的视频为例，人体（动态区域）可能只占画面的20-30%，而背景、地面等静态区域占70-80%。传统方法在优化Gaussian表示时，主导梯度来自静态区域，导致动态区域（如衣服褶皱、手臂摆动）的纹理变得模糊不清。

**核心idea**：如果能显式地将帧特征中的动态信息和静态信息分离开来，并强化动态表征，就能避免静态区域对动态区域的"淹没"效应。DS4D正是基于这一思路，提出在时间轴上解耦动静态特征，在空间轴上跨视角融合动态信息。

## 方法详解

### 整体框架

DS4D的pipeline如下：(1) 输入单视角视频，用Zero123++生成伪多视角图像序列；(2) 用DINOv2提取帧特征；(3) 用大型重建模型从中间帧初始化3D高斯点云；(4) DSFD模块沿时间轴解耦动静态特征；(5) TSSF模块沿空间轴融合动态信息；(6) 通过Deformation MLP生成4D内容。

### 关键设计

1. **动静态特征解耦模块（DSFD）**:

    - 功能：沿时间轴将帧特征分解为动态特征和静态特征
    - 核心思路：选择中间帧特征 $f^{(t/2,j)}$（代表语义基准）和所有帧的平均特征 $\bar{f}^{(\bar{t},j)}$（代表平均运动变化）作为参考帧特征 $r^j$。然后将当前帧特征投影到参考特征方向上得到静态部分，正交分量即为动态特征：
    - 静态特征：$f_{static}^{(i,j)} = \frac{f^{(i,j)} \cdot r^j}{\|r^j\|_2} \cdot \frac{r^j}{\|r^j\|_2}$
    - 动态特征：$f_{dynamic}^{(i,j)} = f^{(i,j)} - f_{static}^{(i,j)}$
    - 最终将动态特征拼接到当前帧特征上，得到解耦特征 $f_d^{(i,j)}$
    - 设计动机：利用向量投影的几何关系，投影到参考帧方向上的分量代表"不变的部分"（静态），正交分量代表"变化的部分"（动态），这既简洁又有物理直觉

2. **时空相似性融合模块（TSSF）**:

    - 功能：从不同视角的解耦特征中自适应选取相似的动态信息进行融合
    - 核心问题：由于空间遮挡，单一视角的动态特征无法完整表达4D空间中的动态信息
    - 设计方案：首先通过视角投影将解耦特征映射到高斯点上得到点特征 $f_p^{(i,j)}$，然后沿空间轴聚合
    - **全局感知融合（GA）**：用全连接层+Softmax生成各视角的score map $\bm{W}$，加权求和所有视角的点特征：$f_a^i = \sum_{j=0}^{v} w^{(i,j)} f_p^{(i,j)}$
    - **距离感知融合（DA）**：前视角（真实输入视角）包含最准确的运动区域信息，计算其他视角与前视角点特征的L1距离，先融合其他视角（降低遮挡严重的视角权重），再与前视角特征融合
    - 设计动机：同一空间区域在不同视角下的纹理和运动相似，利用这种相似性跨视角补全单一视角的遮挡区域信息

3. **动态高斯特征与HexPlane**:

    - 功能：结合融合后的点特征和HexPlane生成的动态高斯特征
    - 核心思路：使用HexPlane对位置、缩放、旋转等高斯属性进行时空正则化。将HexPlane特征 $f_{hg}^i$ 与融合点特征 $f_a^i$ 通过可学习线性变换映射为最终的融合高斯特征
    - 设计动机：HexPlane提供场的平滑性保证，而融合点特征提供丰富的动态信息，两者互补

### 损失函数 / 训练策略

训练损失包含：SDS损失（利用预训练多视角扩散模型的先验）、光度损失（渲染视图与GT图像）和LPIPS损失（伪多视角图像与渲染视图的感知相似度）。

初始化策略：使用大型重建模型从中间帧生成点云来初始化高斯点，提供几何先验并保证拓扑稳定性（消融实验证明这比随机初始化更好）。

## 实验关键数据

### 主实验

| 方法 | 数据集 | CLIP↑ | LPIPS↓ | FVD↓ | FID-VID↓ |
|------|--------|-------|--------|------|----------|
| STAG4D | Consistent4D | 0.9078 | 0.1354 | 986.83 | 26.37 |
| SC4D | Consistent4D | 0.9117 | 0.1370 | 852.98 | 26.48 |
| **DS4D-DA** | **Consistent4D** | **0.9225** | **0.1309** | **784.02** | **24.05** |
| STAG4D | Objaverse | 0.8790 | 0.1811 | 1061.36 | 30.14 |
| SC4D | Objaverse | 0.8490 | 0.1852 | 1067.76 | 40.51 |
| **DS4D-DA** | **Objaverse** | **0.8881** | **0.1759** | **870.95** | **25.38** |

### 消融实验

| 配置 | CLIP↑ | LPIPS↓ | FVD↓ | FID-VID↓ | 说明 |
|------|-------|--------|------|----------|------|
| A. 基线 (无特征增强) | 0.9133 | 0.1341 | 953.63 | 27.37 | 无点初始化、无DSFD |
| B. + 点云初始化 | 0.9151 | 0.1313 | 913.37 | 27.14 | 几何先验提升稳定性 |
| D. + 帧特征(不解耦) | 0.9174 | 0.1350 | 888.66 | 26.85 | 有提升但易过拟合静态 |
| E. + DSFD(解耦) | 0.9186 | 0.1333 | 861.61 | 26.54 | 解耦带来进一步提升 |
| F. + TSSF(平均池化) | 0.9194 | 0.1313 | 839.66 | 26.51 | 简单平均不够好 |
| G. + TSSF-GA | 0.9206 | 0.1311 | 799.94 | 26.18 | 自适应选择更有效 |
| H. + TSSF-DA | **0.9225** | **0.1309** | **784.02** | **24.05** | 距离感知最优 |

### 关键发现

- FVD指标从基线的953.63降至784.02，说明生成结果的时序伪影大幅减少
- DSFD解耦 vs 不解耦（E vs D）：FVD从888.66降至861.61，证明显式解耦的必要性
- TSSF-DA优于TSSF-GA：距离感知融合通过降低遮挡严重的新视角的影响权重，更好地保留了前视角的真实运动信息
- 在真实场景数据集Neu3D上也展现了有效性（PSNR 32.40 vs 4D-GS的32.16）

## 亮点与洞察

- **问题定义精准**：首次明确指出4D生成中动静态区域比例不平衡导致的"过拟合静态"问题
- **方法直觉清晰**：向量投影解耦动静态的思路简洁、优雅，有明确的几何含义
- **可视化有力**：热力图清楚展示了动态特征确实捕捉到了运动区域（如大象的躯干、三角龙的腿部）
- **即插即用性**：DSFD和TSSF可以直接插入到4D-GS等现有方法中使用

## 局限与展望

- 依赖Zero123++生成的伪多视角图像质量，如果多视角生成不准确会影响解耦效果
- 当前参考帧选择策略（中间帧+平均帧）比较简单，更自适应的策略可能带来进一步提升
- 未利用光流等显式运动信息辅助解耦，可以结合光流和3D-aware基础模型的深度特征
- 仅验证了物体级别的4D生成，在大规模动态场景中的效果有待验证

## 相关工作与启发

- **STAG4D**：优化型4D方法的代表，通过时序锚点生成多视角视频，但不区分动静态
- **DreamGaussian4D**：显著减少优化时间但细节质量不足
- **HexPlane**：DS4D采用的时空正则化工具，保证4D场的平滑性
- 启发：动静态解耦的思想可以迁移到其他需要处理运动的3D任务中

## 评分

- 新颖性: ⭐⭐⭐⭐ (动静态解耦思路新颖，但方法相对直接)
- 实验充分度: ⭐⭐⭐⭐ (多数据集验证+充分消融+可视化分析)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，图表质量高)
- 价值: ⭐⭐⭐⭐ (问题定义有价值，方法可迁移)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis](gaussian_variation_field_diffusion_for_high-fidelity_video-to-4d_synthesis.md)
- [\[CVPR 2025\] Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation](../../CVPR2025/3d_vision/efficient_dynamic_scene_editing_via_4d_gaussian-based_static-dynamic_separation.md)
- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)
- [\[ICCV 2025\] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)
- [\[CVPR 2025\] Doppelgangers++: Improved Visual Disambiguation with Geometric 3D Features](../../CVPR2025/3d_vision/doppelgangers_improved_visual_disambiguation_with_geometric_3d_features.md)

</div>

<!-- RELATED:END -->
