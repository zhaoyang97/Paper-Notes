---
title: >-
  [论文解读] Learning to Generate Conditional Tri-Plane for 3D-Aware Expression Controllable Portrait Animation
description: >-
  [ECCV 2024][3D视觉][人像动画] 提出 Export3D，通过对比预训练获取与外观解耦的表情表示（CLeBS），结合表情自适应层归一化（EAdaLN）直接生成条件tri-plane，实现无外观交换的跨身份3D-aware人像表情动画。 - 人像动画的核心需求： 虚拟人服务（跨语言配音、虚拟化身、视频会议等）要求…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "人像动画"
  - "Tri-plane"
  - "表情控制"
  - "对比学习"
  - "3DMM"
---

# Learning to Generate Conditional Tri-Plane for 3D-Aware Expression Controllable Portrait Animation

**会议**: ECCV 2024  
**arXiv**: [2404.00636](https://arxiv.org/abs/2404.00636)  
**代码**: [项目主页](https://export3d.github.io)  
**领域**: 3D视觉  
**关键词**: 人像动画, Tri-plane, 表情控制, 对比学习, 3DMM

## 一句话总结

提出 Export3D，通过对比预训练获取与外观解耦的表情表示（CLeBS），结合表情自适应层归一化（EAdaLN）直接生成条件tri-plane，实现无外观交换的跨身份3D-aware人像表情动画。

## 研究背景与动机

- **人像动画的核心需求**: 虚拟人服务（跨语言配音、虚拟化身、视频会议等）要求在保持源身份的同时转移驱动表情，但跨身份表情转移极具挑战
- **图像变形方法的局限**: 主流2D方法（FOMM、TPSMM）依赖图像变形估计运动，但面部表情作为局部运动易被全局头部运动淹没，且表情与外观高度纠缠
- **3DMM表情参数的缺陷**: 3DMM的表情参数 $\beta \in \mathbb{R}^{64}$ 包含隐含的外观信息，直接使用会导致跨身份转移时出现外观交换（如眼型、脸型变化）
- **3D形变场的问题**: HiDe-NeRF、NOFA等方法通过预测逐点形变场来控制表情，但会产生视频级伪影（闪烁、光照变化）
- **解耦学习的不稳定性**: DPE通过循环一致性学习解耦姿态和表情，但训练不稳定导致时序不一致
- **GAN反转的身份丢失**: 基于EG3D的方法通过潜在编码控制表情，但风格潜在码缺乏编码空间信息和个体细节的能力

## 方法详解

### 整体框架

Export3D 包含两个阶段：（1）对比预训练阶段——在视频数据集上训练CLeBS表情编码器，获取与外观解耦的表情表示；（2）主模型训练——使用混合tri-plane生成器，以源图像和驱动表情参数为输入，通过EAdaLN注入表情信息，生成表情转移后的tri-plane，经体积渲染和超分辨率输出最终图像。

### 关键设计

#### 模块一：对比学习基缩放（CLeBS）

从同一视频采样正负样本对，利用对比学习使表情编码器 $f_e(\cdot)$ 学习与外观无关的表情表示。对比损失为：

$$\mathcal{L}_{cl} = -\log\left(\frac{\exp(\cos(f_I(X_k), f_e(\beta_k))/\tau)}{\sum_{j \neq k} \exp(\cos(f_I(X_j), f_e(\beta_k))/\tau)}\right)$$

其中 $f_I(\cdot)$ 为图像编码器，$\tau$ 为温度参数。由于所有样本来自同一视频（相同外观），此目标函数强制编码器丢弃外观信息。

关键创新在于设计了正交基结构的表情子空间。通过QR分解获得正交基 $V = \{v_1, v_2, \ldots, v_n\} \subseteq \mathbb{R}^d$，将表情参数转换为低维系数并缩放正交基：

$$\beta' = f_e(\beta) = \lambda_1 v_1 + \lambda_2 v_2 + \cdots + \lambda_n v_n \in \mathbb{R}^d$$

其中 $\lambda = (\lambda_1, \ldots, \lambda_n) \in \mathbb{R}^n$（$n \ll 64$），$\langle v_i, v_j \rangle = \delta_{ij}$，不同表情沿正交方向独立控制。

#### 模块二：表情自适应层归一化（EAdaLN）

在ViT块的多头自注意力和Mix-FFN之前注入表情信息，通过缩放和偏移调制视觉token：

$$\text{EAdaLN}(x, \beta'_D) = \sigma(\beta'_D) \times \text{LN}(x) + \mu(\beta'_D) \in \mathbb{R}^{h \times \frac{HW}{2^8}}$$

其中 $\sigma(\beta'_D)$ 和 $\mu(\beta'_D)$ 为从精炼表情参数计算的缩放和偏移因子。相比交叉注意力，EAdaLN将表情作为全局调制信号而非逐位置query，更适合表情这种全局语义属性。

#### 模块三：混合Tri-plane生成器

结合ViT和卷积层构建生成器。源图像通过卷积块编码为视觉特征，经patch merge转换为token，通过EAdaLN-ViT块处理后，使用pixel shuffle上采样并经高斯低通滤波消除网格伪影，最终输出表情转移后的tri-plane：

$$\text{T}_{\beta_D}(S) = \mathbf{G}(S, \beta_D) \in \mathbb{R}^{3 \times 32 \times \frac{H}{2} \times \frac{W}{2}}$$

Tri-plane通过投影和聚合得到特征：$F_{\beta_D}(S) = \frac{1}{3}(F_{\beta_D,xy} + F_{\beta_D,yz} + F_{\beta_D,zx})$，再由MLP解码为颜色和密度进行体积渲染。

### 损失函数 / 训练策略

- CLeBS预训练完成后冻结，不参与后续训练；图像编码器 $f_I$ 预训练后丢弃
- 使用在线EMA对tri-plane进行稳定化，将 $T_{EMA}$ 加到生成的tri-plane上
- 渲染低分辨率图像 $\hat{D}_{raw} \in \mathbb{R}^{3 \times H/4 \times W/4}$，通过超分辨率模块获得最终分辨率
- 使用平面卷积块而非风格调制卷积进行超分辨率（因不使用风格潜在码）
- Pixel shuffle后应用高斯低通滤波消除token拼接的网格伪影

## 实验关键数据

### 主实验

VFHQ数据集上的定量比较：

| 方法 | PSNR↑ | SSIM↑ | AKD↓ | CSIM↑(同ID) | AED↓(同ID) | CSIM↑(跨ID) | AED↓(跨ID) |
|------|-------|-------|------|------------|----------|------------|----------|
| StyleHEAT | 14.23 | 0.428 | 30.41 | 0.464 | 0.161 | 0.505 | 0.242 |
| DPE | 23.24 | 0.750 | 3.66 | 0.831 | 0.083 | 0.586 | 0.253 |
| HiDe-NeRF† | 21.23 | 0.728 | 8.25 | 0.867 | 0.106 | 0.707 | 0.255 |
| **Ours** | **23.56** | 0.704 | **3.45** | 0.811 | **0.082** | 0.694 | **0.208** |

（†表示仅在前景面部区域评估）

### 消融实验

表情编码方式的消融：

| 方法 | PSNR↑ | CSIM↑(同ID) | AED↓(同ID) | CSIM↑(跨ID) | AED↓(跨ID) |
|------|-------|------------|----------|------------|----------|
| Direct 3DMM | 23.08 | 0.789 | 0.105 | 0.648 | 0.209 |
| E2E LeBS (n=25) | 23.11 | 0.745 | 0.109 | 0.670 | 0.218 |
| E2E LeBS (n=10) | 23.24 | 0.751 | 0.110 | 0.672 | 0.238 |
| E2E LeBS (n=5) | 22.63 | 0.658 | 0.140 | 0.632 | 0.246 |
| **CLeBS (Full)** | **23.56** | **0.811** | **0.082** | **0.694** | **0.208** |

EAdaLN vs 交叉注意力：EAdaLN在CSIM（0.811 vs 0.678）和AED（0.082 vs 0.125）上全面优于交叉注意力。

### 关键发现

- t-SNE可视化清晰显示原始3DMM表情参数按身份聚类（外观纠缠），CLeBS处理后聚类消失
- 单独LeBS（无对比预训练）无法解耦外观和表情，减少基向量数仅同时减弱两者
- 对比预训练是关键——同视频采样策略使同外观样本的表情可以被有效区分
- 正交基结构使不同表情方向（如眨眼、嘴唇运动）可以独立线性控制
- EAdaLN优于交叉注意力，因为表情是全局调制信号，不需要位置级的精细关注

## 亮点与洞察

- **外观-表情解耦的新范式**: 不依赖循环一致性或显式标注，而是通过同视频对比学习自然实现解耦，训练稳定且效果显著
- **正交基设计**: 将3DMM的正交结构思想引入学习的表情空间，通过QR分解保证基的正交性，实现可解释的表情方向控制
- **EAdaLN的设计哲学**: 表情作为全局语义信号通过归一化层注入，比交叉注意力更适合这种"调制"而非"选择"的需求

## 局限与展望

- 依赖3DMM提取表情参数，对遮挡或极端角度可能提取不准确
- 超分辨率依赖卷积上采样，可能引入模糊，未探索扩散模型等更强的超分方案
- 训练数据VFHQ的规模和多样性有限，泛化到极端表情或非正面人脸可能不理想
- 未探索音频驱动场景，可扩展CLeBS与语音-表情映射结合

## 相关工作与启发

- **DiT的条件注入思路**: EAdaLN直接借鉴了扩散Transformer的自适应归一化设计，证实其在GAN框架下同样有效
- **LIA的正交基思想**: 本文将LIA的正交运动字典扩展到表情空间，并通过对比学习进一步精炼
- 对虚拟人/数字人领域的启示：外观-表情解耦是实时虚拟通信的核心需求，CLeBS提供了一个轻量且有效的预训练方案

## 评分

⭐⭐⭐⭐ — 对外观-表情纠缠问题提出了优雅的解决方案，对比预训练+正交基+EAdaLN的组合设计新颖，跨身份表情转移效果优异。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] HyPlaneHead: Rethinking Tri-plane-like Representations in Full-Head Image Synthesis](../../NeurIPS2025/3d_vision/hyplanehead_rethinking_tri-plane-like_representations_in_full-head_image_synthes.md)
- [\[ECCV 2024\] Learning 3D-Aware GANs from Unposed Images with Template Feature Field](learning_3d-aware_gans_from_unposed_images_with_template_feature_field.md)
- [\[ECCV 2024\] SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding](sceneverse_scaling_3d_vision-language_learning_for_grounded_scene_understanding.md)
- [\[CVPR 2026\] EG-3DVG: Expression and Geometry Aware Grounding Decoder for 3D Visual Grounding](../../CVPR2026/3d_vision/eg-3dvg_expression_and_geometry_aware_grounding_decoder_for_3d_visual_grounding.md)
- [\[ECCV 2024\] Repaint123: Fast and High-Quality One Image to 3D Generation with Progressive Controllable Repainting](repaint123_fast_and_high-quality_one_image_to_3d_generation_with_progressive_con.md)

</div>

<!-- RELATED:END -->
