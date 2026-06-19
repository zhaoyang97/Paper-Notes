---
title: >-
  [论文解读] One Diffusion to Generate Them All
description: >-
  [CVPR 2025][3D视觉][统一扩散模型] 提出 OneDiffusion，一个 2.8B 参数的统一扩散模型，将所有条件和目标图像建模为噪声尺度不同的帧序列，单个模型即可支持文生图、条件生成、深度估计、分割、多视图生成和 ID 定制等多种任务。 - 当前扩散模型通常为单一任务独立训练，缺乏类似 LLM 的通用性 -…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "统一扩散模型"
  - "多任务生成"
  - "流匹配"
  - "多视图生成"
  - "条件图像生成"
---

# One Diffusion to Generate Them All

**会议**: CVPR 2025  
**arXiv**: [2411.16318](https://arxiv.org/abs/2411.16318)  
**代码**: [GitHub](https://github.com/lehduong/OneDiffusion)  
**领域**: 3D视觉/统一生成  
**关键词**: 统一扩散模型, 多任务生成, 流匹配, 多视图生成, 条件图像生成

## 一句话总结

提出 OneDiffusion，一个 2.8B 参数的统一扩散模型，将所有条件和目标图像建模为噪声尺度不同的帧序列，单个模型即可支持文生图、条件生成、深度估计、分割、多视图生成和 ID 定制等多种任务。

## 研究背景与动机

- 当前扩散模型通常为单一任务独立训练，缺乏类似 LLM 的通用性
- 可控生成依赖外部模块（如 ControlNet 需要专门的条件编码器，个性化模型需要人脸识别网络和辅助损失）
- 不同任务的输入要求差异巨大：多视图生成需要处理任意输入输出视图组合和相机位姿，理解任务需要输出深度、位姿或分割图
- 现有训练方案高度针对特定任务调优，无法跨任务泛化
- 大语言模型（如 GPT-4）已展示出通用模型的价值，希望扩散模型也能实现类似的统一性
- 需要一种无需专用架构和外部损失的统一框架，支持双向的图像合成与理解

## 方法详解

### 整体框架

OneDiffusion 将所有任务统一为帧序列建模。每个样本是一组"视图"$\{\mathbf{x}_i\}_{i=1}^N$，训练时对每个视图独立采样不同的噪声时刻 $t_i$，模型学习联合速度场 $v_\theta(t_1,...,t_N, \mathbf{x}_1,...,\mathbf{x}_N)$。推理时，条件视图的时刻设为0（无噪声），目标视图从高斯噪声开始反向积分生成。通过任务标签（如 `[[text2image]]`, `[[multiview]]`）区分不同任务。

### 关键设计

**1. 变噪声尺度帧序列建模**

- **功能**：统一所有条件生成和预测任务为同一训练目标
- **核心思路**：将条件图像和目标图像视为一个序列的不同"视图"，训练时对每个视图独立采样噪声时刻 $t_i \sim \text{LogNorm}(0,1)$，前向过程为 $\mathbf{x}_i^{t_i} = t_i\mathbf{x}_i + (1-t_i)\epsilon_i$。推理时条件视图 $t_{\setminus K}=0$、目标视图 $t_K=t$ 实现条件采样
- **设计动机**：不同噪声尺度天然区分了条件和目标，无需为不同任务设计不同的条件注入机制（如 ControlNet 的零卷积或 IP-Adapter 的适配器）

**2. 统一 Plücker 射线编码的多视图方案**

- **功能**：支持多视图生成和相机位姿估计
- **核心思路**：使用 Plücker 坐标 $\bm{r}=(\bm{o}\times\bm{d}, \bm{d})$ 表示相机射线，将射线嵌入作为独立"视图"跟在图像 latent 后组成序列（而非通道拼接）。这样既可以将射线设为条件生成多视图图像，也可以将射线设为噪声来预测相机位姿
- **设计动机**：将射线嵌入作为独立视图而非通道拼接，使得视图数量 $N$ 灵活可变，并自然支持位姿估计（反向问题）

**3. 任务标签 + 丰富文本条件**

- **功能**：通过文本指定任务类型和具体条件
- **核心思路**：为每种任务预定义任务标签（如 `[[semantic2image]]`），加上描述性文本。对于语义分割，在提示中加入颜色编码和类别（如 `<#FFFF00 yellow mask: mouse>`），实现灵活的条件指定
- **设计动机**：利用文本的灵活性统一不同任务的条件描述，避免为每种条件设计专门的编码器

### 损失函数 / 训练策略

- 联合流匹配目标：$\mathcal{L}(\theta) = \mathbb{E}[\|v_\theta - u\|^2]$，其中 $u = (\mathbf{x}_1 - \epsilon_1, ..., \mathbf{x}_N - \epsilon_N)$
- 从头训练，三阶段策略：(1) T2I 预训练 $256^2$/$ 512^2$ 各 500K 步；(2) 混合任务训练 $512^2$ 1M 步；(3) T2I 高分辨率微调 $1024^2$
- 采用 Next-DiT 架构，3D RoPE 位置编码支持多分辨率
- 等概率批内采样各任务，AdamW 优化器 $\eta=0.0005$
- 训练硬件：TPU v3-256 + 64×H100

## 实验关键数据

### 主实验

GenEval 文生图基准（$1024 \times 1024$）：

| 方法 | 参数量(B) | 数据量(M) | GenEval↑ |
|------|----------|----------|---------|
| SDXL | 2.6 | - | 0.55 |
| SD3-medium | 2.0 | 1000 | 0.62 |
| FLUX-dev | 12.0 | - | 0.67 |
| FLUX-schnell | 12.0 | - | 0.71 |
| **OneDiffusion** | **2.8** | **75** | **0.65** |

### 消融实验

多视图生成任务中不同组件的影响：

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| w/o Plücker 射线 | 18.2 | 0.72 | 0.28 |
| w/o 混合任务训练 | 20.1 | 0.78 | 0.22 |
| Full OneDiffusion | **22.5** | **0.84** | **0.16** |

### 关键发现

1. 仅用 75M 训练数据就在 GenEval 上达到 0.65，接近使用 1000M+ 数据的专用模型（SD3 0.62, FLUX-dev 0.67）
2. 在多视图生成任务中性能与专攻此任务的方法相当，展示了统一训练不损失单任务能力
3. 模型可零样本泛化到训练中未见过的高分辨率
4. ID 定制可处理非人类面孔（如动漫角色），优于依赖人脸检测器的 InstantID
5. 单一模型同时支持正向（条件→图像）和逆向（图像→条件）任务

## 亮点与洞察

- 框架设计极度简洁：变噪声序列统一所有任务，无需专用模块、外部损失或适配器
- 双向能力：同一模型既能从深度图生成图像，又能从图像预测深度图，条件和目标角色完全可互换
- 多视图方案中将射线嵌入作为独立视图的设计支持灵活的输入输出组合
- 训练数据效率高，75M 数据已可与数十亿级数据的模型竞争

## 局限与展望

- 2.8B 参数在文生图上仍不及 12B 的 FLUX 系列
- 多任务训练中不同任务的数据平衡策略仍需深入研究
- 单任务精度在某些场景下可能不及专用模型
- 未来可扩展更多任务（视频生成、3D 重建等）
- 模型规模增大和更多高质量数据有望进一步提升性能

## 相关工作与启发

- **ControlNet / T2I-Adapter**: 通过外部模块支持条件生成，OneDiffusion 证明统一架构可替代这些专用模块
- **Marigold**: 微调扩散模型用于深度估计，OneDiffusion 将其作为众多任务之一统一处理
- **Stable Video Diffusion**: 视频扩散中的序列建模思想启发了 OneDiffusion 的帧序列设计
- 启发：扩散模型正朝着类似 LLM 的通用化方向发展，统一训练框架是关键路径

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 变噪声序列统一多任务的思路简洁有效
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖生成和预测的多种任务评测
- **写作质量**: ⭐⭐⭐⭐ — 框架描述清晰，任务覆盖全面
- **价值**: ⭐⭐⭐⭐⭐ — 为统一视觉生成模型提供了重要参考路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Matrix3D: Large Photogrammetry Model All-in-One](matrix3d_large_photogrammetry_model_all-in-one.md)
- [\[ICCV 2025\] RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation](../../ICCV2025/3d_vision/robotron-mani_all-in-one_multimodal_large_model_for_robotic_manipulation.md)
- [\[CVPR 2026\] Charge: A Comprehensive Novel View Synthesis Benchmark and Dataset to Bind Them All](../../CVPR2026/3d_vision/charge_a_comprehensive_novel_view_synthesis_benchmark_and_dataset_to_bind_them_a.md)
- [\[NeurIPS 2025\] SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent](../../NeurIPS2025/3d_vision/sceneweaver_all-in-one_3d_scene_synthesis_with_an_extensible_and_self-reflective.md)
- [\[CVPR 2025\] HandOS: 3D Hand Reconstruction in One Stage](handos_3d_hand_reconstruction_in_one_stage.md)

</div>

<!-- RELATED:END -->
