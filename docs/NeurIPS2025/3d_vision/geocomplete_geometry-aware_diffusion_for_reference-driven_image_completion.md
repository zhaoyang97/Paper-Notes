---
title: >-
  [论文解读] GeoComplete: Geometry-Aware Diffusion for Reference-Driven Image Completion
description: >-
  [NeurIPS 2025][3D视觉][图像补全] 提出 GeoComplete，通过将投影点云作为几何条件注入双分支扩散模型，并结合 target-aware masking 策略，实现几何一致的参考驱动图像补全，PSNR 提升 17.1%。 参考驱动图像补全（Reference-driven image complet…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "图像补全"
  - "扩散模型"
  - "几何引导"
  - "点云投影"
  - "参考图像"
---

# GeoComplete: Geometry-Aware Diffusion for Reference-Driven Image Completion

**会议**: NeurIPS 2025  
**arXiv**: [2510.03110](https://arxiv.org/abs/2510.03110)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 图像补全, 扩散模型, 几何引导, 点云投影, 参考图像

## 一句话总结

提出 GeoComplete，通过将投影点云作为几何条件注入双分支扩散模型，并结合 target-aware masking 策略，实现几何一致的参考驱动图像补全，PSNR 提升 17.1%。

## 研究背景与动机

参考驱动图像补全（Reference-driven image completion）利用同一场景的其他视角图像来修复目标图像中的缺失区域。当目标视角与参考视角差异较大时，这一任务尤其具有挑战性。

现有方法的局限：

**传统几何方法**（TransFill、GeoFill）：采用位姿估计→深度重建→3D变形→补丁融合→图像调和的串行流水线，早期错误会级联放大，在遮挡、动态内容或模糊几何场景中容易失败

**生成方法**（RealFill）：基于 LoRA 微调扩散模型直接合成缺失区域，但**缺乏几何线索**（如相机位姿、深度），当目标视角与参考差异大时会产生幻觉结构或错位内容
3. 核心矛盾：需要同时具备**生成能力**（处理复杂场景）和**几何一致性**（保持空间对齐）

## 方法详解

### 整体框架

GeoComplete 包含三个核心组件：

1. **点云生成模块**：从参考和目标图像估计相机参数和深度图，构建 3D 点云并投影
2. **双分支扩散模型**：目标分支处理 masked 图像，云分支处理投影点云，通过联合自注意力融合
3. **Target-aware Masking**：引导模型关注参考图像中对目标视角有信息量的区域

### 关键设计

#### 1. 点云生成（Point Cloud Generation）

**动态物体过滤**：使用 LangSAM（SAM 2.1-Large + 文本提示）分割并移除动态区域（如行人、车辆）。文本提示可由用户提供或 LLM 自动生成。

**几何估计**：使用 **VGGT**（Visual Geometry Grounded Transformer）一次前向传播联合预测：
- 相机参数 $\{\mathbf{c}_i^{ref}\}$, $\mathbf{c}^{tar}$
- 深度图 $\{\mathbf{d}_i^{ref}\}$, $\mathbf{d}^{tar}$

VGGT 避免了传统方法的多阶段误差累积。

**点云投影**：对每个参考图像，从其他视图构建点云并投影：

$$\mathbf{p}_i^{ref} = \pi(\pi^{-1}(\{\mathbf{d}_j^{ref}, \mathbf{c}_j^{ref} | j \neq i\} \cup \{\mathbf{d}^{tar}, \mathbf{c}^{tar}\}), \mathbf{c}_i^{ref})$$

目标视图的投影点云：$\mathbf{p}^{tar} = \pi(\pi^{-1}(\{\mathbf{d}_j^{ref}, \mathbf{c}_j^{ref} | \forall j\}), \mathbf{c}^{tar})$

#### 2. Target-aware Masking（核心创新）

将目标视图投影到每个参考视图，识别**信息区域**（在参考中可见但在目标中缺失的区域）和**冗余区域**。

**条件参考遮罩**（引导模型从互补内容学习）：

$$\hat{\mathbf{x}}_i^{ref} = \mathbf{x}_i^{ref} \odot ((1 - \mathbf{r}_i^{ref}) + \mathbf{r}_i^{ref} \odot \mathbf{m}_i^{rand})$$

保留冗余区域内容，随机遮罩信息区域——驱动模型学习补全这些互补信息。

**条件云遮罩**（引导模型利用几何线索）：

$$\hat{\mathbf{p}}_i^{ref} = \mathbf{p}_i^{ref} \odot \mathbf{m}_i^{point} + v_{fill} \times (1 - \mathbf{m}_i^{point})$$

保留信息区域的几何信息，随机遮罩冗余区域——让模型在视觉信息缺失处依赖几何线索。

两种遮罩的互补设计非常精巧：参考图像遮罩信息区域 → 模型需要从几何中学习；点云保留信息区域 → 提供几何引导。

#### 3. 双分支扩散模型（Dual-branch Diffusion）

基于 Stable Diffusion 2 Inpainting，通过 LoRA（rank=8）微调：

- **目标分支（Target Branch）**：编码 masked 目标图像，生成缺失区域
- **云分支（Cloud Branch）**：编码投影点云，提供几何引导

**联合自注意力**：将两个分支的隐特征拼接 $\mathbf{h}_{cat} \in \mathbb{R}^{2L \times d}$，应用受控注意力掩码：
1. 分支内 token 可互相关注
2. 目标分支每个 token 可关注云分支对应位置的 token
3. 阻止其他跨分支交互

这种设计确保目标分支的 masked token（缺少有意义信息）也能直接获得对应位置的几何引导。

### 损失函数 / 训练策略

扩散损失：

$$\mathcal{L} = \frac{1}{B} \sum_{j=1}^{B} \mathbb{E}_{t,\epsilon}\left[\|\mathbf{w}_j \cdot (\epsilon - \epsilon_\theta(\mathbf{x}_j(t), t, \hat{\mathbf{p}}_j, \hat{\mathbf{x}}_j))\|_2^2\right]$$

$\mathbf{w}_j$ 为有效区域权重，仅在可见区域计算损失。

- 每场景微调 2000 iterations，batch size 16
- LoRA rank=8，训练图像 resize 到 512×512
- VGGT 输入 518×518（中心裁剪）

## 实验关键数据

### 主实验

**RealBench** 数据集（33 场景）：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | DreamSim↓ | DINO↑ | CLIP↑ |
|------|:-----:|:-----:|:------:|:---------:|:-----:|:-----:|
| SD Inpaint | 10.63 | 0.282 | 0.605 | 0.213 | 0.831 | 0.874 |
| Generative Fill | 10.92 | 0.311 | 0.598 | 0.212 | 0.851 | 0.898 |
| Paint-by-Example | 10.13 | 0.244 | 0.642 | 0.237 | 0.797 | 0.859 |
| TransFill | 13.28 | 0.404 | 0.542 | 0.192 | 0.860 | 0.866 |
| RealFill | 14.78 | 0.424 | 0.431 | 0.077 | 0.948 | 0.962 |
| **GeoComplete** | **17.32** | **0.578** | **0.197** | **0.036** | **0.986** | **0.987** |

用户研究（QualBench，25场景，1-5分）：GeoComplete 4.61 vs RealFill 3.98

### 消融实验

| 双分支 | 联合自注意力 | Target-aware | PSNR↑ | SSIM↑ | LPIPS↓ | DINO↑ |
|:-----:|:----------:|:-----------:|:-----:|:-----:|:------:|:-----:|
| ✗ | ✗ | ✗ | 14.78 | 0.424 | 0.431 | 0.948 |
| ✓ | ✗ | ✗ | 16.37 | 0.555 | 0.237 | 0.981 |
| ✓ | ✓ | ✗ | 16.85 | 0.564 | 0.219 | 0.983 |
| ✓ | ✓ | ✓ | **17.32** | **0.578** | **0.197** | **0.986** |

鲁棒性测试（点云噪声/稀疏/LangSAM 错误）：

| 方法 | 0%噪声 | 25%噪声 | 50%噪声 | 75%噪声 |
|------|:------:|:------:|:------:|:------:|
| RealFill | 14.78 | 14.78 | 14.78 | 14.78 |
| Ours w/o CM&JSA | 16.37 | 14.60 | 14.51 | 14.35 |
| **Ours** | **17.32** | **17.14** | **17.03** | **16.90** |

### 关键发现

1. GeoComplete 比 RealFill PSNR 提升 2.54 dB（17.1%），LPIPS 降低 0.234
2. 每个组件均有贡献：双分支 +1.59 PSNR，联合注意力 +0.48，target-aware masking +0.47
3. 条件云遮罩和联合自注意力使模型对点云噪声高度鲁棒（75% 噪声下仍超 RealFill 2.12 dB）
4. 3D 几何先验而非纯生成能力是保持空间一致性的关键

## 亮点与洞察

- **精巧的互补遮罩设计**：参考图像遮罩信息区域 / 点云保留信息区域，形成完美互补
- **受控注意力机制**：token 级的跨分支连接确保几何信息精确传递到对应空间位置
- **鲁棒性设计**：条件遮罩训练策略使模型天然具备对上游误差的鲁棒性
- **端到端几何估计**：VGGT + LangSAM 替代传统多阶段流水线，避免误差累积

## 局限与展望

1. 每个场景需单独微调（2000 iterations），无法实现零样本泛化
2. 依赖 VGGT 的几何估计质量，对极端动态场景可能退化
3. VGGT 输入限制为 518×518，高分辨率场景需要下采样
4. 目前仅处理静态几何，动态内容通过 LangSAM 直接移除而非重建
5. 未探索视频补全场景中的时序一致性

## 相关工作与启发

- **RealFill**：本文的主要基线，通过 LoRA 微调扩散模型做参考驱动补全，但缺乏几何
- **VGGT**：统一预测相机参数/深度/点云的 Transformer，替代传统多阶段估计
- **TransFill**：传统几何方法代表，依赖串行流水线
- 启发：在生成模型中注入显式 3D 几何先验是兼顾生成能力和空间一致性的有效路径

## 评分

- 新颖性: ⭐⭐⭐⭐ 双分支扩散 + 互补遮罩 + 几何注入的组合设计新颖
- 实验充分度: ⭐⭐⭐⭐ 定量/定性/消融/鲁棒性全面覆盖
- 写作质量: ⭐⭐⭐⭐ 技术描述清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ PSNR 提升 17.1% 是显著的实用改进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Reference-Based 3D-Aware Image Editing with Triplanes](../../CVPR2025/3d_vision/reference-based_3d-aware_image_editing_with_triplanes.md)
- [\[NeurIPS 2025\] GOATex: Geometry & Occlusion-Aware Texturing](goatex_geometry_occlusion-aware_texturing.md)
- [\[CVPR 2026\] PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis](../../CVPR2026/3d_vision/pr-iqa_partial-reference_image_quality_assessment_for_diffusion-based_novel_view.md)
- [\[NeurIPS 2025\] WildCAT3D: Appearance-Aware Multi-View Diffusion in the Wild](wildcat3d_appearance-aware_multi-view_diffusion_in_the_wild.md)
- [\[NeurIPS 2025\] Linearly Constrained Diffusion Implicit Models](linearly_constrained_diffusion_implicit_models.md)

</div>

<!-- RELATED:END -->
