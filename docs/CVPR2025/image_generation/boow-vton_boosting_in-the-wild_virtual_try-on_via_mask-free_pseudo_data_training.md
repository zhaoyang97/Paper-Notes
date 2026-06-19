---
title: >-
  [论文解读] BooW-VTON: Boosting In-the-Wild Virtual Try-On via Mask-Free Pseudo Data Training
description: >-
  [CVPR 2025][图像生成][虚拟试穿] 提出 BooW-VTON，通过高质量伪数据构建 + 野外数据增广 + 试穿定位损失，训练出无需人体解析掩码的虚拟试穿扩散模型，在 VITON-HD/StreetVTON/WildVTON 多个基准上全面超越现有方法。 领域现状：基于图像的虚拟试穿（VTON）旨在将目标服装自然地…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "虚拟试穿"
  - "无掩码推理"
  - "伪数据训练"
  - "注意力正则化"
  - "野外场景"
---

# BooW-VTON: Boosting In-the-Wild Virtual Try-On via Mask-Free Pseudo Data Training

**会议**: CVPR 2025  
**arXiv**: [2408.06047](https://arxiv.org/abs/2408.06047)  
**代码**: 无（未提及）  
**领域**: 扩散模型 / 图像生成  
**关键词**: 虚拟试穿, 无掩码推理, 伪数据训练, 注意力正则化, 野外场景

## 一句话总结
提出 BooW-VTON，通过高质量伪数据构建 + 野外数据增广 + 试穿定位损失，训练出无需人体解析掩码的虚拟试穿扩散模型，在 VITON-HD/StreetVTON/WildVTON 多个基准上全面超越现有方法。

## 研究背景与动机

**领域现状**：基于图像的虚拟试穿（VTON）旨在将目标服装自然地渲染到人物图像上。现有主流方法（IDM-VTON、StableVITON 等）采用掩码修复范式——先用人体解析器获取试穿区域掩码，遮盖该区域后用扩散模型修复。在简单电商场景下效果不错。

**现有痛点**：掩码方法有三个根本缺陷。（1）**空间信息丢失**：遮盖试穿区域会破坏原始图像的深度、光照、纹理等空间信息；（2）**前后景断裂**：掩码打断了前景和背景的连贯性；（3）**依赖解析器**：需要额外的人体解析器提供姿态信息，在野外场景（复杂姿态、遮挡）下解析器本身不可靠。这些缺陷在复杂野外场景中会导致明显伪影——配饰丢失、皮肤纹理改变、场景不一致等。

**核心矛盾**：无掩码试穿可以充分利用原始图像的空间和光照信息，但训练无掩码模型需要 {换装后图像, 服装, 原始图像} 的三元组数据，而这种精确对齐的数据不存在。直接从掩码模型蒸馏无掩码学生模型（如 PFDM）会继承教师模型的缺陷，无法泛化到复杂场景。

**本文目标** 如何训练一个不需要掩码输入的高质量虚拟试穿模型，尤其在复杂野外场景下保持服装渲染准确性和非试穿区域保真度。

**切入角度**：不通过蒸馏，而是用掩码模型在简单场景下生成高质量伪数据作为训练信号，再通过数据增广和注意力正则化将能力迁移到复杂场景。

**核心 idea**：用精细化的掩码模型推理生成高质量伪换装数据，配合合成的野外前后景增广和试穿定位损失，训练无掩码试穿扩散模型。

## 方法详解

### 整体框架
以 SDXL 为基础模型，IP-Adapter 和 Reference Net 作为服装编码器。输入为原始人物图像 $P'$（换装状态伪图）和服装图像 $G$，将 $P'$ 的 latent 与加噪的目标 latent 在通道维拼接输入 try-on U-Net。训练流程分三步：（1）用掩码模型在简单场景下两阶段推理生成高质量伪数据；（2）对伪数据进行野外前后景增广；（3）用试穿定位损失约束注意力聚焦于试穿区域。推理时完全不需要掩码或人体解析器。

### 关键设计

1. **两阶段精细化伪数据生成**

    - 功能：从掩码模型的输出中获取高质量的无掩码训练三元组
    - 核心思路：使用 IDM-VTON 作为掩码模型。第一阶段用宽松的粗掩码 $M_{coa}$ 进行试穿，获得中间结果 $P_{mid}$。从 $P_{mid}$ 提取服装区域 $M_{mid}$，与原始服装区域 $M_P$ 取并集得到更精确的掩码 $M$。第二阶段用精确掩码再次推理，生成保留更多非试穿内容的高质量伪数据 $P'$。这样构建出 $\{P', G, P\}$ 三元组，其中 $P'$ 是换装后图像（条件输入），$P$ 是原始图像（监督目标）
    - 设计动机：直接用粗掩码生成的结果有掩码边界伪影，两阶段精细化在简单场景下可以获得接近完美的伪数据，为无掩码模型提供优质监督信号

2. **野外数据增广（In-the-Wild Data Augmentation）**

    - 功能：将简单电商场景的伪数据扩展为复杂野外场景训练数据
    - 核心思路：为 $\{P', P\}$ 同时添加合成的背景和前景。**背景生成**：先用人物透明图和 T2I 模型修复空白区域获得人物+背景混合图，再修复人物区域获得干净背景。**前景生成**：用 GPT-4o 生成物体 prompt，Layerdiffusion 生成透明前景图。训练时随机选取前后景，按 B-P/P'-F 从底到顶叠加组合，并随机应用平移、缩放变换。前景只遮挡非试穿区域（修改试穿掩码 $M^{Aug}$ 以排除前景区域）
    - 设计动机：掩码模型通常只在简单电商数据上训练，直接生成的伪数据不包含复杂前后景。通过合成增广让模型学会在有复杂遮挡和背景干扰时仍保持准确的试穿和保真

3. **试穿定位损失（Try-On Localization Loss）**

    - 功能：约束注意力层只在试穿区域渲染服装特征，防止改变非试穿区域
    - 核心思路：在注意力层中，person latent code 作为 Query，garment tokens 作为 Key/Value，注意力分数 $A_k$ 反映了服装特征流向 2D 人物空间的程度。试穿定位损失最小化非试穿区域的注意力分数：$\mathcal{L}_{ar} = \frac{1}{n}\sum_{k=1}^{n} \text{mean}(A_k(1-M^{Aug}))$。应用于 SDXL 70 个注意力块中的 5-64 号块（图像 token 长度 32×24），同时作用于 cross-attention 和 self-attention 层。仅在训练时使用掩码，推理时完全无需掩码
    - 设计动机：不加约束时注意力会扩散到整个图像，导致非试穿区域（配饰、皮肤、前后景）被服装特征污染。通过显式约束注意力聚焦，实现"知道在哪编辑"的效果

### 损失函数 / 训练策略
总损失 $\mathcal{L} = \mathcal{L}_{LDM} + \lambda_{ar}\mathcal{L}_{ar}$，其中 $\lambda_{ar}=1$。基于 IDM-VTON 权重初始化，只解冻 try-on U-Net。16 × H100 GPU 训练约 12 小时，批量 32，12000 步，学习率 5e-6，Adam 优化器。推理用 30 步 DDIM。

## 实验关键数据

### 主实验

| 方法 | VITON-HD LPIPS↓ | SSIM↑ | StreetVTON FID_u↓ | WildVTON FID_u↓ |
|------|----------------|-------|-------------------|-----------------|
| DCI-VTON | 0.1800 | 0.8545 | 20.95 | 35.66 |
| StableVITON | 0.1479 | 0.8519 | 23.15 | 42.32 |
| IDM-VTON | 0.1223 | 0.8547 | 23.62 | 38.77 |
| **BooW-VTON** | **0.1080** | **0.8618** | **20.50** | **32.53** |

DressCode-Upper: LPIPS **0.0615** vs IDM-VTON 0.0761

### 消融实验

| 配置 | VITON-HD LPIPS↓ | StreetVTON FID_u↓ | WildVTON FID_u↓ |
|------|----------------|-------------------|-----------------|
| Base mask-free | 0.1206 | 28.81 | 57.52 |
| + 高质量伪数据 | 0.1101 | 27.26 | 56.14 |
| + 野外增广 | 0.1173 | 21.70 | 35.62 |
| + 试穿定位损失 (Full) | **0.1080** | **20.50** | **32.53** |

### 关键发现
- 野外数据增广贡献最大：WildVTON FID 从 56.14 直降到 35.62，StreetVTON 从 27.26 降到 21.70，说明合成前后景有效教会模型处理复杂场景
- 试穿定位损失在野外场景上进一步提升：WildVTON FID 从 35.62 降到 32.53，注意力约束避免了非试穿区域被错误修改
- 在简单电商场景（VITON-HD）上，LPIPS/SSIM/PSNR 均超越 IDM-VTON，证明无掩码方式不仅不掉精度反而更好（保留了完整空间信息）
- 未在动漫数据上训练也能泛化到动漫风格试穿，体现了预训练模型的跨域能力

## 亮点与洞察
- **掩码方法的根本缺陷分析**很有说服力：通过深度图对比展示掩码导致的空间信息丢失，为无掩码方案提供了理论动机
- **伪数据 + 增广的二级训练策略**巧妙规避了无掩码训练数据不存在的问题，先在简单场景生成精确伪数据，再通过增广扩展到复杂场景。这种策略可迁移到其他缺少配对数据的图像编辑任务
- **注意力正则化**是一种轻量但有效的方式来控制编辑区域，不增加推理开销（仅训练时使用掩码）

## 局限与展望
- 当试穿 T 恤但原图穿连衣裙时，下半身无参考信息会被随机生成，无法控制
- 同理，下装试穿时上半身不可控，模型无法协调上下装搭配
- 伪数据质量仍受限于 IDM-VTON 的能力，极端姿态或严重遮挡时伪数据可能有瑕疵
- 训练数据只来源于电商数据集（VITON-HD/DressCode），野外泛化依赖于增广质量

## 相关工作与启发
- **vs IDM-VTON**: IDM-VTON 是掩码方法的 SOTA，BooW-VTON 在其基础上初始化但去掉了掩码依赖，所有指标均超越，尤其野外场景优势显著（WildVTON FID 32.53 vs 38.77）
- **vs PFDM**: PFDM 也做无掩码但通过蒸馏，继承了掩码模型的缺陷。BooW-VTON 通过精细化伪数据避免了缺陷传递
- **vs TPD**: TPD 用两阶段改进掩码精度但仍是掩码范式，无法从根本上解决空间信息丢失问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 无掩码试穿思路不完全新（PFDM），但伪数据构建+增广+定位损失的组合有效
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集+完整消融+多个基线+定性定量，实验非常扎实
- 写作质量: ⭐⭐⭐⭐ 动机分析清晰，方法描述详细，图表丰富
- 价值: ⭐⭐⭐⭐ 去掉了试穿的解析器依赖，对实际应用有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] OmniVTON: Training-Free Universal Virtual Try-On](../../ICCV2025/image_generation/omnivton_training-free_universal_virtual_try-on.md)
- [\[CVPR 2025\] Shining Yourself: High-Fidelity Ornaments Virtual Try-on with Diffusion Model](shining_yourself_high-fidelity_ornaments_virtual_try-on_with_diffusion_model.md)
- [\[CVPR 2026\] PG-VTON: Single-Pass Training-Free Virtual Try-On via Patch-Guided Reference Alignment](../../CVPR2026/image_generation/pg-vton_single-pass_training-free_virtual_try-on_via_patch-guided_reference_alig.md)
- [\[CVPR 2025\] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction](pursuing_temporal-consistent_video_virtual_try-on_via_dynamic_pose_interaction.md)
- [\[ECCV 2024\] WildVidFit: Video Virtual Try-On in the Wild via Image-Based Controlled Diffusion Models](../../ECCV2024/image_generation/wildvidfit_video_virtual_try-on_in_the_wild_via_image-based_controlled_diffusion.md)

</div>

<!-- RELATED:END -->
