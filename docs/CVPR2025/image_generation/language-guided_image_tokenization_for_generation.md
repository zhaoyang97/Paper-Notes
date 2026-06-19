---
title: >-
  [论文解读] Language-Guided Image Tokenization for Generation
description: >-
  [CVPR 2025][图像生成][图像分词] TexTok 提出在图像分词（tokenization）阶段引入文本描述作为条件，将高层语义信息卸载给文本，使图像 token 专注于编码细粒度视觉细节，从而在保持甚至提升重建质量的同时实现更高的压缩率，在 ImageNet 上取得了 SOTA 的生成 FID 分数 1.46。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "图像分词"
  - "文本条件"
  - "图像压缩"
  - "Transformer"
  - "高效生成"
---

# Language-Guided Image Tokenization for Generation

**会议**: CVPR 2025  
**arXiv**: [2412.05796](https://arxiv.org/abs/2412.05796)  
**代码**: [https://kaiwenzha.github.io/textok/](https://kaiwenzha.github.io/textok/) (项目页)  
**领域**: 图像生成 / 扩散模型  
**关键词**: 图像分词, 文本条件, 图像压缩, 扩散Transformer, 高效生成

## 一句话总结
TexTok 提出在图像分词（tokenization）阶段引入文本描述作为条件，将高层语义信息卸载给文本，使图像 token 专注于编码细粒度视觉细节，从而在保持甚至提升重建质量的同时实现更高的压缩率，在 ImageNet 上取得了 SOTA 的生成 FID 分数 1.46。

## 研究背景与动机

1. **领域现状**：图像生成的核心依赖图像分词器（tokenizer），将原始像素压缩为紧凑的隐空间表示，让生成模型（扩散模型、自回归模型）在压缩空间高效操作。主流方法包括 VQ-VAE/VQGAN 离散分词和 VAE 连续分词。
2. **现有痛点**：当前分词方法面临压缩率与重建质量的根本性权衡——高压缩降低计算成本但牺牲重建质量，追求质量则带来高计算开销。在高分辨率图像生成中，这一问题尤为严重。
3. **核心矛盾**：图像 token 需要同时承载高层语义信息和底层视觉细节，但在 token 数量有限时两者无法兼顾。
4. **本文目标** 如何在显著降低 token 数量（更高压缩率）的同时保持或提升图像重建和生成质量。
5. **切入角度**：人类描述图像时先概括语义再补充细节——如果文本已经承载了高层语义，图像 token 就可以把全部容量用于编码细粒度细节。
6. **核心 idea**：让文本描述承担语义学习的负担，释放图像 token 的学习容量来捕捉更精细的视觉细节。

## 方法详解

### 整体框架
TexTok 采用 ViT 架构的编码器（tokenizer）和解码器（detokenizer）。输入是图像及对应的文本描述（通过 VLM 离线生成）。编码器接收三类输入的拼接：图像 patch token、可学习图像 token、文本 token（由冻结的 T5 文本编码器提取）。编码器输出中只保留学习到的图像 token 作为隐表示。解码器同样接收三类输入：可学习 patch token、图像 token 和相同的文本 token，输出重建的图像。生成阶段只需用 DiT 生成图像 token，文本 token 在解码时直接提供。

### 关键设计

1. **文本条件注入（Text Token Injection）**:

    - 功能：将高层语义信息通过文本 token 注入编码器和解码器，减轻图像 token 的语义学习负担。
    - 核心思路：使用冻结的 T5 文本编码器（XL 用于 256 分辨率，XXL 用于 512 分辨率）将图像描述编码为文本嵌入 $\mathbf{T} \in \mathbb{R}^{N_t \times D}$，通过线性投影对齐维度后与图像 patch token 和可学习图像 token 拼接，一起输入 ViT 编码器。解码器侧同样注入相同的文本 token。文本编码器全程冻结，不参与训练。
    - 设计动机：直接注入而非强制对齐——与之前强迫图像 token 对齐文本表示的方法不同，TexTok 仅将文本作为辅助条件，避免了视觉-语言表示本质差异导致的重建质量下降。

2. **1D 全局图像 token 架构**:

    - 功能：实现灵活可控的 token 数量，支持从 32 到 256 的不同压缩率。
    - 核心思路：采用 1D tokenizer 范式，通过 $N$ 个随机初始化的可学习 token $\mathbf{L} \in \mathbb{R}^{N \times D}$ 从图像中聚合信息，编码器输出后经线性投影得到 $\mathbf{Z} \in \mathbb{R}^{N \times d}$（$d=8$）。不同于 2D 空间 token 需要固定的下采样率，1D 全局 token 数量可以自由设定。
    - 设计动机：灵活的 token 预算让研究者可以在精度和效率之间按需选择，也使得文本条件在低 token 数量下的增益更加明显。

3. **生成阶段的文本利用策略**:

    - 功能：在推理阶段无缝使用文本信息。
    - 核心思路：对于文本到图像生成，直接使用给定的文本描述。对于类别条件生成，DiT 基于类别生成隐 token，然后从预生成的该类别描述列表中采样一条未见过的描述，与生成的隐 token 一起送入解码器产生最终图像。生成阶段只需要生成图像 token，文本 token 免费提供。
    - 设计动机：文本描述在文本到图像任务中是天然可用的，不需要额外标注开销；对于类别条件任务则通过 VLM 离线批量生成，一次性成本低且可复用。

### 损失函数 / 训练策略
训练使用 $\ell_2$ 重建损失、GAN 对抗损失、感知损失（perceptual loss）和 LeCAM 正则化损失的组合。GAN 判别器使用 StyleGAN 判别器（~24M 参数）。编码器和解码器各 12 层 ViT-Base（~176M 参数），token 通道维度 $d=8$。DiT 生成器使用 patch size=1，训练 350 epochs。

## 实验关键数据

### 主实验

| 设置 | 分辨率 | Token 数 | rFID (重建) | gFID (生成) | 相对 Baseline 提升 |
|------|--------|----------|-----------|-----------|------------------|
| TexTok-32 | 256×256 | 32 | 2.40 | 3.55 | rFID -37.2%, gFID -28.6% |
| TexTok-64 | 256×256 | 64 | 1.53 | 2.88 | rFID -25.0%, gFID -12.7% |
| TexTok-256 | 256×256 | 256 | 0.69 | 2.68 | rFID -24.2%, gFID -7.9% |
| TexTok-32 | 512×512 | 32 | 2.33 | 3.61 | rFID -69.7%, gFID -60.8% |
| TexTok-256 + DiT-XL | 256×256 | 256 | - | **1.46** | SOTA |
| TexTok-256 + DiT-XL | 512×512 | 256 | - | **1.62** | SOTA |

TexTok 在 512 分辨率下仅用 32 个 token 就超越了原始 DiT 使用 4096 个 token 的性能，实现 **93.5× 推理加速**。

### 消融实验

| 配置 | rFID (256) | 说明 |
|------|-----------|------|
| TexTok (Full) | 1.04 | 编码器+解码器均有文本 |
| Tokenizer only | 1.11 | 仅编码器有文本 |
| Detokenizer only | 1.28 | 仅解码器有文本 |
| Baseline (w/o text) | 1.49 | 无文本条件 |
| TexTok + 文本到图像 | 2.82 FID, 29.23 CLIP | T2I 任务也受益 |

### 关键发现
- **token 越少，文本增益越大**：32 token 时 rFID 改进 37.2%（256 分辨率）和 69.7%（512 分辨率），256 token 时改进约 24%。说明文本在低带宽下承担了更多语义职责。
- **高分辨率增益更显著**：512 分辨率下文本条件带来的提升几乎是 256 分辨率的两倍，因为高分辨率图像的语义冗余更多。
- TexTok 可以用一半 token 数（256 分辨率）甚至 1/4 token 数（512 分辨率）达到与 Baseline 相同的 rFID。
- 编码器和解码器两侧同时注入文本效果最好，但编码器侧的贡献更大。

## 亮点与洞察
- **在分词阶段而非生成阶段使用文本**：这是一个反直觉但极其有效的设计——之前文本条件总是在生成阶段使用，而 TexTok 首次将其前移到分词阶段,将语义负担卸载给文本,堪称「职责分离」的典范。
- **免费的午餐**：在文本到图像任务中，文本描述本身就是训练数据的一部分，不需要额外标注；在类别条件任务中也只需一次性用 VLM 生成描述，后续训练和推理几乎无额外开销。
- **93.5× 推理加速的实际价值**：将 512 分辨率的 DiT 从 4096 token 压缩到 32 token 且性能不降，这对于实际部署高分辨率生成模型意义重大，可以直接迁移到视频生成等更高维场景。

## 局限与展望
- **依赖文本描述质量**：重建和生成质量受文本描述质量影响，若 VLM 生成的描述有误，可能会引导 tokenizer 编码错误的语义。
- **推理时仍需文本**：类别条件生成需要在推理时提供文本描述，增加了系统复杂度。
- **仅验证了人脸/物体等场景**：在 ImageNet 上的效果显著，但在更复杂的场景（如医学图像、遥感图像）中文本描述是否同样有效未被验证。
- 可以探索将文本条件与其他模态条件（如深度图、分割掩码）结合，进一步提升分词效率。
- 自适应地动态决定每张图需要多少 token（基于图像复杂度）可能进一步优化效率。

## 相关工作与启发
- **vs TiTok**: TiTok 也使用 1D 全局 token，但未利用文本条件；TexTok 在相同 token 数下全面超越。
- **vs SD-VAE**: SD-VAE 使用 1024 个 2D 空间 token（$d=4$），Baseline（无文本）只用 32 个 1D token 就已超越其重建性能，加上文本后优势更大。
- **vs 文本对齐方法（如 LQAE, Spae）**: 这些方法强制图像 token 对齐文本空间导致重建质量下降，TexTok 仅用文本作为外部条件辅助，补充而非替代视觉表示。
- 文本辅助分词的思路可以迁移到视频分词（文本描述时序变化）和 3D 分词（文本描述空间结构）。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在分词阶段引入文本条件，思路简洁优雅且效果显著
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖多分辨率、多 token 数、多任务，消融全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，实验展示逻辑性强
- 价值: ⭐⭐⭐⭐⭐ 为高效图像生成提供新范式，93.5× 加速极具实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Yo'Chameleon: Personalized Vision and Language Generation](yochameleon_personalized_vision_and_language_generation.md)
- [\[CVPR 2025\] Efficient Long Video Tokenization via Coordinate-based Patch Reconstruction](efficient_long_video_tokenization_via_coordinate-based_patch_reconstruction.md)
- [\[CVPR 2025\] Visual Lexicon: Rich Image Features in Language Space](visual_lexicon_rich_image_features_in_language_space.md)
- [\[CVPR 2026\] MacTok: Robust Continuous Tokenization for Image Generation](../../CVPR2026/image_generation/mactok_robust_continuous_tokenization_for_image_generation.md)
- [\[ICCV 2025\] Efficient Autoregressive Shape Generation via Octree-Based Adaptive Tokenization](../../ICCV2025/image_generation/efficient_autoregressive_shape_generation_via_octree-based_adaptive_tokenization.md)

</div>

<!-- RELATED:END -->
