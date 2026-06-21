---
title: >-
  [论文解读] GarmentGPT: Compositional Garment Pattern Generation via Discrete Latent Tokenization
description: >-
  [ICLR 2026][图像生成][缝纫纸样生成] GarmentGPT 把缝纫纸样（sewing pattern）的连续边界曲线用 RVQ-VAE 量化成离散 codebook token，再让微调后的 VLM 自回归地"选词"生成这些 token，从而把纸样生成从低层坐标回归变成高层符号化的组合推理任务，并配套造出百万级真实人像-纸样数据集。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "缝纫纸样生成"
  - "RVQ-VAE"
  - "离散 token 化"
  - "视觉语言"
  - "服装数字化"
---

# GarmentGPT: Compositional Garment Pattern Generation via Discrete Latent Tokenization

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=XzXKnazRBF](https://openreview.net/forum?id=XzXKnazRBF)  
**代码**: [https://github.com/ChimerAI-MMLab/Garment-GPT](https://github.com/ChimerAI-MMLab/Garment-GPT)  
**领域**: 服装数字化 / 结构化生成 / 多模态生成  
**关键词**: 缝纫纸样生成, RVQ-VAE, 离散 token 化, Vision-Language Model, 服装数字化  

## 一句话总结
GarmentGPT 把缝纫纸样（sewing pattern）的连续边界曲线用 RVQ-VAE 量化成离散 codebook token，再让微调后的 VLM 自回归地"选词"生成这些 token，从而把纸样生成从低层坐标回归变成高层符号化的组合推理任务，并配套造出百万级真实人像-纸样数据集。

## 研究背景与动机
- **领域现状**：服装数字化是数字人、虚拟试衣、电商的基础需求，而缝纫纸样（2D 面板 + 边曲线 + 缝合关系）是决定 3D 服装最终形状、版型、风格的"蓝图"。传统纸样制作高度依赖打版师的经验直觉，是数字服装规模化的瓶颈。
- **现有痛点**：自动生成纸样的两条主流路线都停留在"原始数据空间"、缺乏高层推理。① 扩散模型（如 SewingLDM）擅长拟合分布，但只是"数据复制者"，不理解服装构造原理；② 直接用 VLM（如 ChatGarment、AIpparel）回归原始浮点坐标——这是把强推理引擎硬塞去做低层数值回归，成千上万个坐标误差会灾难性累积。
- **核心矛盾**：纸样既需要**精确的几何约束**（顶点、控制点、半径要准），又需要**高层的组合拓扑**（哪些边缝在一起、面板如何拼装），而连续坐标回归无法同时兼顾精度与符号推理；同时缺乏"真实照片 ↔ 精确纸样"的大规模配对数据，模型无法泛化到真实照片输入。
- **本文目标**：让纸样生成对齐 LLM 的符号推理能力，从真实人像/文本直接产出结构正确、可编辑、可制造的纸样。
- **核心 idea**：**离散组合范式**——借鉴 Latent Diffusion 把问题映射到紧凑语义隐空间的思路，先用 RVQ-VAE 把纸样曲线"词典化"成离散 token（代表面板、曲线、连接等有意义的部件），再让 VLM **自回归预测 token 序列而非回归坐标**，把生成变成"按裁剪原理拼词"的组合任务。

## 方法详解

### 整体框架
GarmentGPT 由两个核心模块串联：① **量化模块**（RVQ-VAE）把每条边的连续曲线和每个面板的位置参数分别编码成离散 codebook 索引；② **序列生成/编辑模块**（微调 VLM）以图像/文本/已有纸样序列为条件，自回归地预测一串带特殊 token 的"服装序列"，再用模式匹配解构序列、查 codebook 解码回完整纸样（尺寸、面板位姿、边的几何与缝合关系）。

```mermaid
flowchart LR
    A[纸样: 面板+边曲线+RT位姿] --> B[RVQ-VAE 编码]
    B --> C1[Edge Codebook 边索引]
    B --> C2[RT Codebook 位姿索引]
    C1 & C2 --> D[层级化 Tokenization<br/>SoG/SoP/SoE/SoS...]
    E[图像 / 文本 / 已有序列] --> F[微调 VLM<br/>LLaVA-1.5 / Qwen-VL]
    D -.训练目标.-> F
    F --> G[自回归预测 token 序列]
    G --> H[模式匹配解构 + 查 codebook 解码]
    H --> I[完整纸样: 尺寸/位姿/边/缝合]
```

### 关键设计
**1. RVQ-VAE 纸样量化：边几何与面板位姿用独立 codebook 分而治之。** 这是整个范式的地基。纸样的每条边被归为四类几何（直线、二次/三次贝塞尔曲线、圆弧），各有专属参数。作者把每条边均匀采样成 $N$ 个点喂进轻量 ResNet 编码器，把面板的平移-旋转（RT，在 SMPL A-pose 下）拼成单一向量用更小的 ResNet 编码，然后通过**残差量化**把连续隐向量映射到层级 codebook 索引——多个 codebook 之间是残差关系，残差层数 $Q$ 控制压缩率与重建质量的平衡。关键之处在于"表征纯净性"：边和 RT 参数使用**完全不共享**的编码器-解码器和 codebook，避免几何与位姿信息互相污染。解码时按索引查回量化向量，边解码器还原端点和类型专属属性（圆弧半径、贝塞尔控制点经无损二次转三次），RT 直接回归连续值，实现高保真重建。

**2. 层级化 Tokenization：用一套特殊 token 把纸样的拓扑结构语法化。** 量化得到索引后还需把它们组织成 VLM 能消化的序列。作者设计了一组配对的特殊 token $T = \{\langle\text{SoG}\rangle, \langle\text{EoG}\rangle, \langle\text{SoP}\rangle, \langle\text{EoP}\rangle, \langle\text{SoL}\rangle, \langle\text{EoL}\rangle, \langle\text{SoE}\rangle, \langle\text{EoE}\rangle, \langle\text{SoS}\rangle, \langle\text{EoS}\rangle, \langle\text{ESEG}\rangle\}$，分别标记"整件服装/面板/位置参数/边/缝合关系"的起止，$\langle\text{ESEG}\rangle$ 作为同一面板内不同边索引的分隔符。整条序列像一棵语法树：$\langle\text{SoG}\rangle$ 后接尺寸 → 各面板数据（每个面板含标识、位姿、按绘制顺序排列的边序列）→ 缝合关系（描述需缝合的边对，如 right_btorso 第 4 边与 left_btorso 第 4 边）→ $\langle\text{EoG}\rangle$ 收尾。这把几何拓扑显式编码成了符号序列，正好对上 VLM 的语言建模强项。

**3. VLM 自回归"选词"：把坐标回归改写成 token 选择。** 作者用 LLaVA-1.5-7B / Qwen-2.5-VL 作骨干，做三处适配：① **词表扩展**——把拓扑特殊 token 和 codebook 索引（0–1023）作为可学习 embedding 加进 LLM 词表；② **多模态输入构造**——生成任务用图文对，编辑任务用"序列+编辑指令"对，视觉与文本经投影层融合；③ **训练范式**——微调 VLM 自回归预测拓扑 token 序列，最小化对 ground-truth 序列的交叉熵 $L_{\text{VLM}} = -\frac{1}{N}\sum_{i=1}^{N} \log P(\text{token}_i \mid \text{token}_{<i}, C)$，其中 $C$ 是图像/文本/序列上下文。因为输出是从有限词典里"选词"而非回归无界浮点数，VLM 的符号推理优势被释放，根本上避开了连续坐标回归的误差累积。

**4. 量化器的复合损失：分几何类型精细约束重建。** RVQ-VAE 的训练目标是加权复合损失 $L_{\text{quant}} = \lambda_{\text{cls}}L_{\text{cls}} + \lambda_{\text{vertex}}L_{\text{vertex}} + \lambda_{\text{control}}L_{\text{control}} + \lambda_{\text{commit}}L_{\text{commit}}$：$L_{\text{cls}}$ 用交叉熵预测边的几何类型；$L_{\text{vertex}} = \|v_{\text{pred}} - v_{\text{gt}}\|_2^2$ 约束端点精度；$L_{\text{control}}$ 按曲线类型差异化计算（直线用三等分点作代理控制点，贝塞尔算控制点 L2，圆弧则 $L_{\text{control}}^{\text{arc}} = \|r_{\text{pred}} - r_{\text{gt}}\|_2^2 + \text{BCE}(d_{\text{pred}}, d_{\text{gt}}) + \text{BCE}(a_{\text{pred}}, a_{\text{gt}})$ 同时约束半径、绘制方向、大小弧标识）；$L_{\text{commit}} = \beta\|z_e(x) - \text{sg}(z_q)\|_2^2$ 维持编码向量与 codebook 条目的对应、驱动码本更新。

**5. 数据策展管线：从 GarmentCode 合成百万真实人像-纸样配对。** 真实路线最大的卡点是缺"真实照片 ↔ 精确纸样"配对。作者四阶段造数据：① **纹理提取增强**——用 Grounded-SAM + FabricDiffusion 处理已有数据集，得到 RGT-164K（16.4 万张独特纹理图）；② **运动感知仿真**——从 AMASS 抽 SMPL 姿态，用 GarmentCode + ContourCraft 渲染不同姿态/纹理的着衣人体视频；③ **真实感转换**——选关键帧经 Qwen-Image-Edit 转成照片级图像，靠物理仿真保证服装结构一致；④ **质量过滤**——多级筛掉服装脱离、不当暴露、覆盖异常等情况，把对齐可接受率从 64.3% 提到 99.6%。最终产出 RG-1M（百万级真实人像-GarmentCode 配对）和 RG-Bench（首个从真实人像评测纸样生成的 benchmark）。

## 实验关键数据

### 主实验（结构化 GarmentCode 数据集）

| 方法 | 设置 | Panel Acc.↑ | Edge Acc.↑ | Stitch Acc.↑ | Vertices L2↓ | Rotation L2↓ | Translation L2↓ |
|------|------|------|------|------|------|------|------|
| ChatGarment | — | 60.22% | 42.12% | 49.21% | 30.15 | 10.51 | 10.03 |
| AIpparel | — | 78.92% | 74.31% | 56.57% | 25.55 | 3.87 | 5.22 |
| GarmentGPT (LLaVA-1.5) | Text-only | 64.03% | 76.71% | 53.16% | 48.19 | 0.84 | 8.80 |
| GarmentGPT (LLaVA-1.5) | Image-only | 93.53% | 89.75% | 80.98% | 17.33 | 0.56 | 2.93 |
| **GarmentGPT (LLaVA-1.5)** | **Image+Text** | **95.62%** | **90.48%** | **81.84%** | 18.43 | 0.59 | 3.05 |
| GarmentGPT (LLaVA-1.5) | Editing | 93.80% | 94.62% | 92.95% | 11.07 | 0.97 | 2.93 |

- 相比连续回归 SOTA（AIpparel），Panel Acc. +16.7%、Stitch Acc. +25.3%，印证离散组合范式优势；图文融合优于单模态（Panel 93.53%→95.62%）；编辑任务 Stitch Acc. 高达 92.95%，说明离散表征对局部修改友好。

### Real-Garments Benchmark（真实照片，>2000 张）

| 方法 | Panel Acc.↑ | Edge Acc.↑ | Stitch Acc.↑ | Vertices L2↓ |
|------|------|------|------|------|
| ChatGarment | 25.34% | 17.82% | 18.45% | 71.28 |
| AIpparel | 38.76% | 41.25% | 27.34% | 58.92 |
| GarmentGPT (Image-only) | 88.67% | 84.28% | 76.34% | 19.45 |
| **GarmentGPT (Image+Text)** | **90.84%** | **85.92%** | **77.56%** | 20.67 |

- 真实场景下所有方法都掉点（验证 benchmark 难度），但 GarmentGPT 仍保留约 95% 性能，Panel Acc. 90.84% 是最佳 baseline 的 **2.3 倍**，说明离散 token 学到了姿态不变的鲁棒表征。

### 消融：RVQ-VAE 残差层数 Q

| 指标 | Q=1 | Q=3 | Q=5 | Q=8 |
|------|------|------|------|------|
| Total Loss↓ | 3.72 | 0.36 | 0.15 | **0.08** |
| Vertex Loss↓ (×10⁻³) | 5.9 | 0.39 | 0.14 | **0.05** |
| Curve Acc.↑ | 93.3% | 98.9% | 99.5% | **99.8%** |

### 关键发现
- 残差层数从 1 增到 8，重建质量大幅提升（曲线精度 93.3%→99.8%），Q=8 才能保证可靠的纸样 token 化——量化重建质量是整个 pipeline 的下限。
- VLM 骨干越大越好但收益递减：Qwen-3B→7B→32B 的 Panel Acc. 为 85.56%→90.31%→91.05%；LLaVA-1.5-7B（95.62%）在该任务上反而表现最强。

## 亮点与洞察
- **范式转换的洞见**：把纸样生成从"低层坐标回归"重构为"高层 token 选择"，让 VLM 做擅长的符号/组合推理而非吃力的数值回归——这是论文最核心、也最可迁移到其他结构化几何（CAD、矢量图、网格）的思想。
- **几何与位姿解耦量化**：边曲线和面板 RT 用独立 codebook，避免表征互相干扰，是 RVQ-VAE 能高保真重建的关键工程细节。
- **数据闭环补齐真实落地最后一公里**：用仿真+真实感转换+多级过滤造出百万级真实人像配对，把对齐可接受率从 64.3% 拉到 99.6%，解决了"真实照片→纸样"长期缺数据的瓶颈，且配套了 RG-Bench 评测标准。

## 局限与展望
- 真实 benchmark 仍由"虚拟人渲染→真实感转换"合成而来，与互联网真实照片仍有域差距（论文虽展示了少量真实照片结果，但量级与多样性有限）。
- 纸样几何被限定为四类边（直线/二次三次贝塞尔/圆弧），更复杂的褶皱、特殊剪裁是否能被有限词典充分表达仍待验证。
- codebook 大小（1024）与残差层数（Q=8）是固定超参，对超复杂服装是否够用、词典如何随服装品类扩展尚未探讨。
- 评测主要在结构准确性与几何误差上，缺少对最终 3D 仿真穿着效果、可制造性的端到端人工评估。

## 相关工作与启发
- **缝纫纸样生成**：优化类（Sensitive Couture、FoldSketch）依赖物理优化难规模化；自回归类（GarmentCode、DressCode、SewFormer、AIpparel、ChatGarment）用 Transformer 序列生成但推理慢、误差传播；前馈扩散类（StableGarment、SewingLDM）并行高效但几何约束不精——GarmentGPT 用离散 token 化在精度与效率间取得平衡。
- **结构化数据生成**：VLM 已被用于 3D 网格、骨架、矢量图、CAD 生成；本文把这套"离散表征 + 自回归"思路专门为纸样的几何+拓扑约束量身定制，启发是——任何"既要精确几何又要符号拓扑"的结构化生成都可考虑"先量化成词典、再让 LLM 选词"的范式。
- **隐空间生成**：直接对标 Latent Diffusion 的"映射到紧凑语义隐空间"哲学，把它从图像扩散迁移到了离散自回归的几何生成。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 首个把缝纫纸样生成 operationalize 到离散隐空间、用 VLM 自回归选 token 的框架，范式转换清晰且可迁移；但 RVQ-VAE + VLM 自回归的组件本身在邻域已有先例。
- **实验充分度**: ⭐⭐⭐⭐ 结构化 + 真实两套 benchmark、多骨干消融、残差层数消融都到位，提升幅度显著（真实场景 2.3×）；但真实照片样本偏合成、缺端到端可制造性评估。
- **写作质量**: ⭐⭐⭐⭐ 动机-范式-方法层层递进，token 化语法和损失定义清晰，图示完整。
- **价值**: ⭐⭐⭐⭐ 配套百万级数据集 + 新 benchmark + 代码开源，把数字服装设计从专家打版推向"随手拍照即生成可制造纸样"，落地价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] WeTok: Powerful Discrete Tokenization for High-Fidelity Visual Reconstruction](wetok_powerful_discrete_tokenization_for_high-fidelity_visual_reconstruction.md)
- [\[ICCV 2025\] Multimodal Latent Diffusion Model for Complex Sewing Pattern Generation](../../ICCV2025/image_generation/multimodal_latent_diffusion_model_for_complex_sewing_pattern_generation.md)
- [\[ICLR 2026\] ReDDiT: Rehashing Noise for Discrete Visual Generation](reddit_rehashing_noise_for_discrete_visual_generation.md)
- [\[ICLR 2026\] Long-Text-to-Image Generation via Compositional Prompt Decomposition](long-text-to-image_generation_via_compositional_prompt_decomposition.md)
- [\[ICLR 2026\] Discrete Adjoint Matching](discrete_adjoint_matching.md)

</div>

<!-- RELATED:END -->
