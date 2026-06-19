---
title: >-
  [论文解读] TAR3D: Creating High-Quality 3D Assets via Next-Part Prediction
description: >-
  [ICCV2025][3D视觉][3D生成] 提出TAR3D框架——首次将三平面表示量化为离散几何部件并用GPT自回归生成，通过3D VQ-VAE编码任意面数网格为固定长度序列+TriPE位置编码保留3D空间信息，在文本/图像→3D任务上全面超越现有方法。 领域现状：条件3D生成已取得显著进展，主要分为三类路线：(1) SD…
tags:
  - "ICCV2025"
  - "3D视觉"
  - "3D生成"
  - "自回归"
  - "VQ-VAE"
  - "三平面表示"
  - "下一部件预测"
  - "GPT"
---

# TAR3D: Creating High-Quality 3D Assets via Next-Part Prediction

**会议**: ICCV2025  
**arXiv**: [2412.16919](https://arxiv.org/abs/2412.16919)  
**代码**: [GitHub](https://github.com/HVision-NKU/TAR3D)  
**领域**: 3D视觉  
**关键词**: 3D生成, 自回归, VQ-VAE, 三平面表示, 下一部件预测, GPT

## 一句话总结
提出TAR3D框架——首次将三平面表示量化为离散几何部件并用GPT自回归生成，通过3D VQ-VAE编码任意面数网格为固定长度序列+TriPE位置编码保留3D空间信息，在文本/图像→3D任务上全面超越现有方法。

## 研究背景与动机

**领域现状**：条件3D生成已取得显著进展，主要分为三类路线：(1) SDS优化(DreamFusion等)但耗时且有多面问题；(2) 多视图合成(Zero123等)但依赖视图一致性、易丢失细节；(3) 3D原生扩散(Direct3D/Clay等)但与LLM统一建模困难。

**现有痛点**：MeshGPT/MeshXL等尝试将LLM自回归引入3D，但直接量化网格面的顶点→序列长度=9×面数，工业级模型动辄几十万面→序列爆炸、不可扩展。

**核心问题**：能否将任意面数的3D网格编码为固定长度序列，让GPT自回归建模3D目标成为可能？

**关键洞察**：三平面表示(triplane)天然具备固定尺寸压缩3D信息的能力，且其2D特征图结构适合借鉴图像VQ量化策略→可以实现面数无关的固定长度离散序列。

## 方法详解

### 整体架构
TAR3D包含两个核心模块：**3D VQ-VAE**（将3D形状编码为离散三平面特征）和**3D GPT**（自回归预测码本索引序列）。训练分两阶段：先训VQ-VAE得到好的离散表示，再训GPT学习序列生成。

### 3D VQ-VAE

#### 3D形状编码器
- 输入：从3D网格表面均匀采样81,920个点云(含法线)，P ∈ R^{B×Np×6}
- 对点云施加Fourier位置编码以捕获高频细节
- Transformer架构：1层交叉注意力 + 8层自注意力
- 点云信息通过交叉注意力注入可学习query token（3×32×32=3072个token，通道768）
- 输出：三平面潜在表示 ẑ ∈ R^{B×3072×dz}

#### 量化器
- 可学习码本 Z = {z_k}_{k=1}^K，K=16384，通道dq=8
- 线性投影将连续特征映射到码本维度
- 逐元素最近邻量化：zq = argmin ||z̃_ij - z_k||
- **序列长度固定为3×32×32=3072**，与网格面数完全无关

#### 3D几何解码器
- N2=6层注意力层，每层包含2次特征变形+自注意力
- **平面内自注意力**：忽略平面轴→每个平面独立处理
- **平面间自注意力**：沿高度维度拼接三个平面→跨平面信息交互(PII)
- 上采样三平面特征到256×256分辨率
- 从三平面双线性插值采样查询点特征→MLP预测占据值
- 查询点=20480体素点 + 20480近表面点

### 3D GPT

#### 序列构建
- 基于预训练VQ-VAE的码本索引构建序列
- **平面内**：光栅扫描顺序(raster scan)
- **平面间**：三个平面同位置索引相邻排列
- 条件prompt作为预填充(prefilling) token嵌入

#### TriPE位置编码
- 融合2D RoPE + 1D RoPE的自定义3D位置编码
- **TriP_2D**：将2D位置编码每个元素重复3次并相邻排列→保留平面内2D空间信息
- **TriP_1D**：将1D位置编码(3个元素)各重复h×w次→区分三个特征平面
- **TriPE = TriP_2D + TriP_1D**（逐元素加法）
- 比朴素1D RoPE能保留更多3D空间信息

#### 自回归生成
- 序列 s ∈ {0,...,K-1,K}^{3·h·w}
- GPT-L架构(LLamaGen设定)：24层Transformer，16头，维度1024
- 条件编码：DINO(ViT-B16)编码图像 / FLAN-T5 XL编码文本
- 推理使用CFG(scale=7.5)提升质量和对齐度
- 训练目标：最大化三平面索引序列的对数似然

### 训练细节
- VQ-VAE损失：L = λ_rec · L_rec(BCE) + λ_cb · L_cb(码本学习)
- λ_rec=1, λ_cb=0.1, β=0.25
- 优化器：AdamW，学习率1e-4，CosineAnnealing
- VQ-VAE：batch=128，100K步，8×A100
- GPT：batch=80，100K步

## 实验关键数据

### 数据集

| 数据集 | 规模 | 用途 |
|--------|------|------|
| ShapeNet | 52,472网格/55类 | 训练48,597 + 测试2,592 |
| Objaverse | ~100K(从800K筛选) | 训练~99K + 评估1K |
| GSO | ~1,000真实扫描 | 域外泛化验证 |

### Image-to-3D定量对比(Tab.1)

| 方法 | PSNR↑ | SSIM↑ | CLIP↑ | LPIPS↓ | CD↓ | F-Score↑ |
|------|-------|-------|-------|--------|------|----------|
| Shap-E | 10.991 | 0.702 | 0.834 | 0.325 | 0.156 | 0.163 |
| SyncDreamer | 11.269 | 0.706 | 0.837 | 0.320 | 0.158 | 0.178 |
| Michelangelo | 11.928 | 0.734 | 0.864 | 0.278 | 0.117 | 0.226 |
| InstantMesh | 11.560 | 0.721 | 0.847 | 0.303 | 0.137 | 0.179 |
| LGM | 11.363 | 0.714 | 0.841 | 0.317 | 0.149 | 0.172 |
| **TAR3D** | **13.626** | **0.763** | **0.868** | **0.216** | **0.066** | **0.303** |

- **PSNR提升1.7+**，**CD降低43%+**，**F-Score提升34%+**，全指标大幅领先

### 消融实验

| 消融项 | CD↓ | F-Score↑ |
|--------|------|----------|
| 3D VAE(无量化) | 0.018 | 0.811 |
| **3D VQ-VAE** | **0.016** | **0.822** |
| 无PII(平面信息交互) | 0.023 | 0.661 |
| **有PII** | **0.016** | **0.822** |

### 三平面尺寸消融

| 三平面尺寸 | CD↓ | 推理时间 |
|------------|------|----------|
| 3×16×16 | 0.157 | 17.7s |
| **3×32×32** | **0.066** | **67.6s** |
| 3×48×48 | 0.062 | 143.9s |

- 32×32是质量/效率最佳平衡点：CD从0.157降到0.066，时间可接受；48×48仅微降但时间翻倍。

## 亮点与洞察

1. **固定长度序列是关键突破**：MeshGPT序列长度∝面数→不可扩展；TAR3D通过三平面+VQ量化将任意面数压缩为3072个token→真正可扩展的3D自回归。

2. **TriPE的巧妙设计**：不是简单用1D位置编码处理展平序列，而是融合2D(平面内空间)+1D(跨平面身份)→消融实验显示朴素1D RoPE丢失了关键几何细节。

3. **PII的重要性**：F-Score从0.661→0.822(+24%)→三个平面之间的信息交互对高质量重建至关重要，不能独立处理。

4. **VQ-VAE反超VAE**：通常认为量化会损失信息，但TAR3D的VQ-VAE重建质量竟然超过连续VAE(CD 0.016 vs 0.018)→说明离散码本学到了更好的几何部件表示。

5. **GPT范式统一多模态3D生成**：图像/文本统一作为prefilling token→自然支持多模态条件生成，且与LLM生态兼容。

## 局限与展望

1. **仅生成几何无纹理**：当前版本只生成形状，纹理依赖外部合成器(SyncMVD)→需要端到端纹理生成。
2. **推理效率**：32×32设置需67.6秒→对比扩散方法的几秒生成偏慢，自回归的序列依赖限制了并行化。
3. **占据场表示**：用占据场而非SDF/DMTet→可能限制拓扑复杂度和薄结构重建。
4. **数据规模受限**：仅用~100K Objaverse数据→论文提到未来将引入Objaverse-XL等进行Scaling Law探索。
5. **未探索更大GPT**：仅用GPT-L(24层/1024维)→Scaling行为和涌现能力有待验证。

## 相关工作与启发

- **MeshGPT/MeshXL**：直接量化网格面→序列过长；TAR3D用三平面绕开了这个根本瓶颈
- **Direct3D/Clay**：3D扩散路线→TAR3D提供了自回归的替代方案，与LLM统一更自然
- **LLamaGen**：图像自回归生成的成功验证→TAR3D将其扩展到3D领域
- **启发**：三平面+VQ的思路可扩展到4D(动态3D)生成；TriPE的位置编码设计可用于其他结构化序列任务

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次三平面量化+GPT自回归3D生成，解决序列长度瓶颈
- 实验充分度: ⭐⭐⭐⭐ 多数据集+多基线对比+充分消融，但缺少更大规模实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分，图示直观
- 价值: ⭐⭐⭐⭐⭐ 为3D生成与LLM统一开辟了可行路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] TAPNext: Tracking Any Point (TAP) as Next Token Prediction](tapnext_tracking_any_point_tap_as_next_token_prediction.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[NeurIPS 2025\] ARMesh: Autoregressive Mesh Generation via Next-Level-of-Detail Prediction](../../NeurIPS2025/3d_vision/armesh_autoregressive_mesh_generation_via_next-level-of-detail_prediction.md)
- [\[NeurIPS 2025\] Dynamics of Spontaneous Topic Changes in Next Token Prediction with Self-Attention](../../NeurIPS2025/3d_vision/dynamics_of_spontaneous_topic_changes_in_next_token_prediction_with_self-attenti.md)
- [\[CVPR 2026\] PointNSP: Autoregressive 3D Point Cloud Generation with Next-Scale Level-of-Detail Prediction](../../CVPR2026/3d_vision/pointnsp_autoregressive_3d_point_cloud_generation_with_next-scale_level-of-detai.md)

</div>

<!-- RELATED:END -->
