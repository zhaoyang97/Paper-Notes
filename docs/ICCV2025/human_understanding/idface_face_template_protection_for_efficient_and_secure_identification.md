---
title: >-
  [论文解读] IDFace: Face Template Protection for Efficient and Secure Identification
description: >-
  [ICCV 2025][人体理解][人脸模板保护] 提出 IDFace，一种基于同态加密（HE）的人脸模板保护方法，通过近等距变换（实值向量→三值向量）和空间高效编码两项技术，使 100 万加密模板的检索仅需 126ms，相比无保护仅 2× 开销。 领域现状：人脸识别系统（FRS）已广泛部署于机场、卡口等场景…
tags:
  - "ICCV 2025"
  - "人体理解"
  - "人脸模板保护"
  - "同态加密"
  - "生物特征隐私"
  - "大规模身份识别"
  - "三值量化"
---

# IDFace: Face Template Protection for Efficient and Secure Identification

**会议**: ICCV 2025  
**arXiv**: [2507.12050](https://arxiv.org/abs/2507.12050)  
**代码**: 无  
**领域**: Human Understanding / Face Recognition Security  
**关键词**: 人脸模板保护, 同态加密, 生物特征隐私, 大规模身份识别, 三值量化

## 一句话总结

提出 IDFace，一种基于同态加密（HE）的人脸模板保护方法，通过近等距变换（实值向量→三值向量）和空间高效编码两项技术，使 100 万加密模板的检索仅需 126ms，相比无保护仅 2× 开销。

## 研究背景与动机

**领域现状**：人脸识别系统（FRS）已广泛部署于机场、卡口等场景。随着 GDPR 等隐私法规出台，人脸模板的安全存储愈发重要。多项研究已证明，即使不知内部参数，也能从未保护的模板重建人脸图像，甚至用于冒充身份攻击商用系统。

**现有痛点**：同态加密（HE）理论上可在加密域做任意运算，但 HE 基本为多项式环上的代数运算设计，而人脸模板是实值向量（d=512, 在超球面上）。不加裁剪地将 HE 与 FRS 结合导致极低效率——以往 HE 方案比不加保护慢数百倍不等，在百万级身份库上根本不可用。

**核心矛盾**：HE 消息槽（message slot）位宽远大于模板表示所需位宽（2048-bit 槽 vs 16-bit 模板值），导致巨大的空间浪费和不必要的计算开销。此外，加密域的乘法成本远高于加法。

**本文切入角度**：针对 HE 与人脸模板的不匹配，提出两项互补技术：(1) 将实值单位向量变换为三值向量 {-1,0,1}^d，使内积运算退化为加法，彻底消除加密域乘法；(2) 利用变换后每个分量仅需 1 比特表示的特性，将多个模板打包进一个消息槽，最大化 SIMD 利用率。

## 方法详解

### 整体框架

IDFace 采用"数据库加密"范式：注册时服务端加密所有人脸模板，查询时在加密域计算匹配分数，密钥服务器解密找到最高分数的身份。系统分为本地服务器（存公钥+加密库）和密钥服务器（存私钥），满足 ISO 24745 安全要求。

### 关键设计

1. **近等距变换 $T_\alpha$**：将 $\mathbf{x} \in \mathbb{S}^{d-1}$ 变换为 $\mathcal{Z}_\alpha^d \subset \{-1,0,1\}^d$。具体做法：取绝对值最大的 α 个分量，保留其符号为 ±1，其余置零。作者证明当 d=512, α=341 时，变换为 (0.111, o(1), θ)-等距映射——即两向量内积的变化以极高概率不超过 0.111。

   设计动机：变换后内积 $\langle T_\alpha(\mathbf{x}), T_\beta(\mathbf{y})\rangle$ 仅需加减法（查表操作），避免了加密域中极为昂贵的乘法运算。

   $T_\alpha$ 数学分析：利用序统计量理论，作者严格证明了变换的近等距性质。对于 $d=512, \alpha=341$，$\epsilon = |cos\theta - 2P(\theta) \cdot d/\alpha|$ 的最大值为 0.111，失败概率 δ=o(1)。

2. **空间高效编码（Encode/Decode）**：变换后内积值范围为 [-α, α]，将三值向量拆为正向量 $\mathbf{x}^+$ 和负向量 $\mathbf{x}^-$（$\mathbf{x} = \mathbf{x}^+ - \mathbf{x}^-$），使内积值非负。然后以 $p > \alpha$ 为基，将 m 个模板打包编码到一个消息槽：$\mathbf{x}^\dagger = \sum_{i=1}^m p^{i-1} \cdot \mathbf{x}_i^\dagger$。

   对于 Paillier（2048-bit 消息空间），可打包 m=342 个模板；对于 CKKS（50-bit 精度），m=8。这使存储和计算效率同时大幅提升。

3. **查询加速（β < α）**：查询端可用更小的 β 值（如 63 或 127），进一步减少加法次数（从 2(α-1) 降到 2(β-1)），并增加打包数 m，以少量精度损失换取更快速度。

### 损失函数 / 训练策略

IDFace 是一个即插即用方法，不涉及模型训练。它与现有 FRS（ArcFace, AdaFace 等）直接兼容，不改变特征提取器。

## 实验关键数据

### 主实验：效率对比（百万身份库）

| 方法 | 注册时间 | 检索时间 | 存储 | 加密原语 |
|------|---------|---------|------|---------|
| 无保护 | N/A | 0.063s | 2GB | N/A |
| IronMask | 4,416s | 97.9s | 1024GB | FC |
| HERS (512) | 199s | 48.9s | 16.5GB | CKKS |
| MFBR-ID | 1641s | 0.545s | 132.1GB | BFV |
| **IDFace (63, CKKS)** | **72s** | **0.126s** | **4.125GB** | **CKKS** |

IDFace(β=63) 比 HERS 快 97.6×，比 MFBR-ID 快 4.5×，存储减少 31.8×。相比无保护仅 **2× 开销**。

### 消融实验：精度影响

| 数据集 | 指标 | 无保护 | (341,341) | (341,127) | (341,63) |
|--------|------|--------|-----------|-----------|----------|
| LFW | Accuracy | 99.82% | 99.78% | 99.80% | 99.82% |
| CFP-FP | Accuracy | 99.24% | 99.24% | 99.19% | 98.99% |
| AgeDB | Accuracy | 98.00% | 97.97% | 97.80% | 97.23% |
| IJB-C(V) | TAR@1e-3 | 98.39% | 98.27% | 98.14% | 97.78% |

在 LFW/CFP-FP/AgeDB 上精度损失均 < 1%，即使在最激进的 β=63 设置下。IJB-C 识别场景误差稍大但仍可接受。

### 关键发现

- α=341 比 α=512 效果更好（去掉绝对值小的噪声分量反而提升），验证了"少数关键分量主导内积"的直觉。
- 对多种特征提取器（ArcFace, MagFace, SphereFace2, ElasticFace, AdaFace-KPRPE）均有一致的低精度损失表现，证明方法的通用性。
- 安全分析：基于 AHE 的 IND-CPA 安全性，满足不可逆性、可撤销性和不可链接性三项 BTP 标准安全要求。

## 亮点与洞察

- **2× 开销突破**：首次将 HE 保护的百万级人脸检索效率拉到与无保护接近的量级，具有实际部署价值。
- **即插即用**：不需要重训 FRS，对开放集场景友好——用户模板和训练数据完全独立。
- 三值量化的**近等距性理论分析**独具价值，可推广到其他基于内积的检索任务。

## 局限与展望

- 当前仅考虑被动攻击者（窃取加密数据库），未分析主动攻击者（篡改查询/响应）的安全性。
- 两服务器模型要求密钥服务器绝对安全，实际部署需硬件安全模块（HSM）支持。
- 变换引入的精度损失在极端低 FAR 要求（如 1e-4）下更为明显。

## 相关工作与启发

- HERS [Engelsma 2022] 提出按列加密数据库的矩阵-向量乘积思路，IDFace 在此基础上通过三值化消除乘法。
- MFBR [Bassit 2022-2025] 也使用查找表消除乘法，但存储开销过大（132GB vs 4GB）。
- 变换 $T_\alpha$ 在技术上等同于 IronMask 的纠错码解码算法，但 IDFace 首次发现并利用其近等距性质。

## 评分

- 新颖性：⭐⭐⭐⭐ — 三值量化+高效编码的组合创新
- 技术深度：⭐⭐⭐⭐⭐ — 严格的理论分析（近等距性证明、安全性论证）
- 实验充分度：⭐⭐⭐⭐⭐ — 多方法对比、多数据集、多特征提取器、效率+精度双维度
- 实用性：⭐⭐⭐⭐⭐ — 2× 开销使大规模安全人脸检索真正可用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Efficient Video Face Enhancement with Enhanced Spatial-Temporal Consistency](../../CVPR2025/human_understanding/efficient_video_face_enhancement_with_enhanced_spatial-temporal_consistency.md)
- [\[ICCV 2025\] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [\[AAAI 2026\] CLIP-FTI: Fine-Grained Face Template Inversion via CLIP-Driven Attribute Conditioning](../../AAAI2026/human_understanding/clip-fti_fine-grained_face_template_inversion_via_clip-driven_attribute_conditio.md)
- [\[ICCV 2025\] One-Shot Knowledge Transfer for Scalable Person Re-Identification](one-shot_knowledge_transfer_for_scalable_person_re-identification.md)
- [\[ICML 2026\] Efficient, Validation-Free Intrinsic Quality Estimation for Large-Scale Face Recognition Datasets](../../ICML2026/human_understanding/efficient_validation-free_intrinsic_quality_estimation_for_large-scale_face_reco.md)

</div>

<!-- RELATED:END -->
