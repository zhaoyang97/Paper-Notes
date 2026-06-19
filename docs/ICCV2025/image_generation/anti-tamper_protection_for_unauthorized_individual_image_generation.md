---
title: >-
  [论文解读] Anti-Tamper Protection for Unauthorized Individual Image Generation
description: >-
  [ICCV 2025][图像生成][防篡改保护] 提出Anti-Tamper Perturbation (ATP)，在频域中将保护扰动（阻止伪造生成）和授权扰动（检测净化篡改）分离嵌入，当攻击者尝试净化保护信息时触发防篡改机制拒绝服务，在各种净化攻击下实现100%保护成功率。 随着个性化图像生成技术（如DreamBooth、…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "防篡改保护"
  - "个性化图像生成"
  - "对抗扰动"
  - "频域水印"
  - "图像版权"
---

# Anti-Tamper Protection for Unauthorized Individual Image Generation

**会议**: ICCV 2025  
**arXiv**: [2508.06325](https://arxiv.org/abs/2508.06325)  
**代码**: [https://github.com/Seeyn/Anti-Tamper-Perturbation](https://github.com/Seeyn/Anti-Tamper-Perturbation)  
**领域**: AI安全 / 图像生成防护  
**关键词**: 防篡改保护, 个性化图像生成, 对抗扰动, 频域水印, 图像版权

## 一句话总结
提出Anti-Tamper Perturbation (ATP)，在频域中将保护扰动（阻止伪造生成）和授权扰动（检测净化篡改）分离嵌入，当攻击者尝试净化保护信息时触发防篡改机制拒绝服务，在各种净化攻击下实现100%保护成功率。

## 研究背景与动机
随着个性化图像生成技术（如DreamBooth、Textual Inversion）的发展，在线服务商提供了便捷的定制人像生成功能。然而，伪造攻击者可以利用这些服务，盗用他人照片生成虚假人像，严重侵犯肖像权和隐私。

现有防护手段通过注入**保护扰动**来降低生成图像质量。但这种方法存在致命弱点：攻击者可以通过**净化技术**（如JPEG压缩、缩放、高斯模糊等）轻易移除保护扰动，恢复攻击能力。即使是MetaCloak等号称鲁棒的保护方法，在面对净化攻击时保护成功率仍会显著下降。

ATP的核心洞察是：与其试图让保护扰动抵抗净化（这是一场军备竞赛），不如转换思路——**当净化发生时检测到它**。这就像实物防伪封条：一旦被拆开（净化），留下的痕迹就会被服务商发现，从而拒绝服务请求。关键挑战在于：保护扰动本身会改变图像信息，这在概念上也是一种"篡改"，如何让授权扰动对保护扰动免疫、却对净化操作敏感？

## 方法详解

### 整体框架
ATP由两个组件构成：保护扰动$P_{Prot}$和授权扰动$P_{Auth}$。整个流程为：原始图像→BDCT变换到频域→用二值掩码分别在不同频域区域施加两种扰动→BIDCT变换回像素域→得到受保护的授权图像。下游服务商在处理生成请求时验证授权消息完整性，若被篡改则拒绝请求。

### 关键设计

1. **掩码引导的扰动融合（Mask-Guided Perturbation Blending）**:

    - 功能：确保保护扰动和授权扰动互不干扰
    - 核心思路：在频域中使用二值掩码$M$分离两种扰动的作用区域
    $P_{AP}(I) = F^{-1}[M \odot P_{Auth}(F(I)) + (1-M) \odot P_{Prot}(F(I))]$
    - 掩码$M$由Bernoulli分布采样（$p=0.5$），1的区域施加授权扰动，0的区域施加保护扰动
    - 在频域操作的优势：由于每个像素值是所有频率系数的线性组合，两种扰动在像素域中均匀分布，不可区分，防止攻击者选择性净化
    - 设计动机：直接在像素域分离扰动会导致两种扰动可被区分，攻击者可选择性地只净化保护扰动

2. **块离散余弦变换（Block Discrete Cosine Transform, BDCT）**:

    - 功能：高效地将图像变换到频域
    - 核心公式：将图像分割为$N\times N$的非重叠块（$N=16$），对每块应用DCT：
      $C_{u,v} = \alpha(u)\alpha(v)\sum_{i=0}^{N-1}\sum_{j=0}^{N-1}I_{i,j}\phi(u,i,N)\phi(v,j,N)$
    - 逆变换BIDCT将频域系数变换回像素域
    - 设计动机：对整幅图像直接做DCT计算量过大，BDCT按块处理更高效

3. **授权扰动（Authorization Perturbation）**:

    - 功能：在频域中嵌入授权消息，作为防篡改验证依据
    - 核心思路：使用卷积自编码器$f_\theta$在掩码指定的频域区域嵌入二进制授权消息$m$（长度$L$），同时训练消息解码器$D_m$提取消息
    - 编码过程：$g_\theta(C) = (1-M) \odot C + M \odot f_\theta(M \odot C, m)$，$I_{enc} = F^{-1}[g_\theta(F(I))]$
    - 损失函数：$\mathcal{L} = \mathcal{L}_{con} + \lambda_{adv}\mathcal{L}_{adv,G} + \lambda_{rec}\mathcal{L}_{rec} + \lambda_{reg}\mathcal{L}_{reg}$
    - 包含消息一致性损失$\mathcal{L}_{con} = \|D_m(M \odot f_\theta(C)) - m\|_2^2$和频域正则化$\mathcal{L}_{reg} = \|f_\theta(C) - C\|_2^2$
    - 与传统水印的区别：传统水印追求鲁棒性（抗净化），授权扰动反其道而行之，需要**对净化敏感**
    - 设计动机：嵌入可验证信息，净化操作会破坏信息完整性从而触发警报

4. **改进的频域PGD（Protection Perturbation）**:

    - 功能：在频域中准确生成保护扰动，避免干扰授权区域
    - 原始问题：标准PGD中的$\Pi(\cdot)$（投影）和$\text{sgn}(\cdot)$在像素域操作，会不可避免地影响掩码保护的频率系数
    - 改进方案（Algorithm 1）：将符号函数和投影约束移到频域执行
        - 先计算像素域梯度$\nabla$
        - 变换到频域并应用掩码：$\nabla_{freq} = M_p \odot F(\nabla)$
        - 在频域中应用符号函数和步长
        - 投影约束在频域中的$\epsilon$-球内执行
    - 设计动机：确保保护扰动只修改掩码指定的频率系数，不影响授权扰动区域

### 损失函数 / 训练策略
- 授权扰动网络在FFHQ数据集（70,000人脸图像）上训练
- 保护扰动使用CelebA-HQ和VGGFace2子集（每个数据集50人，每人8张图像）
- 基础生成模型：Stable Diffusion v2-1，个性化生成算法：DreamBooth
- ATP可与任意基于PGD的保护扰动算法集成（Anti-DB, AdvDM, CAAT, MetaCloak）

## 实验关键数据

### 主实验（净化攻击下的保护成功率PSR）

| 方法 | Clean | JPEG 50 | Resize 4x | GridPure |
|------|-------|---------|-----------|----------|
| Anti-DB | 高 | 下降 | 下降 | 下降 |
| Anti-DB + **ATP** | 高 | **100%** | **100%** | **100%** |
| MetaCloak | 高 | 部分下降 | 部分下降 | 下降 |
| MetaCloak + **ATP** | 高 | **100%** | **100%** | **100%** |

无净化时的保护性能（CelebA-HQ）：

| 方法 | CLIP-IQAC↓ | ISM↓ | FDFR↑ |
|------|-----------|------|-------|
| Anti-DB | -0.287 | 0.462 | 0.458 |
| Anti-DB+ATP | -0.314 | 0.465 | **0.521** |
| AdvDM | -0.336 | 0.417 | 0.664 |
| AdvDM+ATP | -0.362 | 0.412 | **0.668** |

### 消融实验

| 融合方案 | Bit-error (×$10^{-3}$) | 说明 |
|---------|----------------------|------|
| 无BDCT + 无掩码 + 无改进PGD | 349.84 | 像素域直接混合，授权信息严重损坏 |
| 有BDCT + 无掩码 | 42.03 | 频域有帮助但无掩码仍干扰 |
| 无BDCT + 有掩码 | 360.31 | 像素域掩码保护效果差 |
| 有BDCT + 有掩码 + 标准PGD | 81.72 | 标准PGD破坏掩码约束 |
| **完整ATP方案** | **0.47** | 三者结合极大降低Bit-error |

### 关键发现
- ATP使所有基线方法在净化攻击下达到100%保护成功率
- ATP不降低（甚至略微提升）无净化场景下的保护性能
- 频域授权扰动比像素域更有效嵌入信息且对净化更敏感
- 改进频域PGD是保证掩码约束的关键（Bit-error从81.72降至0.47）
- 自适应攻击：攻击者需同时知道BDCT参数和掩码值才能绕过（搜索空间约$2^{786414}$）
- ATP的美学影响小于原始保护扰动方法（CLIP-IQAC更高，更接近原图质量）

## 亮点与洞察
- 范式转换：从"抵抗净化"到"检测净化"，跳出了保护扰动与净化的军备竞赛
- 频域分离设计巧妙：利用DCT线性特性实现扰动在像素域的均匀分布，同时在频域中互不干扰
- 即插即用：ATP可与任意PGD-based保护算法集成，只需修改梯度下降过程
- 防篡改思想借鉴了实物防伪的直觉（如封条、防拆封设计）

## 局限与展望
- 依赖服务商主动验证——若攻击者在自己设备上运行生成模型则完全无效
- 自适应攻击中，若攻击者同时获知BDCT参数和掩码值（虽极不可能），保护失败
- 增加了部署端的计算开销（需执行授权验证流程）
- 未来方向：设计"被破坏后自动降低生成质量"的授权扰动，无需显式验证步骤

## 相关工作与启发
- **vs MetaCloak**: MetaCloak试图让保护扰动本身抵抗净化（鲁棒性路线），但PSR仍会下降；ATP通过检测机制实现100%
- **vs FaceSigns**: 传统水印追求鲁棒性（抗净化），ATP需要的恰好相反——对净化高敏感
- **vs GridPure**: 作为最强净化攻击，GridPure能显著降低各种保护方法的效果，但无法绕过ATP的防篡改检测

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次提出保护扰动的防篡改机制，范式创新显著
- 实验充分度: ⭐⭐⭐⭐ 覆盖4种保护算法、3种净化方式、自适应攻击、美学影响等全面评估
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，图例直观，技术路线循序渐进
- 价值: ⭐⭐⭐⭐ 为AI生成内容的版权保护提供了新的防御维度，具有实际部署价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] StableGuard: Towards Unified Copyright Protection and Tamper Localization in Latent Diffusion Models](../../NeurIPS2025/image_generation/stableguard_towards_unified_copyright_protection_and_tamper_localization_in_late.md)
- [\[NeurIPS 2025\] Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models](../../NeurIPS2025/image_generation/perturb_a_model_not_an_image_towards_robust_privacy_protection_via_anti-personal.md)
- [\[NeurIPS 2025\] BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Edit](../../NeurIPS2025/image_generation/blurguard_a_simple_approach_for_robustifying_image_protection_against_ai-powered.md)
- [\[CVPR 2026\] Adapter Shield: A Unified Framework with Built-in Authentication for Preventing Unauthorized Zero-Shot Image-to-Image Generation](../../CVPR2026/image_generation/adapter_shield_a_unified_framework_with_built-in_authentication_for_preventing_u.md)
- [\[CVPR 2025\] Nearly Zero-Cost Protection Against Mimicry by Personalized Diffusion Models](../../CVPR2025/image_generation/nearly_zero-cost_protection_against_mimicry_by_personalized_diffusion_models.md)

</div>

<!-- RELATED:END -->
