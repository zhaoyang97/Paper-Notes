---
title: >-
  [论文解读] DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction
description: >-
  [ICLR 2026][医学图像][CT重建] 提出DM4CT——首个系统性的CT重建扩散模型基准，涵盖十种扩散方法和七种基线方法，在医疗、工业和同步辐射三类数据集上进行全面评估，揭示了扩散模型在CT重建中的优势与局限。 CT重建是典型的逆问题，从投影测量中恢复未知物体。当测量稀疏或含噪时，问题是病态的，需要先验知识…
tags:
  - "ICLR 2026"
  - "医学图像"
  - "CT重建"
  - "扩散模型"
  - "benchmark"
  - "逆问题"
  - "稀疏视图重建"
---

# DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction

**会议**: ICLR 2026  
**arXiv**: [2602.18589](https://arxiv.org/abs/2602.18589)  
**代码**: [有](https://github.com/DM4CT/DM4CT)  
**领域**: 医学图像  
**关键词**: CT重建, 扩散模型, benchmark, 逆问题, 稀疏视图重建

## 一句话总结

提出DM4CT——首个系统性的CT重建扩散模型基准，涵盖十种扩散方法和七种基线方法，在医疗、工业和同步辐射三类数据集上进行全面评估，揭示了扩散模型在CT重建中的优势与局限。

## 研究背景与动机

CT重建是典型的逆问题，从投影测量中恢复未知物体。当测量稀疏或含噪时，问题是病态的，需要先验知识。先验方法从经典正则化（TV）发展到深度学习（监督学习、DIP），再到最近的扩散模型。

扩散模型在图像生成领域成功后被引入逆问题求解，但CT成像面临特殊挑战：**相关噪声、伪影结构、系统几何依赖、值域不匹配**等，使直接应用扩散模型比自然图像生成困难得多。然而缺乏统一基准来系统评估各种扩散方法。

**核心贡献**：不是提出新算法，而是构建首个系统性基准，回答"扩散模型在CT中到底表现如何"。

## 方法详解

### 整体框架

DM4CT本身不提新算法，而是把CT重建放回贝叶斯框架里横向评测各路扩散方法：后验 $p(\boldsymbol{x}|\boldsymbol{y}) \propto p(\boldsymbol{x})p(\boldsymbol{y}|\boldsymbol{x})$ 中，扩散模型提供先验分数 $\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{x}_t)$，逆向SDE被改造成条件逆向SDE，所有方法的差别本质上都落在如何近似那一项难算的测量条件分数 $\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{y}|\boldsymbol{x}_t)$ 上。基准做的就是把这个统一视角、可控的数据/几何配置、以及一套公平的实现拼到一起，得到"扩散模型在CT里到底行不行"的可复现答案。

### 关键设计

**1. 统一分类体系：把十种扩散方法的设计选择摊开对比**

这些方法表面上五花八门，但只要追问"测量条件分数怎么近似"，就能收敛成五条主线。最常见的是**数据一致性梯度引导（DC-grad）**，在每步去噪后由当前估计 $\hat{\boldsymbol{x}}_0$ 算数据保真梯度 $\boldsymbol{g}_t = \nabla_{\boldsymbol{x}_t}\mathcal{L}(\boldsymbol{A}\hat{\boldsymbol{x}}_0 - \boldsymbol{y})$ 沿轨迹推一把，用步长 $\eta$ 调引导强度，代表是DPS、PSLD；更硬的做法是**数据一致性优化步（DC-step）**，在去噪迭代间塞进一个完整的最小化 $\boldsymbol{x}_t^* = \arg\min \mathcal{L}(\boldsymbol{A}\boldsymbol{x}_t - \boldsymbol{y})$，把测量约束彻底吃进去，代表是ReSample；介于两者之间还有解耦先验与保真、交替求解的**即插即用（DMPlug）**，借伪逆重建（FBP/SIRT近似）在测量空间和图像空间间搭桥的**伪逆引导（PGDM、MCG）**，以及干脆用参数化分布近似后验、不沿轨迹逐步采样的**变分贝叶斯（Reddiff）**。这条主线让"为什么某方法在某配置下崩"有了可解释的坐标，而不是黑箱跑分。

**2. 三类数据集加五种仿真配置：覆盖从医疗到工业的分布差异与退化谱**

数据上刻意拉开域差距——医疗CT用2016 Low Dose CT Challenge（9卷训练、1卷测试，512×512），工业CT用LoDoInd管状多材料样品（3000训练、500测试切片），还新采集了一套同步辐射CT（两块岩石样品在高能同步辐射设施扫描，768×768高分辨率），后者填补了现有评估几乎只有仿真、缺真实测量的空白。退化上设计五种递进配置系统压测鲁棒性：40角度无噪、20角度轻噪、80角度强噪、80角度噪声叠环形伪影、以及40角度限角。这样每种方法都要在稀疏、含噪、有结构性伪影、几何不全等多重病态下亮相，避免只在某个甜区刷高分。

**3. 七种强基线：让扩散方法跟经典、迭代、监督学派同台**

为了判断扩散到底带来多少增益，基准配齐了各范式的代表：解析法FBP与代数迭代SIRT打底，神经网络先验DIP和隐式表示INR代表无监督学习，R2Gaussian代表高斯散布新路线，FISTA-SBTV与ADMM-PDTV代表带正则的迭代重建，监督学习则用SwinIR作上限参照。所有方法统一在diffusers框架下实现、共享同一前向算子和评测脚本，保证对比的公平性，也方便后续接入新方法。

## 实验关键数据

### 主实验

**医疗数据集重建性能（PSNR/SSIM，部分配置）**

| 方法 | Config i (40角无噪) | Config ii (20角轻噪) | Config iv (80角噪声+环形) |
|------|:---:|:---:|:---:|
| FBP | 26.98/0.69 | 9.89/0.03 | 14.50/0.13 |
| SIRT | 30.40/0.80 | 26.23/0.47 | 25.86/0.40 |
| SwinIR (监督) | 32.45/0.88 | **29.92/0.83** | **30.79/0.85** |
| DDS (扩散最佳) | **31.43/0.84** | - | - |
| ReSample | 32.03/0.85 | 27.92/0.73 | 29.70/0.76 |
| INR | 33.21/0.86 | 26.15/0.76 | 29.50/0.74 |

**同步辐射真实数据（PSNR/SSIM）**

| 方法 | 200投影 | 100投影 | 60投影 |
|------|:---:|:---:|:---:|
| SwinIR | **33.75/0.76** | **33.05/0.73** | **32.41/0.70** |
| Reddiff | 28.43/0.56 | 28.24/0.54 | 28.06/0.51 |
| DDS | 28.36/0.55 | 28.10/0.51 | 27.90/0.49 |
| SIRT | 28.16/0.56 | 28.06/0.54 | 27.92/0.52 |

### 消融实验

**先验与数据一致性权衡**：以DPS为例，步长 $\eta$ 过小则先验主导（模糊），过大则测量噪声主导（崩溃）。最优 $\eta$ 需精细调节。

**像素空间 vs 潜空间扩散**：
- 潜空间（PSLD）：梯度需经VQ-VAE解码器传播，无噪声条件下也产生不连续伪影
- 优化步（ReSample）可修复不连续，但有噪声时过拟合测量

**零空间分析**：DC-grad（DPS）允许更多零空间内容，DC-step（ReSample）更严格约束，伪逆（PGDM）居中。

### 关键发现

1. **无单一扩散方法全面最优**，性能因数据集和配置而异
2. 扩散模型整体略优于经典/MBIR方法，但通常不如全监督SwinIR
3. 扩散模型恢复的细节虽然视觉逼真但可能偏离真值，导致指标不如INR/SwinIR的平滑重建
4. 真实数据上性能普遍低于仿真数据，暴露训练数据质量和分布偏移问题
5. 像素扩散通常比潜空间扩散更节省内存和时间

## 亮点与洞察

- **首个系统性CT扩散基准**：统一代码框架（diffusers），公平对比，开源代码和数据
- **统一分类体系**清晰梳理了方法间的设计选择和权衡
- **真实同步辐射数据集**弥补了现有评估缺乏真实数据的空白
- **深刻的实践洞察**：揭示了值域不匹配、有限训练数据、几何复杂性等真实部署挑战
- **零空间分析**提供了理解不同数据一致性策略的新视角

## 局限与展望

- 仅评估2D切片重建，未涉及3D重建（螺旋/锥束几何更具挑战性）
- 未包含flow-based方法（如FlowDPS），这是新兴方向
- 临床相关性评估不足（未做分割/放射科医师评分等下游任务评价）
- 扩散模型训练成本高，潜空间模型总训练时间更长
- 自然图像预训练的自编码器未必适合CT数据
- 跨设备/跨协议泛化性未测试

## 相关工作与启发

- 将DIP中DC loss的思想与扩散模型结合是有趣方向（参照DC loss论文）
- INR+扩散先验的混合方法可能兼具结构保真和细节恢复
- 扩散模型的不确定性量化能力（多次采样取均值/方差）有临床价值
- 对稀疏视图和高噪声场景，学习型先验优势最大；密集/低噪声时经典方法已足够

## 评分

| 维度 | 分数 |
|------|:---:|
| 创新性 | ★★★★☆ |
| 理论深度 | ★★★☆☆ |
| 实验充分性 | ★★★★★ |
| 实用价值 | ★★★★★ |
| 写作质量 | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?](../../NeurIPS2025/medical_imaging/are_pixel-wise_metrics_reliable_for_sparse-view_computed_tomography_reconstructi.md)
- [\[ICLR 2026\] OpenPros: A Large-Scale Dataset for Limited View Prostate Ultrasound Computed Tomography](openpros_a_large-scale_dataset_for_limited_view_prostate_ultrasound_computed_tom.md)
- [\[ICLR 2026\] U2-BENCH: Benchmarking Large Vision-Language Models on Ultrasound Understanding](u2-bench_benchmarking_large_vision-language_models_on_ultrasound_understanding.md)
- [\[ICLR 2026\] Moving Beyond Diffusion: Hierarchy-to-Hierarchy Autoregression for fMRI-to-Image Reconstruction](moving_beyond_diffusion_hierarchy-to-hierarchy_autoregression_for_fmri-to-image_.md)
- [\[ICLR 2026\] Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation](adaptive_domain_shift_in_diffusion_models_for_cross-modality_image_translation.md)

</div>

<!-- RELATED:END -->
