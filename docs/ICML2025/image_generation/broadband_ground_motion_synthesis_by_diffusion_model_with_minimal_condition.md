---
title: >-
  [论文解读] Broadband Ground Motion Synthesis by Diffusion Model with Minimal Condition
description: >-
  [ICML2025][图像生成][扩散模型] 提出 HEGGS（High-fidelity Earthquake Groundmotion Generation System），利用地震数据集中波形天然可配对的特性，结合条件隐扩散模型与 ACM 振幅校正模块，仅需最少条件信息（经纬度、震源深度、震级）即可端到端生成高保真三分量地震波形。
tags:
  - "ICML2025"
  - "图像生成"
  - "扩散模型"
  - "seismic waveform"
  - "ground motion synthesis"
  - "paired data"
---

# Broadband Ground Motion Synthesis by Diffusion Model with Minimal Condition

**会议**: ICML2025  
**arXiv**: [2412.17333](https://arxiv.org/abs/2412.17333)  
**代码**: 未开源  
**领域**: 扩散模型 / 地震波形合成  
**关键词**: diffusion model, seismic waveform, ground motion synthesis, latent diffusion, paired data

## 一句话总结

提出 HEGGS（High-fidelity Earthquake Groundmotion Generation System），利用地震数据集中波形天然可配对的特性，结合条件隐扩散模型与 ACM 振幅校正模块，仅需最少条件信息（经纬度、震源深度、震级）即可端到端生成高保真三分量地震波形。

## 研究背景与动机

- **数据稀缺问题**：大震级地震本身罕见，导致训练数据不足，现有方法（主要基于 GAN）生成的波形缺乏地震学真实性（如 P/S 波到时不准、振幅失真）。
- **条件信息昂贵**：先前方法通常需要震源机制（focal mechanism）、VS30 等额外地质信息，获取成本高且不总可用。
- **生成质量不足**：SeismoGen、ConSeisGen、BBGAN 等基于 GAN 的方法在相位到时、包络相关性等地震学指标上表现不佳。
- **核心动机**：利用区域地震数据集中「同一地震事件被多个台站观测」的天然配对特性，让扩散模型从配对波形中隐式学习震源特征，无需显式提供昂贵的地质条件。

## 方法详解

### 最小条件集

仅需 6 个条件参数：
- 台站经纬度 $(s_{lat}, s_{lon})$
- 震源经纬度 $(e_{lat}, e_{lon})$
- 震源深度 $e_{dep}$
- 地震震级 $M_L$

### 配对扩散模型（Pair-Exploiting Diffusion）

对同一地震事件，从数据集采样一对波形 $(W^{src}, W^{tgt})$，转换为频谱图 $(X^{src}, X^{tgt})$。假设从 $X^{src}$ 和目标条件向量 $\vec{c}_{tgt}$ 可提取足够的地震特征来生成 $X^{tgt}$，定义潜在变换映射 $\eta$：

$$\eta(x_t^{src}, \vec{c}_{tgt}, t) \sim q(x_t^{tgt} | X^{tgt})$$

扩散损失在样本空间上：

$$\mathcal{L}_{DM}' = \mathbb{E}_{(X^{src}, X^{tgt}, \vec{c}_{tgt}), \epsilon, t} \| X^{tgt} - \mathbf{m}_\theta(x_t^{src}, \vec{c}_{tgt}, t) \|^2$$

其中 $\mathbf{m}_\theta(x, \vec{c}, t) = \mathbf{x}_\theta(\eta(x, \vec{c}, t), \vec{c}, t)$，是潜在变换 + 去噪模型的复合。

### 端到端训练

引入自编码器（编码器 $\mathcal{E}_{AE}$ + 解码器 $\mathcal{D}_{AE}$），在隐空间进行扩散，但采用端到端损失直接监督波形重建：

$$\mathcal{L}_{ours} = \mathbb{E}_{(X^{src}, X^{tgt}, \vec{c}_{tgt}), \epsilon, t} \| X^{tgt} - \mathcal{D}_{AE}(\mathbf{m}_\theta(z_t^{src}, \vec{c}_{tgt}, t)) \|^2$$

其中 $z_t^{src} = \sqrt{\bar\alpha_t} \mathcal{E}_{AE}(X^{src}) + \sqrt{1 - \bar\alpha_t} \epsilon$。

### 振幅校正模块（ACM）

解码器后接 ACM 进行振幅校正，解决预训练 VAE 无法捕获振幅信息的问题，使模型能直接生成未归一化的真实振幅波形。

### 推理流程

- **有参考波形**：以 $W^{src}$ 编码后加噪作为起点，配合 $\vec{c}_{tgt}$ 反向去噪
- **无参考波形**：从纯高斯噪声出发，仅凭条件向量生成（可用于虚拟地震模拟）

## 实验关键数据

### 数据集

三大洲三个地震数据库：SCEDC（北美）、KMA（东亚）、INSTANCE（欧洲）。

### 定量对比（SCEDC 数据集，部分结果）

| 模型 | 输入 | P_MAE(s)↓ | S_MAE(s)↓ | env.corr↑ | MSE↓ |
|------|------|-----------|-----------|-----------|------|
| SeismoGen | w/o $W^{src}$ | 1.956 | 3.625 | 0.490 | 1.412 |
| ConSeisGen | w/o $W^{src}$ | 3.972 | 6.899 | 0.325 | 0.746 |
| BBGAN | w/o $W^{src}$ | 6.421 | 10.42 | 0.195 | 1.615 |
| LDM | w/ $W^{src}$ | 0.563 | 0.781 | 0.773 | 0.243 |
| **HEGGS** | **w/o** $W^{src}$ | **0.503** | **0.800** | **0.796** | **0.153** |
| **HEGGS** | **w/** $W^{src}$ | **0.476** | **0.548** | **0.819** | **0.151** |

- HEGGS 在所有三个数据集上全面领先，P/S 波到时误差比 GAN 方法降低一个数量级。
- 即使不使用参考波形（纯噪声生成），HEGGS 也优于所有基线带参考波形的结果。

### 消融实验（SCEDC）

| 配置 | P_MAE(s) | S_MAE(s) | env.corr |
|------|----------|----------|----------|
| LDM（归一化波形） | 1.114 | 1.729 | 0.693 |
| + 配对数据 | 0.563 | 0.781 | 0.773 |
| + 端到端训练 | 0.801 | 1.537 | 0.624 |
| + ACM（=HEGGS） | **0.476** | **0.548** | **0.819** |

配对训练使相位到时精度翻倍；ACM 使模型能处理真实振幅并进一步提升质量。

## 亮点与洞察

1. **巧妙利用数据特性**：同一地震事件被多台站同时记录的天然配对性，让模型在训练时隐式学到震源特征，无需显式提供昂贵条件。
2. **端到端训练绕过预训练瓶颈**：地震领域缺乏预训练 VAE，端到端方案让编码器、扩散模块、解码器联合优化。
3. **双模式推理**：既可用已有地震波形做条件生成（精度更高），也可纯噪声起步合成虚拟地震（应用更灵活）。
4. **震级操控能力**：修改条件中的 $M_L$ 即可生成不同震级的波形，频谱分析显示角频率和地震矩 $M_0$ 变化趋势符合理论预期。
5. **单 GPU 可训练**：在单 GPU 上即可完成训练，计算效率高。

## 局限与展望

- **区域局限**：模型在各区域数据上独立训练，尚未实现全球统一模型，跨区域泛化能力未验证。
- **条件有限**：最小条件集虽降低使用门槛，但可能限制了极端场景（如超大震级外推）的生成质量。
- **频谱分辨率**：合成频谱图分辨率略低于真实数据，细粒度高频特征还原不够精细。
- **下游验证缺失**：虽然声称可用于预警系统和抗震设计，但未展示实际下游任务的效果。
- **无代码开源**：可复现性受限。

## 相关工作与启发

- **SeismoGen / ConSeisGen / BBGAN**：GAN 系地震波形生成方法，均作为基线被超越。
- **Latent Diffusion Model (LDM)**：Rombach et al., 2022，HEGGS 在其基础上引入配对训练和端到端学习。
- **音乐生成启发**：方法设计受 Ghosal et al., 2023 条件音乐生成方法启发（先生成频谱图，再转换为波形）。
- **EQTransformer**：用于评估生成波形的 P/S 波到时质量。
- **跨领域启示**：扩散模型在「信号生成」任务上的迁移潜力——同样的 LDM 框架可从图像迁移到地震波形。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 配对训练策略和端到端框架在地震领域有新意
- 实验充分度: ⭐⭐⭐⭐ — 三大洲数据集 + 多种地震学指标 + 消融实验，较为全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式推导完整，地震学背景介绍充分
- 价值: ⭐⭐⭐⭐ — 对地震工程和灾害预警有实际应用潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Model Immunization from a Condition Number Perspective](model_immunization_from_a_condition_number_perspective.md)
- [\[ICCV 2025\] OminiControl: Minimal and Universal Control for Diffusion Transformer](../../ICCV2025/image_generation/ominicontrol_minimal_and_universal_control_for_diffusion_transformer.md)
- [\[ICML 2025\] DDIS: When Model Knowledge Meets Diffusion Model](when_model_knowledge_meets_diffusion_model_diffusion-assisted_data-free_image_synthesis.md)
- [\[CVPR 2026\] D2C: Accelerating Diffusion Model Training under Minimal Budgets via Condensation](../../CVPR2026/image_generation/d2c_diffusion_dataset_condensation.md)
- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](../../ECCV2024/image_generation/smoodi_stylized_motion_diffusion_model.md)

</div>

<!-- RELATED:END -->
