---
title: >-
  [论文解读] Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models
description: >-
  [ICML2025][图像生成][inference-time scaling] 系统性研究了在不依赖外部模型（VLM/CLIP）的前提下，对文本到图像扩散模型的初始噪声优化算法施加 Best-of-N 推理时缩放的效果，发现性能会迅速达到平台期（plateau），少量优化步数即可逼近该设置下的最大性能，且不同底层扩散模型上的最优算法不同。
tags:
  - "ICML2025"
  - "图像生成"
  - "inference-time scaling"
  - "扩散模型"
  - "initial noise optimization"
  - "Best-of-N"
  - "performance plateau"
  - "注意力机制"
---

# Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models

**会议**: ICML2025  
**arXiv**: [2506.12633](https://arxiv.org/abs/2506.12633)  
**作者**: Changhyun Choi, Sungha Kim, H. Jin Kim
**代码**: [initno official](https://github.com/xiefan-guo/initno)  
**领域**: 图像生成  
**关键词**: inference-time scaling, text-to-image diffusion, initial noise optimization, Best-of-N, performance plateau, attention maps

## 一句话总结

系统性研究了在不依赖外部模型（VLM/CLIP）的前提下，对文本到图像扩散模型的初始噪声优化算法施加 Best-of-N 推理时缩放的效果，发现性能会迅速达到平台期（plateau），少量优化步数即可逼近该设置下的最大性能，且不同底层扩散模型上的最优算法不同。

## 研究背景与动机

### 推理时缩放：从 LLM 到扩散模型
推理时缩放（Inference-Time Scaling）在大语言模型中展现了巨大潜力——在不增加模型参数的前提下，通过投入更多推理计算来提升生成质量（OpenAI 2024; Guo et al., 2025）。这一思路自然被引入扩散模型领域：Ma et al. (2025) 的开创性工作表明，通过搜索更好的初始噪声可以显著提升 T2I 任务性能。

### 现有方法的核心瓶颈
然而，已有的推理时缩放方法**均依赖外部模型**（如 VLM 或 CLIP）来评估生成图像的质量。这带来了严峻的实际限制：

- **显存需求大**：同时加载扩散模型 + 评估模型需要高端 GPU，消费级显卡（如 8-12GB VRAM）难以支撑
- **部署成本高**：在个人桌面环境而非大型实验室中几乎不可行
- **推理时延增加**：每次评估都需额外的前向传播

### 无外部模型的替代路径
另一条路径是仅利用扩散模型自身的信息（如注意力图）来优化初始噪声，代表方法包括 CONFORM、InitNO 和 Self-Cross guidance。这些方法无需额外模型，但一个关键问题尚未被回答：

> **仅依靠扩散模型自身选择好的初始噪声时，投入更多计算（Best-of-N 缩放）是否能持续提升性能？**

本文正是对这一问题的系统性实证研究。

## 方法详解

### 整体框架：Best-of-N 初始噪声选择

核心思路非常直接：
1. 分配总共 $N$ 次损失计算预算
2. 采样多个候选初始噪声，用各算法的损失函数评估每个噪声的质量
3. 选择损失值最低的噪声作为扩散模型的初始噪声
4. 用该噪声执行标准去噪过程，生成最终图像

若损失函数能准确反映生成图像质量，则随着 $N$ 增大，性能应单调递增。本文通过大规模实验验证这一假设。

### 三种无外部模型的噪声优化算法

**1. CONFORM (Meral et al., 2024)**
- 利用 T2I 扩散模型的**交叉注意力图（cross-attention maps）**进行对比学习
- 损失函数为 InfoNCE，将文本 prompt 中的物体和属性分为正负样本对
- 每个噪声独立评估，$N$ 次计算对应 $N$ 个候选噪声
- 优势：直接利用 prompt 语义结构，无需迭代优化

**2. InitNO (Guo et al., 2024)**
- 核心思想：并非所有从标准正态分布采样的噪声都与给定 prompt 匹配，存在"有效"和"无效"噪声之分
- 损失函数包含两部分：
    - $1 - \text{minmax\_cross}$：minmax_cross 是每个目标物体的最大交叉注意力权重中的最小值，衡量所有物体是否都被充分关注
    - **自注意力图重叠度**：测量各物体对应的最大交叉注意力 patch 的自注意力图之间的重叠程度（重叠越大，物体混淆越严重）
- 关键特点：对单个噪声进行**迭代优化**（最多 10 步），因此 $N$ 次计算仅产生 $N/10$ 个候选
- 包含提前终止机制：若损失低于预定阈值，立即停止优化

**3. Self-Cross Guidance (Qiu et al., 2024)**
- 损失与 InitNO 类似，但关键区别在于**不只关注最大交叉注意力 patch**
- 使用**所有 patch 的自注意力图**，以各自的交叉注意力权重为加权，计算物体间重叠
- 对物体空间分离的衡量更全局化
- 作者建议先用 InitNO 再用 Self-Cross guidance（两阶段优化）
- 每个噪声独立评估，$N$ 次计算对应 $N$ 个候选

### 候选噪声数量的差异

| 算法 | 每个候选的计算开销 | $N$ 次计算的有效候选数 |
|------|------|------|
| CONFORM | 1 次损失计算 | $N$ |
| InitNO | 10 次损失计算（迭代优化） | $N/10$ |
| Self-Cross Guidance | 1 次损失计算 | $N$ |

这一差异使得公平比较需要统一以"损失计算次数"为计算预算单位。

## 实验关键数据

### 实验设置
- **扩散模型**：Stable Diffusion 1.5 (SD1.5)、SD2 等多种骨干网络
- **数据集**：4 个测试集，涵盖不同组合复杂度
    - `animal_animal`：两个动物的组合
    - `animal_object`：动物与物体的组合
    - `object_object`：两个物体的组合
    - `similar_subjects`：相似主体（最具挑战性，易混淆）
- **评估**：生成图像与 prompt 的对齐度（文本-图像一致性）
- **缩放范围**：$N$ 从小到大系统性变化，观察性能趋势

### Table 1: SD1.5 上各算法的性能随 N 变化趋势（性能平台期现象）

| 算法 | N=10 | N=50 | N=100 | N=500 | 性能趋势 |
|------|------|------|------|------|------|
| CONFORM | 基线附近 | 接近最优 | 平台期 | 无显著提升 | 快速饱和 |
| InitNO | 显著提升 | 接近最优 | 平台期 | 无显著提升 | 中速饱和 |
| Self-Cross | 基线附近 | 接近最优 | 平台期 | 无显著提升 | 快速饱和 |

核心发现：**所有算法在 $N$ 较小时即达到性能平台期**，继续增加计算预算不会带来有意义的性能提升，这与默认超参数设置和直觉预期形成鲜明对比。

### Table 2: 不同骨干模型上的最优算法对比

| 骨干模型 | 最优算法 | 说明 |
|------|------|------|
| SD1.5 (UNet) | 因数据集而异 | CONFORM 和 InitNO 各有优势场景 |
| SD2 (UNet) | 与 SD1.5 不同 | 最优算法发生变化 |
| 其他骨干 | 待探索 | SOTA 算法随模型变化 |

**关键发现**：无外部模型的初始噪声优化的 SOTA 算法**随底层扩散模型变化而变化**，表明当前没有一种算法在所有模型上都是最优的，这为未来研究指出了方向。

## 亮点与洞察

- **实证贡献清晰有力**：首次系统性证明了无外部模型设定下推理时缩放存在**性能天花板**（plateau），打破了"更多计算 = 更好性能"的直觉假设
- **实用价值高**：明确告诉从业者在消费级 GPU 上，少量优化步数即可达到最优，**无需浪费计算资源**。这为资源受限场景下的 T2I 部署提供了直接指导
- **揭示损失函数的局限性**：性能平台期的出现暗示这些算法的损失函数**并不能完美反映生成图像质量**——如果损失函数是完美代理指标，性能应随 $N$ 单调递增
- **模型依赖性发现**：不同扩散模型上最优算法不同，说明注意力图的信息量和质量因模型而异，需要针对性地设计噪声优化策略
- **公平比较框架**：统一以"损失计算次数"为预算单位，为不同复杂度的算法提供了公平的比较基准（尤其是 InitNO 的迭代优化 vs. CONFORM 的单次评估）

## 局限与展望

- **仅覆盖三种算法**：当前仅评估 CONFORM、InitNO 和 Self-Cross guidance，未来可能出现的新型无外部模型算法或许能突破平台期
- **损失函数设计空间未探索**：平台期的根本原因可能在于现有损失函数基于注意力图的信息表达能力不足，寻找更强的自监督信号是关键方向
- **骨干模型范围有限**：主要在 SD1/SD2 系列上实验，未覆盖更新的架构如 SDXL、SD3、Flux 等基于 DiT 的模型
- **评估指标单一性**：性能评估主要依赖文本-图像对齐度，未充分考虑图像的美学质量、多样性等维度
- **与外部模型方法的差距**：性能平台期表明无外部模型方法存在固有上限，如何在不增加显存的前提下引入更强的质量信号（如轻量级评估器）是值得探索的方向
- **计算开销分析不够细**：虽然以损失计算次数统一预算，但未详细报告各算法的墙钟时间（wall-clock time）和实际显存占用

## 相关工作

### 推理时缩放
- **LLM 领域**：OpenAI (2024)、Guo et al. (2025)、Zhang et al. (2025) 等证明了推理时增加计算可替代模型扩大
- **扩散模型领域**：Ma et al. (2025) 首次将推理时缩放引入 T2I，通过 VLM 作为验证器搜索最优初始噪声；Li et al. (2025)、Zhuo et al. (2025) 进一步拓展
- **本文定位**：聚焦于**无外部模型**的约束场景，揭示该设定下推理时缩放的根本性局限

### 初始噪声优化
- **CONFORM** (Meral et al., 2024)：基于交叉注意力图的对比学习，用 InfoNCE 约束 prompt 与注意力的一致性
- **InitNO** (Guo et al., 2024)：提出"有效/无效噪声"概念，通过迭代优化确保初始噪声与 prompt 语义对齐
- **Self-Cross Guidance** (Qiu et al., 2024)：结合全局自注意力与交叉注意力信息，比 InitNO 更全面地衡量物体分离度

### T2I 扩散模型基础
- **Stable Diffusion** (Rombach et al., 2022)：潜在扩散模型，在自动编码器的潜在空间中进行 DDPM 去噪
- **DDPM** (Ho et al., 2020)：去噪扩散概率模型，扩散模型的基础框架
- **交叉注意力机制**：文本编码器输出作为 key，UNet 中间特征作为 query，形成跨模态对齐

## 评分

- 新颖性: ⭐⭐⭐ — 研究问题重要且及时，但方法本身（Best-of-N）相对直接，核心贡献是实证发现而非算法创新
- 实验充分度: ⭐⭐⭐⭐ — 多算法 × 多数据集 × 多骨干的系统性实验，较为全面；但骨干模型和评估指标可以更丰富
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，实验设计合理，结论表述明确
- 价值: ⭐⭐⭐⭐ — 对 T2I 推理时缩放的实际部署有直接指导意义，为该方向的研究者节省了大量试错成本

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Rethinking Prompt Design for Inference-time Scaling in Text-to-Visual Generation](../../CVPR2026/image_generation/rethinking_prompt_design_for_inference-time_scaling_in_text-to-visual_generation.md)
- [\[NeurIPS 2025\] Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing](../../NeurIPS2025/image_generation/inference-time_scaling_for_flow_models_via_stochastic_generation_and_rollover_bu.md)
- [\[NeurIPS 2025\] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](../../NeurIPS2025/image_generation/progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)
- [\[ICML 2025\] One Image is Worth a Thousand Words: A Usability Preservable Text-Image Collaborative Erasing Framework](one_image_is_worth_a_thousand_words_a_usability_preservable_text-image_collabora.md)
- [\[CVPR 2025\] Scaling Down Text Encoders of Text-to-Image Diffusion Models](../../CVPR2025/image_generation/scaling_down_text_encoders_of_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
