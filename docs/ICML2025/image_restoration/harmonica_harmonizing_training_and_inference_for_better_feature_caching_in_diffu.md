---
title: >-
  [论文解读] HarmoniCa: Harmonizing Training and Inference for Better Feature Caching in Diffusion Transformer Acceleration
description: >-
  [ICML 2025][图像恢复][Transformer] 提出 HarmoniCa 框架，通过 Step-Wise Denoising Training (SDT) 和 Image Error Proxy-Guided Objective (IEPO) 两大设计解决现有学习型特征缓存方法中训练与推理不对齐的问题，在 PixArt-α 等 8 种模型上实现超 40% 延迟降低（2.07× 理论加速）且不损失生成质量。
tags:
  - "ICML 2025"
  - "图像恢复"
  - "Transformer"
  - "特征缓存"
  - "推理加速"
  - "训练-推理对齐"
  - "去噪训练"
---

# HarmoniCa: Harmonizing Training and Inference for Better Feature Caching in Diffusion Transformer Acceleration

**会议**: ICML 2025  
**arXiv**: [2410.01723](https://arxiv.org/abs/2410.01723)  
**代码**: [https://github.com/YSGuoUST/HarmoniCa](https://github.com/YSGuoUST/HarmoniCa)  
**领域**: 图像复原  
**关键词**: 扩散 Transformer, 特征缓存, 推理加速, 训练-推理对齐, 去噪训练

## 一句话总结
提出 HarmoniCa 框架，通过 Step-Wise Denoising Training (SDT) 和 Image Error Proxy-Guided Objective (IEPO) 两大设计解决现有学习型特征缓存方法中训练与推理不对齐的问题，在 PixArt-α 等 8 种模型上实现超 40% 延迟降低（2.07× 理论加速）且不损失生成质量。

## 研究背景与动机

**领域现状**：Diffusion Transformer（DiT）在图像生成任务上表现卓越，但推理成本极高——每次生成需要多步去噪，每步都要完整前向传播整个 Transformer，导致实际部署面临巨大挑战。

**现有方法**：特征缓存（Feature Caching）是一种有效的加速策略，核心思想是存储并复用相邻去噪步骤间的冗余计算特征，从而跳过部分层的计算。现有学习型缓存方法（如 Learning-to-Cache）通过训练一个轻量路由器来自适应决定哪些层可以缓存。

**现有痛点**：现有学习型缓存存在两个关键不对齐问题：
   - **(a) 忽略前序步骤的影响**：训练时每步独立优化，但推理时前一步的缓存决策会改变中间特征，影响当前步的最优缓存策略。这种训练/推理间的"步骤断裂"导致次优的缓存路由。
   - **(b) 目标不对齐**：训练优化的是预测噪声的对齐度（即 $\epsilon$-prediction loss），但最终目标是生成高质量图像。noise 层面的小差异可能在图像层面被放大（或反之），导致缓存策略无法真正服务于图像质量。

**本文目标**：如何让缓存路由器的训练过程与实际推理过程一致？如何让优化目标直接关联最终图像质量？

**切入角度**：核心观察——训练时应模拟推理的逐步去噪过程（而非独立训练每步），且应用图像域的误差信号指导缓存决策。

**核心 idea**：通过"协调"（Harmonize）训练和推理来构建更好的特征缓存——SDT 让训练看到推理时的真实中间状态，IEPO 让优化目标从 noise 空间转向 image 空间。

## 方法详解

### 整体框架
HarmoniCa 是一个学习型特征缓存框架，适用于各种 DiT 架构。整体流程：
- **输入**：预训练的 DiT 模型（冻结参数）+ 噪声图像
- **可学习部分**：轻量级缓存路由器（决定每层的缓存/计算策略）
- **训练阶段**：通过 SDT 逐步去噪训练路由器，使用 IEPO 作为优化目标
- **推理阶段**：路由器为每个去噪步骤的每一层做出缓存决策——缓存命中则复用上一步特征，否则正常计算
- **输出**：加速后的高质量生成图像

其核心思想是让训练过程尽可能接近推理过程，使得路由器在训练中就"见过"推理时会遇到的累积误差场景。

### 关键设计

1. **Step-Wise Denoising Training (SDT)**:

    - 功能：在训练缓存路由器时模拟完整的逐步去噪过程，而非独立训练每个时间步
    - 核心思路：传统方法在训练时随机采样一个时间步 $t$，独立计算该步的缓存损失。SDT 则让模型从 $t_T$ 逐步去噪到 $t_0$，每一步都使用上一步的实际输出（包含缓存引入的近似误差）作为输入。这样路由器能感知前序步骤的缓存效应。
    - 设计动机：在推理中，第 $t$ 步的输入是第 $t+1$ 步的输出——如果第 $t+1$ 步因缓存使用了近似特征，那么第 $t$ 步的输入已经偏离了"无缓存"时的理想轨迹。SDT 让路由器在训练时就学会在这种偏移下做出合理决策，弥合了训练-推理间的步骤连续性差距。
    - 与之前方法的区别：Learning-to-Cache 等方法每步独立训练，相当于假设每步输入都是干净的无误差信号，而 SDT 让训练见到累积误差的真实场景。

2. **Image Error Proxy-Guided Objective (IEPO)**:

    - 功能：用一个高效的代理函数近似最终图像误差，替代传统的 noise prediction loss 来指导缓存决策
    - 核心思路：直接计算图像误差代价太高（需要完成整个去噪链），IEPO 设计了一个轻量级代理来近似图像域误差。具体来说，利用去噪过程中 noise prediction 与最终图像之间的已知数学关系（DDPM/DDIM 的解析公式），将 $\epsilon$ 空间的误差通过缩放因子映射到图像空间进行评估。这让每步的优化目标 $\mathcal{L}_{\text{IEPO}}$ 直接反映对最终图像质量的影响。
    - 设计动机：noise 空间和 image 空间的误差分布不同——某些层在 noise 空间误差小但在 image 空间影响大（反之亦然）。IEPO 确保路由器优先缓存对图像质量影响最小的层，而非仅在 noise 空间误差最小的层。
    - 额外优势：IEPO 是 image-free 的（不需要真实图像），仅依赖去噪过程本身的中间变量，因此相比需要真实图像监督的方法，训练时间减少约 25%。

3. **缓存路由器设计**:

    - 功能：为 DiT 每一层在每个去噪步决定是"计算"还是"从缓存读取"
    - 核心思路：路由器是一个轻量网络，输入当前步的特征统计量，输出每层的二值决策（通过 Gumbel-Softmax 实现可微训练）。在给定算力预算（cache ratio）下，路由器学习最优的缓存分配策略。
    - 设计动机：不同层在不同去噪阶段的冗余程度不同——某些层在早期步骤变化剧烈（不应缓存），同一层在后期步骤几乎不变（可以安全缓存）。路由器通过学习自适应地发现这种模式。

### 训练策略
- 冻结预训练 DiT 参数，仅训练轻量路由器
- 结合 SDT 逐步训练 + IEPO 目标函数
- 支持可调的缓存预算（cache ratio），在速度和质量间灵活权衡
- Image-free 训练：不需要准备真实图像数据集，仅用随机噪声即可训练
- 训练效率：相比 Learning-to-Cache 减少约 25% 训练时间

## 实验关键数据

### 主实验

论文在 **8 种模型**、**4 种采样器**、分辨率从 $256 \times 256$ 到 $2K$ 上进行了广泛验证。

| 模型 | 方法 | 加速比 | 图像质量 | 说明 |
|------|------|--------|----------|------|
| PixArt-α | 无缓存（基线） | 1.0× | 基线 | 完整计算 |
| PixArt-α | Learning-to-Cache | ~1.5× | 轻微下降 | 步骤不连续 |
| PixArt-α | **HarmoniCa** | **2.07×** | **持平/提升** | >40% 延迟降低 |
| DiT (256×256) | 无缓存 | 1.0× | 基线 | — |
| DiT (256×256) | **HarmoniCa** | ~1.8× | 无损 | 低分辨率验证 |
| 高分辨率模型 (2K) | 无缓存 | 1.0× | 基线 | — |
| 高分辨率模型 (2K) | **HarmoniCa** | >1.5× | 无损 | 高分辨率泛化 |

### 消融实验

| 配置 | 图像质量变化 | 加速效率 | 说明 |
|------|-------------|----------|------|
| Full (SDT + IEPO) | 最优 | 2.07× | 完整 HarmoniCa |
| w/o SDT（独立步训练） | 明显下降 | ~1.8× | 步骤连续性缺失致路由次优 |
| w/o IEPO（用 noise loss） | 中等下降 | ~1.9× | 目标不对齐，部分层缓存不当 |
| w/o SDT & IEPO | 显著下降 | ~1.5× | 退化为 Learning-to-Cache |
| 不同 cache ratio (20%-60%) | 平滑退化 | 1.3×-2.5× | 灵活的速度-质量权衡 |

### 关键发现
- **SDT 贡献大于 IEPO**：去掉 SDT 的质量下降比去掉 IEPO 更严重，说明训练-推理的步骤对齐是核心瓶颈
- **Image-free 训练不仅不损质量，还更快**：IEPO 的代理机制避免了真实图像的前处理开销，训练时间比 Learning-to-Cache 少 25%
- **跨模型/采样器/分辨率的鲁棒性强**：在 8 种模型和 4 种采样器上一致有效，说明方法的通用性好
- **高缓存比下优势更明显**：当 cache ratio 较高（>50%）时，HarmoniCa 相对基线的优势更大——因为高缓存比下路由决策更关键，SDT+IEPO 的对齐效果被放大

## 亮点与洞察
- **训练-推理对齐**的思想非常通用：不仅适用于特征缓存，任何存在训练与推理过程不匹配的场景（如自回归生成中的 exposure bias）都可借鉴 SDT 的逐步训练思路
- **IEPO 的代理设计**极其巧妙：避免了直接计算图像误差的高昂代价，利用扩散模型本身的数学结构（$x_0$ 与 $\epsilon$ 的解析关系）构建了免费的图像误差近似
- **Image-free 训练**是一个实用亮点：降低了数据准备门槛，且减少训练时间，对实际部署非常友好
- **2.07× 加速且不降质量**的结果在 DiT 加速文献中非常突出——大多数方法在此加速比下会有明显质量损失

## 局限与展望
- **缓存开销本身未讨论**：存储中间层特征需要额外显存，在 batch 较大或模型层数多时可能成为瓶颈
- **仅验证图像生成**：虽然框架理论上适用于视频 DiT（如 Sora），但论文未验证视频生成场景
- **路由器的泛化性**：路由器针对特定模型训练，更换 DiT 架构需重新训练，缺乏跨模型迁移性
- **与其他加速方法的组合**：未探索与量化、蒸馏、token 裁剪等其他加速技术的组合效果
- **IEPO 代理的精度上界**：代理函数终究是近似，在极端 cache ratio 下（如 >70%）代理误差可能累积

## 相关工作与启发
- **vs Learning-to-Cache**：Learning-to-Cache 独立优化每步且用 noise loss，HarmoniCa 同时解决步骤连续性（SDT）和目标对齐（IEPO）两个问题，全面超越
- **vs AsymRnR**：AsymRnR 是免训练的 token 裁剪方法，与 HarmoniCa 的学习型缓存互补——前者减少每层计算量，后者跳过整层计算，两者有组合潜力
- **vs 知识蒸馏（如 Progressive Distillation）**：蒸馏减少去噪步数，缓存减少每步计算量，正交的加速方向可叠加
- **与 Consistency Model 的关系**：Consistency Model 通过一步生成绕过多步去噪，而 HarmoniCa 保留多步框架但加速每步——在需要高质量/可控生成的场景下更合适

## 评分
- 新颖性: ⭐⭐⭐⭐ SDT 和 IEPO 两个设计切中现有方法的核心痛点，对齐训练-推理的思路有普遍启发性
- 实验充分度: ⭐⭐⭐⭐⭐ 8 模型 × 4 采样器 × 多分辨率的全面验证，消融充分
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，SDT 和 IEPO 的动机推导逻辑顺畅
- 价值: ⭐⭐⭐⭐⭐ 2× 加速无损质量 + 训练高效，对 DiT 实际部署有直接推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](../../NeurIPS2025/image_restoration/encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [\[ICML 2025\] TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation](timedart_a_diffusion_autoregressive_transformer_for_self-supervised_time_series_.md)
- [\[ECCV 2024\] Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators](../../ECCV2024/image_restoration/efficient_diffusion_transformer_with_step-wise_dynamic_attention_mediators.md)
- [\[ICML 2026\] DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention](../../ICML2026/image_restoration/dyllm_efficient_diffusion_llm_inference_via_saliency-based_token_selection_and_p.md)
- [\[CVPR 2025\] Progressive Focused Transformer for Single Image Super-Resolution](../../CVPR2025/image_restoration/progressive_focused_transformer_for_single_image_super-resolution.md)

</div>

<!-- RELATED:END -->
