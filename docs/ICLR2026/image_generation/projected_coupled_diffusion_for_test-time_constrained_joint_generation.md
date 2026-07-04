---
title: >-
  [论文解读] Projected Coupled Diffusion for Test-Time Constrained Joint Generation
description: >-
  [ICLR2026][图像生成][测试时约束生成] **领域现状**：扩散模型已经成为图像、视频、语言、图、机器人轨迹等生成任务里的通用建模工具。很多实用系统并不只需要“无条件生成一个样本”，而是在推理阶段加入额外目标，例如 classifier guidance、inpainting、reward guidance 或 projected diffusion，让已有模型在不重训的情况下朝某个条件或约束采样。
tags:
  - "ICLR2026"
  - "图像生成"
  - "测试时约束生成"
  - "联合生成"
  - "投影扩散"
  - "耦合扩散"
  - "多模型协同"
---

# Projected Coupled Diffusion for Test-Time Constrained Joint Generation

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=1FEm5JLpvg](https://openreview.net/forum?id=1FEm5JLpvg)
**代码**: https://github.com/EdmundLuan/pcd  
**领域**: 扩散模型 / 图像生成  
**关键词**: 测试时约束生成、联合生成、投影扩散、耦合扩散、多模型协同
**代码**: 待确认  
**领域**: generative models  
这篇论文提出 Projected Coupled Diffusion（PCD），在不重新训练扩散模型的前提下，把多个预训练边际扩散模型通过耦合代价联合采样，并在每个扩散步做投影来严格满足测试时硬约束，在机器人轨迹、人脸配对生成和物体操作任务上同时改善相关性与约束满足。

## 一句话总结
**领域现状**：扩散模型已经成为图像、视频、语言、图、机器人轨迹等生成任务里的通用建模工具。很多实用系统并不只需要“无条件生成一个样本”，而是在推理阶段加入额外目标，例如 classifier guidance、inpainting、reward guidance 或 projected diffusion，让已有模型在不重训的情况下朝某个条件或约束采样。

## 研究背景与动机
### 整体框架
PCD 的输入是一组预训练扩散模型或 score model，例如两个图像 latent diffusion 模型、多个机器人轨迹扩散模型，或者两条操作轨迹的 diffusion policy。每个模型仍然负责自己的边际分布；PCD 不改模型参数，而是在反向扩散的每一步加入耦合代价梯度，再把更新后的样本投影到测试时给出的可行集合中，最后输出一组满足约束且相互协调的联合样本。

## 方法详解
### 主实验
论文覆盖三个应用场景：多机器人导航、PushT 物体操作、人脸配对生成。下面摘取主文中最能说明 PCD 同时处理“相关性 + 硬约束”的结果。

## 实验关键数据
PCD 最漂亮的地方是把“联合性”和“可行性”拆成两个可插拔算子。耦合代价不必知道所有约束，投影也不必懂生成语义，二者在每个扩散步相遇后就形成一个简洁的测试时控制循环。

## 亮点与洞察
PCD 依赖耦合代价的梯度或次梯度。若任务约束是离散逻辑、不可微仿真指标或黑盒安全规则，就需要可微近似、采样式估计或其他 surrogate，效果可能取决于近似质量。

## 局限性 / 可改进方向
**vs Classifier Guidance**: classifier guidance 用条件 likelihood 的梯度推动单个扩散模型采样到目标属性；PCD 把它看作 $Y$ 被固定成条件、$X$ 无投影的特殊情形。PCD 的优势是可以让多个变量同时移动，并额外支持硬约束投影。

## 相关工作与启发
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Test-Time Iterative Error Correction for Efficient Diffusion Models](test-time_iterative_error_correction_for_efficient_diffusion_models.md)
- [\[ICLR 2026\] MILR: Improving Multimodal Image Generation via Test-Time Latent Reasoning](milr_improving_multimodal_image_generation_via_test-time_latent_reasoning.md)
- [\[CVPR 2026\] Progress by Pieces: Test-Time Scaling for Autoregressive Image Generation](../../CVPR2026/image_generation/progress_by_pieces_test-time_scaling_for_autoregressive_image_generation.md)
- [\[ICLR 2026\] VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model](vfscale_intrinsic_reasoning_through_verifier-free_test-time_scalable_diffusion_m.md)
- [\[ICML 2026\] Linearizing Vision Transformer with Test-Time Training](../../ICML2026/image_generation/linearizing_vision_transformer_with_test-time_training.md)

</div>

<!-- RELATED:END -->
