---
title: >-
  [论文解读] Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking
description: >-
  [CVPR 2025][图像生成][扩散模型编辑] 理论和实验统一分析了扩散模型编辑会"无意间"破坏鲁棒不可见水印的现象——正向加噪使水印 SNR 指数衰减，反向去噪的流形收缩效应将水印信号当作"非自然残差"消除，即使 VINE 等最先进水印在强编辑（$t^*=0.8$）下也降至接近随机猜测（~60% bit accuracy）。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "扩散模型编辑"
  - "鲁棒水印"
  - "水印移除"
  - "信息论分析"
  - "内容溯源"
---

# Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking

**会议**: CVPR 2025  
**arXiv**: [2603.12949](https://arxiv.org/abs/2603.12949)  
**代码**: 无  
**领域**: 图像生成 / 数字水印  
**关键词**: 扩散模型编辑, 鲁棒水印, 水印移除, 信息论分析, 内容溯源

## 一句话总结

理论和实验统一分析了扩散模型编辑会"无意间"破坏鲁棒不可见水印的现象——正向加噪使水印 SNR 指数衰减，反向去噪的流形收缩效应将水印信号当作"非自然残差"消除，即使 VINE 等最先进水印在强编辑（$t^*=0.8$）下也降至接近随机猜测（~60% bit accuracy）。

## 研究背景与动机

**领域现状**：深度学习水印系统（StegaStamp、TrustMark、VINE）通过端到端训练+可微噪声层，在 JPEG/缩放/裁剪等传统后处理下保持高鲁棒性（>95% bit accuracy）。

**现有痛点**：扩散模型编辑（InstructPix2Pix、DragDiffusion、TF-ICON 等）引入了全新的变换类别——先注入大噪声再通过生成先验重建。这与传统后处理本质不同，水印系统未针对此训练。

**核心矛盾**：水印本质是"低幅度结构化扰动"，而扩散去噪器被训练来移除一切"非自然残差"——水印恰好就是这样的残差。编辑者不是在刻意攻击水印，但编辑过程本身就会破坏水印。

**本文目标** 在什么条件下扩散编辑会导致水印不可恢复？有什么理论原则解释这种崩溃？

**切入角度**：将扩散编辑建模为 Markov kernel（正向加噪+条件去噪），推导水印 SNR 衰减和互信息衰减的界，给出 Fano 型不可恢复条件。

**核心 idea**：对传统后处理鲁棒 ≠ 对生成式变换鲁棒——扩散编辑的流形收缩效应系统性地消除了水印信号。

## 方法详解

### 整体框架

三部分：(1) 将水印图像的扩散编辑建模为 Markov kernel $K_\mathcal{T}(\tilde{x} | x_w, y)$，(2) 推导 SNR 衰减和互信息衰减的理论界，(3) 设计标准化评估协议 DEW-ST 横跨 7 种扩散编辑器 × 3 种水印系统。

### 关键设计

1. **水印信号模型**：

    - $x_w = x + \gamma \mathbf{s}(\mathbf{m}, \mathbf{k}, x)$：水印是低幅度加性扰动
    - 正向加噪后：$x_t = \sqrt{\bar\alpha_t} x_w + \sqrt{1-\bar\alpha_t} \epsilon$
    - 水印分量 SNR = $\gamma^2 \bar\alpha_t \|\mathbf{s}\|^2 / (1-\bar\alpha_t)$ → 随 $t$ 指数衰减

2. **互信息衰减界**：

    - 推导 $I(\mathbf{m}; \tilde{x})$ 沿扩散轨迹的衰减，连接到 Fano 不等式 → 给出 bit error rate 的下界
    - 关键结论：当编辑强度 $t^*$ 超过临界值时，水印不可恢复变成信息论层面的必然

3. **DEW-ST 评估协议**：

    - 7 种编辑器：InstructPix2Pix、UltraEdit、DragDiffusion、DragFlow、InstantDrag、TF-ICON、SHINE
    - 3 种水印：StegaStamp (物理鲁棒)、TrustMark (多分辨率)、VINE (扩散感知训练)
    - 多种编辑强度 $t^* \in \{0.2, 0.4, 0.6, 0.8\}$

### 频域分析

水印信号在不同频段的保留率 $\rho_\Omega$ 分析：扩散编辑对高频和中频带水印信号的抑制最强，低频信号保留相对较好。

## 实验关键数据

### 主实验

| 变换 | 强度 | StegaStamp | TrustMark | VINE | PSNR(dB) |
|------|------|-----------|-----------|------|---------|
| 无处理 | — | 99.4% | 99.7% | 99.8% | 41.2 |
| JPEG Q50 | — | 96.1% | 98.2% | 98.9% | 33.5 |
| InstructPix2Pix | $t^*$=0.4 | 71.5% | 76.1% | 85.4% | 29.8 |
| InstructPix2Pix | $t^*$=0.8 | **53.2%** | **55.0%** | **60.7%** | 25.1 |
| DragDiffusion | medium | 63.4% | 67.9% | 78.6% | 28.7 |
| TF-ICON 合成 | — | 58.9% | 63.2% | 74.8% | 28.1 |

（注：作者声明实验数据为"hypothetical but realistic"，反映文献趋势）

### 按编辑类型

| 编辑类型 | StegaStamp | TrustMark | VINE |
|---------|-----------|-----------|------|
| 局部编辑 | ~75% | ~80% | ~88% |
| 全局编辑 | ~55% | ~58% | ~63% |

### 关键发现

- **扩散编辑 vs 传统后处理的巨大鸿沟**：JPEG Q50 下 StegaStamp 保持 96.1%，但 InstructPix2Pix $t^*=0.4$ 就降到 71.5%。$t^*=0.8$ 时接近随机猜测
- **VINE 虽最鲁棒但仍不够**：VINE 采用扩散感知训练，在轻编辑下保持 85%+，但强编辑下仍降至 60%
- **合成/插入操作特别致命**：TF-ICON、SHINE 即使保持全局逼真，水印也崩溃（55-74%）
- **局部编辑也能破坏全局水印**：因为扩散 latent 空间的去噪耦合会影响编辑区域之外的像素
- **频域分析**：高频水印信号被强烈抑制，低频相对保留——但大多数水印编码在中高频

## 亮点与洞察

- **"无意移除"的视角非常重要**：这不是攻击——用户只是在正常编辑图片。但编辑过程本身就会系统性地破坏水印。这对内容溯源基础设施的可靠性提出了根本质疑。
- **信息论分析直击要害**：不是说"这个水印方法结果不好"，而是推导出"在这个编辑强度下，任何水印方法都不可能可靠恢复"。这是一个impossibility result。
- **对水印设计的建设性建议**：(a) 扩散原生指纹（如 Tree-Ring，嵌入初始噪声）比后处理水印更鲁棒；(b) 优化语义不变性而非像素鲁棒性。

## 局限与展望

- **实验数据是"hypothetical but realistic"**：作者坦承表格中的数据不是真实实验而是基于文献趋势的模拟值。需要真实实验验证
- **未测试 Tree-Ring 等扩散原生水印**：理论预测它们更鲁棒，应实际对比
- **未考虑水印系统的对抗性微调**：如果水印系统在训练时引入扩散编辑作为噪声层，鲁棒性可能改善
- **隐私张力**：强水印 vs 编辑自由 vs 隐私之间存在根本矛盾，本文未深入探讨

## 相关工作与启发

- **vs VINE (W-Bench)**：VINE 已发现扩散编辑对水印的威胁并做了扩散感知训练，本文进一步给出理论分析和跨编辑器的系统评估。
- **vs ForensicZip**：ForensicZip 处理的是取证 token 压缩，而本文处理的是水印 vs 编辑。但两者共享一个 insight：扩散模型的去噪过程会移除"非自然"（off-manifold）信号。
- **启发**：未来水印设计应该考虑"生成流形"而非"像素空间"——水印信号需要与数据流形兼容才能在扩散编辑后存活。

## 评分

- 新颖性: ⭐⭐⭐⭐ 系统性分析很全面，信息论bound有理论价值；但"扩散破坏水印"的观察不是全新的
- 实验充分度: ⭐⭐⭐ 覆盖了7种编辑器和3种水印，但实验数据是模拟的而非真实实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导+实验+设计建议的三段式非常清晰
- 价值: ⭐⭐⭐⭐ 对内容溯源/水印设计领域有重要警示和指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Robust-Wide: Robust Watermarking against Instruction-driven Image Editing](../../ECCV2024/image_generation/robust-wide_robust_watermarking_against_instruction-driven_image_editing.md)
- [\[CVPR 2025\] EmoEdit: Evoking Emotions through Image Manipulation](emoedit_evoking_emotions_through_image_manipulation.md)
- [\[NeurIPS 2025\] Gradient Variance Reveals Failure Modes in Flow-Based Generative Models](../../NeurIPS2025/image_generation/gradient_variance_reveals_failure_modes_in_flow-based_generative_models.md)
- [\[NeurIPS 2025\] FineGRAIN: Evaluating Failure Modes of Text-to-Image Models with Vision Language Model Judges](../../NeurIPS2025/image_generation/finegrain_evaluating_failure_modes_of_text-to-image_models_with_vision_language_.md)
- [\[CVPR 2025\] Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation](trust_your_critic_robust_reward_modeling_and_reinforcement_learning_for_faithful.md)

</div>

<!-- RELATED:END -->
