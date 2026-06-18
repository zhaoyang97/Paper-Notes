---
title: >-
  [论文解读] Growing a Twig to Accelerate Large Vision-Language Models
description: >-
  [ICCV 2025][多模态VLM][VLM加速] 提出 TwigVLM，通过在 VLM 早期层上"生长"一个轻量级 twig 模块，同时实现 twig 引导的视觉 token 剪枝（TTP，prefilling 加速）和自推测解码（SSD，decoding 加速），在 LLaVA-1.5-7B 上剪枝 88.9% 视觉 token 后保留 96% 精度，长回答生成速度提升 154%，在精度和速度上均大幅超越现有方法。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "VLM加速"
  - "视觉token剪枝"
  - "自推测解码"
  - "轻量级模块"
  - "推理效率"
---

# Growing a Twig to Accelerate Large Vision-Language Models

**会议**: ICCV 2025  
**arXiv**: [2503.14075](https://arxiv.org/abs/2503.14075)  
**代码**: [github.com/MILVLG/twigvlm](https://github.com/MILVLG/twigvlm)  
**领域**: 多模态VLM  
**关键词**: VLM加速, 视觉token剪枝, 自推测解码, 轻量级模块, 推理效率

## 一句话总结

提出 TwigVLM，通过在 VLM 早期层上"生长"一个轻量级 twig 模块，同时实现 twig 引导的视觉 token 剪枝（TTP，prefilling 加速）和自推测解码（SSD，decoding 加速），在 LLaVA-1.5-7B 上剪枝 88.9% 视觉 token 后保留 96% 精度，长回答生成速度提升 154%，在精度和速度上均大幅超越现有方法。

## 研究背景与动机

### 问题定义

大型视觉语言模型（VLM）在开放世界多模态理解中表现优异，但高计算开销严重阻碍其实际部署。VLM 的推理过程分为 prefilling（处理输入 token）和 decoding（逐个生成回答 token）两个阶段，需要同时加速两个阶段以实现实际可用的推理效率。

### 已有方法的不足

**早期层注意力信号质量差**：FastV 等方法利用 VLM 第 2 层的注意力图指导视觉 token 剪枝，但实验发现早期层的注意力对任务不敏感——不同 prompt 选出几乎相同的 token，导致剪枝后精度显著下降

**仅加速 prefilling 阶段**：现有 token 剪枝方法（FastV、SparseVLM、VisionZip 等）主要关注 prefilling 加速，而 decoding 阶段是推理过程中耗时最长的部分。当生成 ≥32 个 token 时，decoding 时间已远超 prefilling 时间

**KV-cache 限制了剪枝对 decoding 的加速**：由于 KV-cache 机制，视觉 token 剪枝对 self-attention 的 decoding 加速效果有限，且完全不加速占主要计算量的 FFN 层

### 核心动机

作者通过两个精心设计的先导实验揭示了关键洞察：

**先导实验 1（注意力质量）**：用不同深度 $D$ 层的注意力图指导第 $K=2$ 层的 token 剪枝，发现深层注意力（$D=18$）比浅层（$D=2$）的 RelAcc 高出约 14 个百分点。这是因为深层更靠近预测头，其注意力图能更准确理解多模态 token 之间的关系。

**先导实验 2（时间开销分析）**：当生成长度 $S \geq 32$ 时，decoding 时间远超 prefilling 时间。FastV 虽然有效减少 prefilling 时间，但在 decoding 阶段仅获得极有限的加速（RelSpd 仅 104.3%）。

**关键问题**：能否在一个统一框架中同时解决注意力质量差和 decoding 加速不足这两个问题？

## 方法详解

### 整体框架

TwigVLM 的核心思想简洁优雅：在预训练 VLM 的早期层（第 $K$ 层）上附加一个轻量级 twig 模块（$T$ 层 transformer），形成一个浅层网络 $\mathcal{M}_s$。这个 twig 模块在训练后同时服务于两个目的：
1. **Prefilling 阶段**：用 twig 层的深层注意力（而非 VLM 早期层的浅层注意力）引导视觉 token 剪枝（TTP）
2. **Decoding 阶段**：浅层网络作为 draft 模型，深层 VLM 作为 target 模型，执行自推测解码（SSD）

### 关键设计

#### 1. **Twig 模块的架构与训练**

- **功能**：在 VLM 的第 $K$ 层后附加 $T$ 层 transformer 块，构建一个浅层子网络
- **核心思路**：
    - 深层 VLM 记为 $\mathcal{M}_b = \{\mathcal{T}_l\}_{l=1}^L$，twig 模块为 $\{\mathcal{G}_t\}_{t=1}^T$
    - 浅层网络 $\mathcal{M}_s = \{\mathcal{T}_k\}_{k=1}^K \cup \{\mathcal{G}_t\}_{t=1}^T$，其中 $K+T \ll L$
    - 初始化策略：twig 的 $T$ 层从 VLM 的第 $K+1$ 到 $K+T$ 层复制权重（最优选择，因为输入分布最匹配）
    - 训练时冻结 VLM 的全部参数，仅训练 twig 模块（$T$ 层 + 预测头），使用与 VLM 相同的训练数据和标准自回归损失
    - 训练开销仅约为 VLM 训练的 10%
- **设计动机**：从 VLM 相邻层初始化确保 twig 层的输入-输出分布与 VLM 对齐，使得 twig 能以极小的训练代价获得接近深层网络的语义理解能力

#### 2. **Twig 引导的 Token 剪枝 (TTP)**

- **功能**：利用 twig 最后一层的注意力图指导第 $K$ 层的视觉 token 剪枝
- **核心思路**：

$$\hat{\mathbf{X}}_{\mathcal{M}_b}^{(K)} = \mathcal{P}(\mathbf{X}_{\mathcal{M}_b}^{(K)}, \mathbf{A}_{\mathcal{M}_s}^{(K+T)}, R)$$

  其中 $\mathcal{P}$ 为 FastV 风格的 TopR 选择函数，但注意力来源从第 $K$ 层替换为 twig 的第 $K+T$ 层。

  额外引入 **FinalWipe** 策略：在 VLM 的第 $K_f$ 层（如 24 层）之后移除所有视觉 token。此时平均保留 token 数重新定义为：

$$\bar{R} = [M \times K + R \times (K_f - K)] / L$$

  这允许在相同 $\bar{R}$ 下使用更大的 $R$，从而保留更多中间层视觉 token 以提升精度。

- **设计动机**：先导实验 1 证明深层注意力比浅层注意力对 token 选择更有效；twig 层虽然与 VLM 第 $K+T$ 层同深度，但因为直接连接预测头，其注意力图质量更高（96.0% vs. 86.2% RelAcc）

#### 3. **自推测解码 (SSD)**

- **功能**：利用浅层网络快速生成 draft token，再由深层网络并行验证，加速 decoding
- **核心思路**：
    - 浅层 $\mathcal{M}_s$（draft 模型）自回归生成多个 draft token
    - 深层 $\mathcal{M}_b$（target 模型）在单次前向传播中并行验证这些 token
    - 被接受的 token 直接作为最终输出，被拒绝的从该位置重新采样
    - 关键优势：draft 和 target 共享前 $K$ 层的计算和 KV-cache，进一步提升效率
    - SSD 产生与 target 模型**完全相同**的输出——精度不受影响
- **设计动机**：先导实验 2 证明 decoding 是长回答场景的主要瓶颈；TwigVLM 天然包含深浅两个子网络，无需额外模型即可实现推测解码

### 损失函数 / 训练策略

- **损失函数**：标准自回归语言建模损失
- **训练策略**：仅训练 twig 块，冻结 VLM 所有参数
- **默认超参数**：$T=3$, $K=2$, $K_f=24$
- **训练数据**：与 base VLM 使用相同的多模态指令微调数据
- **硬件**：8×A100 GPU

## 实验关键数据

### 主实验

**LLaVA-1.5-7B 上剪枝 88.9% 视觉 token（保留平均 64 个 token）**：

| 方法 | GQA | MMB | MME | VQA-T | SQA-I | VQA-V2 | RelAcc |
|------|-----|-----|------|-------|-------|--------|--------|
| LLaVA-1.5-7B (上界) | 61.9 | 64.7 | 1862 | 58.2 | 69.5 | 78.5 | 100% |
| FastV | 44.1 | 45.9 | 1218 | 50.7 | 70.0 | 52.0 | 77.0% |
| VisionZip‡ | 57.0 | 61.5 | 1756 | 56.0 | 68.8 | 74.2 | 95.2% |
| **TwigVLM** | **58.8** | **60.4** | **1760** | **55.8** | **70.0** | **75.6** | **96.0%** |

**生成速度对比（MM-Vet 长回答，$\bar{S}$ ≈ 100 tokens）**：

| 方法 | RelSpd |
|------|--------|
| FastV ($\bar{R}$=64) | ~104% |
| VisionZip ($\bar{R}$=64) | ~106% |
| **TwigVLM ($\bar{R}$=64)** | **~154%** |

**Video-LLaVA 视频理解（保留 135 个 token）**：

| 方法 | TGIF | MSVD | MSRVTT | ActivityNet | RelAcc |
|------|------|------|--------|-------------|--------|
| FastV | 23.1 | 38.0 | 19.3 | 30.6 | 52.1% |
| VisionZip | 42.4 | 63.5 | 52.1 | 43.0 | 93.2% |
| **TwigVLM** | **44.7** | **68.3** | **54.6** | 41.5 | **96.3%** |

### 消融实验

| 配置 | RelAcc | RelSpd | 说明 |
|------|--------|--------|------|
| 注意力来自 VLM 第 K 层 | 82.3% | - | 浅层注意力信号差 |
| 注意力来自 VLM 第 K+T 层 | 86.2% | - | 同深度但距离预测头远 |
| **注意力来自 twig 最后层** | **96.0%** | - | 靠近预测头，质量最高 |
| 仅 FastV token 剪枝 | - | 104.3% | 仅加速 prefilling |
| 仅 SSD | - | 146.7% | 仅加速 decoding |
| **TTP + SSD** | - | **153.6%** | 两者互补，加速最大 |
| 随机初始化 twig | 87.2% | 120.4% | 缺乏 VLM 知识 |
| 从 VLM 最后 T 层初始化 | 90.4% | 131.4% | 分布不匹配 |
| **从 VLM K:K+T 层初始化** | **96.0%** | **153.6%** | 最优，分布最匹配 |
| T=1 | 93.9% | 154.1% | 精度不足 |
| **T=3** | **96.0%** | **153.6%** | 精度-速度最优平衡 |
| T=4 | 95.8% | 145.4% | 速度下降 |

### 关键发现

1. **注意力深度决定剪枝质量**：twig 层注意力比 VLM 同深度层高 9.8% RelAcc，证实"距离预测头越近的注意力越有效"
2. **TTP 和 SSD 完美互补**：TTP 加速 prefilling，SSD 加速 decoding，两者组合实现 153.6% 的整体加速
3. **初始化策略至关重要**：从 VLM 相邻层初始化比随机初始化高 8.8% RelAcc 和 33.2% RelSpd
4. **FinalWipe 提升精度**：在后期层移除视觉 token 不降低精度（因视觉 token 在深层贡献已减弱），反而允许中间层保留更多 token
5. **视频任务泛化性强**：TwigVLM 在 Video-LLaVA 上的 RelAcc 从 VisionZip 的 93.2% 提升至 96.3%

## 亮点与洞察

1. **先导实验驱动的设计**：两个先导实验精准定位了现有方法的两个关键弱点，方法设计围绕这两个发现展开，逻辑清晰
2. **一个模块两种用途**：twig 模块同时用于 token 选择和推测解码，架构设计极其优雅
3. **SSD 不损失精度**：自推测解码产生与原始模型完全相同的输出，精度完全不受 SSD 影响
4. **长回答场景的突破**：在 MM-Vet 等需要生成长回答的场景中，TwigVLM 的加速优势（154%）远超仅做 token 剪枝的方法（~104%）
5. **训练效率高**：仅训练约 VLM 参数的一小部分，训练时间仅为 VLM 的约 10%

## 局限与展望

1. **Twig 模块引入额外计算**：虽然 twig 很浅（3 层），但在 prefilling 阶段仍增加了注意力计算开销
2. **依赖后训练**：需要对每个 base VLM 分别训练 twig 模块，不如纯推理时方法（如 FastV）灵活
3. **超参数敏感性**：$K$、$T$、$K_f$ 的最优组合需要针对每个 VLM 单独调优
4. **未探索更大模型**：主实验集中在 7B 模型，更大模型（如 72B）上的效果尚未验证
5. **SSD 的 token 接受率是瓶颈**：当 twig 和 VLM 的分布差异大时，draft token 的接受率会下降

## 相关工作与启发

- 与 FastV 的关系：TwigVLM 解决了 FastV 的两个根本问题——浅层注意力不敏感和无法加速 decoding
- 与 VisionZip 的区别：VisionZip 在视觉编码阶段剪枝，不感知文本 prompt；TwigVLM 在 LLM 内部剪枝，能利用文本信息选择相关视觉 token
- 与独立推测解码方法的区别：传统推测解码需要一个独立的小模型，而 TwigVLM 的 draft 和 target 共享前 $K$ 层，无需额外模型

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 一个优雅的统一框架同时解决 token 剪枝和 decoding 加速两个问题
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三个 VLM 基准（图像+视频），六项消融，速度对比详尽
- **写作质量**: ⭐⭐⭐⭐⭐ — 先导实验→动机→方法→实验的叙事逻辑极其清晰
- **价值**: ⭐⭐⭐⭐⭐ — 首次揭示 decoding 加速对 VLM 部署的重要性，提供了简洁有效的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves](ideator_jailbreaking_and_benchmarking_large_visionlanguage_m.md)
- [\[ICCV 2025\] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving](fine-grained_evaluation_of_large_vision-language_models_in_autonomous_driving.md)
- [\[ICCV 2025\] MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models](multiverse_a_multi-turn_conversation_benchmark_for_evaluating_large_vision_and_l.md)
- [\[CVPR 2025\] MBQ: Modality-Balanced Quantization for Large Vision-Language Models](../../CVPR2025/multimodal_vlm/mbq_modality-balanced_quantization_for_large_vision-language_models.md)
- [\[ICCV 2025\] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)

</div>

<!-- RELATED:END -->
