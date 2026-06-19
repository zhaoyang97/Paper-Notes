---
title: >-
  [论文解读] Masking Meets Supervision: A Strong Learning Alliance
description: >-
  [CVPR 2025][模型压缩][监督学习] 提出 Masked Sub-branch (MaskSub)——在监督学习中引入高比例 (50%) mask 增强的通用框架，通过主分支(无mask)和子分支(有mask)的自蒸馏结构解决强 mask 增强导致训练不稳定的问题，在 DeiT-III、MAE 微调、CLIP 微调、BERT 训练以及 ResNet/Swin 等多种场景中均取得一致性能提升。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "监督学习"
  - "mask增强"
  - "自蒸馏"
  - "ViT训练"
  - "正则化"
---

# Masking Meets Supervision: A Strong Learning Alliance

**会议**: CVPR 2025  
**arXiv**: [2306.11339](https://arxiv.org/abs/2306.11339)  
**代码**: [https://github.com/naver-ai/augsub](https://github.com/naver-ai/augsub)  
**领域**: 模型压缩/训练优化  
**关键词**: 监督学习, mask增强, 自蒸馏, ViT训练, 正则化

## 一句话总结

提出 Masked Sub-branch (MaskSub)——在监督学习中引入高比例 (50%) mask 增强的通用框架，通过主分支(无mask)和子分支(有mask)的自蒸馏结构解决强 mask 增强导致训练不稳定的问题，在 DeiT-III、MAE 微调、CLIP 微调、BERT 训练以及 ResNet/Swin 等多种场景中均取得一致性能提升。

## 研究背景与动机

Masked Image Modeling（MAE 等）通过高比例随机 mask（>50%）实现了强大的自监督预训练，但监督学习一直无法有效利用这种强 mask 增强：

- **高 mask 比例在监督学习中失效**：当 mask 比例超过 50%，直接用 cross-entropy 训练会严重退化——标准 train loss 不收敛，验证精度下降
- **正则化强度过界**：高 mask 是一种极强的正则化，超过最优正则化窗口后对 loss 收敛产生"恶性效果"
- **监督学习的进步空间**：现有 SOTA 训练配方（DeiT-III、RSB）已经精心调优了正则化组合，在其基础上额外增加强正则化极具挑战性
- **MIM 与监督学习的差距缩小**：如 DeiT-III 所示，新的监督学习配方已追上 MAE 预训练的效果，但二者各有优势，如何融合 mask 的正则化能力到监督学习中成为关键问题

核心动机：设计一种架构，将强 mask 增强从主训练流程中"隔离"出来，通过自蒸馏（以主分支输出为 soft label）为子分支提供放松的学习目标，避免强正则化对主损失收敛的干扰。

## 方法详解

### 整体框架

MaskSub 在标准监督训练基础上增加一个子分支：主分支 $f_\theta(\mathbf{x}|r_{mask}=0)$ 不做 mask，用标准 cross-entropy 训练；子分支 $f_\theta(\mathbf{x}|r_{mask}=r)$ 做高比例 mask（默认 50%），以主分支的 softmax 输出作为蒸馏目标训练。两个分支**共享全部参数**，总损失为主分支 CE 损失 + 子分支蒸馏损失的平均。

### 关键设计

1. **自蒸馏子分支结构**:
    - 功能：将强 mask 增强的训练信号与主损失解耦，避免相互干扰
    - 核心思路：主分支正常训练（stop-gradient），子分支用主分支的 softmax 输出 $\sigma(f_\theta(\mathbf{x}|r_{mask}=0))$ 作为 soft label，替代 one-hot ground-truth。总损失 = $\frac{1}{2}[\text{CE}(o_1, \text{label}) + \text{CE}(o_2, \text{softmax}(o_1.\text{detach}()))]$。子分支使用 MAE 风格 mask——直接移除 masked token 以减少计算量（50% mask ≈ 1.5× 标准训练成本）
    - 设计动机：(1) 直接对子分支用 hard label 训练会因 mask 引入的高难度导致不稳定；(2) 主分支的 soft label 是放松的、自适应的——主分支输出好时子分支目标与 GT 接近（难任务），输出差时目标模糊（容易任务），实现自动难度控制

2. **自动难度控制机制**:
    - 功能：随训练进展自适应调整子分支的学习难度
    - 核心思路：通过分析梯度大小验证：训练初期子分支梯度小（目标松弛，学习容易），随着主分支性能提升，子分支梯度逐渐增大（目标接近 GT，学习变难）。这与知识蒸馏文献的发现一致——弱教师产生简单目标，强教师产生挑战性目标
    - 设计动机：可视为 sample-wise mask 增强——仅对主分支已成功分类的样本施加挑战性 mask 训练，对失败样本不施加额外压力

3. **扩展到其他 Drop 正则化**:
    - 功能：MaskSub 框架不限于 mask 增强，可扩展到所有 drop-based 正则化
    - 核心思路：三种变体——(1) **MaskSub**：子分支做随机 patch mask，性能提升最大；(2) **DropSub**：子分支做强 dropout（element-wise 随机 drop），在 ViT 训练中重新启用被放弃的 dropout；(3) **PathSub**：子分支使用更高的 drop-path 概率。所有变体仅需将 $r_{mask}$ 替换为对应 drop 概率即可
    - 设计动机：Drop-based 正则化（dropout、drop-path）在过强时同样会退化监督学习，MaskSub 框架提供了统一的解决方案

### 损失函数

- 主分支损失：$\mathcal{L}_{main} = \text{CE}(\sigma(f_\theta(\mathbf{x}|r=0)), \mathbf{y})$（或 BCE 版本）
- 子分支损失：$\mathcal{L}_{sub} = \text{CE}(\sigma(f_\theta(\mathbf{x}|r)), \sigma(f_\theta(\mathbf{x}|r=0)).\text{detach()})$
- 总损失：$\mathcal{L} = \frac{1}{2}(\mathcal{L}_{main} + \mathcal{L}_{sub})$
- 无额外超参数——mask 比例统一设为 50%，在所有实验中不变

## 实验关键数据

### 主实验（ImageNet-1k 从头训练，DeiT-III 配方）

| 网络 | DeiT-III 400ep | + MaskSub | DeiT-III 800ep | + MaskSub |
|------|---------------|-----------|---------------|-----------|
| ViT-S/16 | 80.4 | **81.1** (+0.7) | 81.4 | **81.7** (+0.3) |
| ViT-B/16 | 83.5 | **84.1** (+0.6) | 83.8 | **84.2** (+0.4) |
| ViT-L/16 | 84.5 | **85.2** (+0.7) | 84.9 | **85.3** (+0.4) |
| ViT-H/14 | 85.1 | **85.7** (+0.6) | 85.2 | **85.7** (+0.5) |

### 微调实验

| 预训练方法 | 网络 | Baseline | + MaskSub |
|-----------|------|----------|-----------|
| MAE | ViT-B/16 | 83.6 | **83.9** (+0.3) |
| MAE | ViT-H/14 | 86.9 | **87.2** (+0.3) |
| BEiTv2 | ViT-B/16 | 85.5 | **85.6** (+0.1) |
| CLIP | ViT-B/16 | 84.8 | **85.2** (+0.4) |
| CLIP | ViT-L/14 | 87.5 | **87.8** (+0.3) |

### 层次架构（ResNet + Swin Transformer）

| 网络 | Baseline | + MaskSub |
|------|----------|-----------|
| ResNet-50 (RSB) | 79.7 | **80.0** (+0.3) |
| ResNet-101 (RSB) | 81.4 | **82.1** (+0.7) |
| ResNet-152 (RSB) | 81.8 | **82.8** (+1.0) |
| Swin-T | 81.3 | **81.4** (+0.1) |
| Swin-S | 83.0 | **83.4** (+0.4) |
| Swin-B | 83.5 | **83.9** (+0.4) |

### 与 SOTA 预训练方法对比

| 方法 | ViT-B Epochs | Top-1 | 计算成本 |
|------|-------------|-------|---------|
| MAE | 1600 | 83.6 | - |
| DeiT-III | 800 | 83.8 | ×1.0 |
| CoSub | 800 | 84.2 | ×2.0 |
| **MaskSub** | **400** | **84.1** | **×0.75** |
| **MaskSub** | **800** | **84.2** | **×1.5** |

### 关键发现

- **ViT-H + MaskSub 400ep (85.7) > ViT-H DeiT-III 800ep (85.2)**：用一半训练量超越原始配方
- MaskSub 400ep 已达到或超越 MAE 1600ep 预训练的微调精度——监督学习+MaskSub 效率极高
- 在 ResNet-152 上提升 1.0%（81.8→82.8），在已经极度优化的 RSB 配方上仍有显著增益
- 训练分析证实：MaskSub 同时改善了标准 loss 和 mask loss 的收敛速度——不是简单的正则化 tradeoff
- 子分支梯度从小到大的变化模式验证了自动难度控制的假设
- 与 CoSub 相比：同等性能但计算成本仅为其 75%（MaskSub 400ep vs CoSub 800ep）
- 无额外数据增强、无额外参数、无额外优化器步骤——实现极致简洁

## 亮点与洞察

- **极致简洁的设计**：核心实现仅需 ~10 行 PyTorch 代码（见 Algorithm 1），无超参数调节（统一 50% mask），无额外模型参数
- **解耦思想的优雅应用**：将"强正则化"从主训练流中分离到子分支，通过自蒸馏连接——既保护主 loss 收敛，又充分利用了 mask 的正则化收益
- **自动难度控制的深度洞察**：MaskSub 本质上实现了 sample-wise、epoch-wise 的自适应正则化强度——早期放松、后期加强，与课程学习的精神一致
- **跨架构普适性**：从 ViT 到 ResNet 到 Swin，从监督预训练到 MAE/BEiT/CLIP 微调到 BERT——这种广泛的适用性证明了框架的通用性

## 局限与展望

- 对层次架构（ResNet/Swin）不适用 MAE-style token 移除，需填充零值/mask-token，导致计算量翻倍
- 50% 统一 mask 比例虽然简洁但可能非最优——不同任务/模型/训练阶段的最优比例可能不同
- 仅展示了分类任务的效果，对检测/分割等密集预测任务的影响未验证
- 与 CoSub 等方法的理论差异（co-training vs self-distillation）虽有讨论但缺乏深入分析
- 子分支梯度的 stop-gradient 设计使得 mask 的正则化效果不会回传影响主分支——是否存在更优的梯度设计值得探索

## 相关工作与启发

- 与 MAE 的本质区别：MAE 用 mask 做自监督重建，MaskSub 用 mask 做监督学习的正则化——两者目标完全不同但共享 mask 操作
- 与 CoSub 的区别：CoSub 用 drop-path 构建子分支做 co-training（两分支互学），MaskSub 用 mask 构建子分支做单向自蒸馏（主分支教子分支），MaskSub 更通用且计算更高效
- 与 SupMAE 的区别：SupMAE 在 MAE 训练中加入监督损失（MAE 配方+监督），MaskSub 在监督训练中加入 mask（监督配方+mask），完全相反的融合方向
- 启发：强正则化在监督学习中"失效"的根本原因不是正则化本身有害，而是直接施加于主损失时破坏了收敛——通过分支解耦可以释放被浪费的正则化潜力

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 mask 增强引入监督学习的子分支自蒸馏设计简洁有效，但与已有自蒸馏/双分支框架有一定相似
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 ViT/ResNet/Swin、预训练/微调/BERT、多种配方，极其全面
- 写作质量: ⭐⭐⭐⭐⭐ 分析透彻（loss 曲线、梯度分析、难度控制），伪代码简洁直观，论文结构组织优秀
- 价值: ⭐⭐⭐⭐⭐ 提供了一种通用、零超参、即插即用的监督学习改进方案，适用范围极广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Disentangling Latent Shifts of In-Context Learning with Weak Supervision](../../NeurIPS2025/model_compression/disentangling_latent_shifts_of_in-context_learning_with_weak_supervision.md)
- [\[ICML 2025\] Weak-to-Strong Jailbreaking on Large Language Models](../../ICML2025/model_compression/weak-to-strong_jailbreaking_on_large_language_models.md)
- [\[NeurIPS 2025\] Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing](../../NeurIPS2025/model_compression/synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)
- [\[CVPR 2025\] Incremental Object Keypoint Learning (KAMP)](incremental_object_keypoint_learning.md)
- [\[ICCV 2025\] ACAM-KD: Adaptive and Cooperative Attention Masking for Knowledge Distillation](../../ICCV2025/model_compression/acam-kd_adaptive_and_cooperative_attention_masking_for_knowledge_distillation.md)

</div>

<!-- RELATED:END -->
