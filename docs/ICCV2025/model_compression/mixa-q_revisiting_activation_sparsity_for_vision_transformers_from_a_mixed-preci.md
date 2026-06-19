---
title: >-
  [论文解读] MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective
description: >-
  [ICCV 2025][模型压缩][量化] 提出 MixA-Q，一种混合精度激活量化框架，将窗口级激活稀疏性（原本用于剪枝）转化为量化维度的利用——对不重要的窗口分配更低比特宽度而非完全跳过计算，在 COCO 目标检测上实现 PTQ 无损 1.35× 加速和 QAT 无损 1.25× 加速，同时具有更好的 OOD 鲁棒性。
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "量化"
  - "Activation Sparsity"
  - "Transformer"
  - "Efficient Inference"
---

# MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective

**会议**: ICCV 2025  
**arXiv**: [2507.19131](https://arxiv.org/abs/2507.19131)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: Mixed-Precision Quantization, Activation Sparsity, Vision Transformer, Swin Transformer, Efficient Inference

## 一句话总结

提出 MixA-Q，一种混合精度激活量化框架，将窗口级激活稀疏性（原本用于剪枝）转化为量化维度的利用——对不重要的窗口分配更低比特宽度而非完全跳过计算，在 COCO 目标检测上实现 PTQ 无损 1.35× 加速和 QAT 无损 1.25× 加速，同时具有更好的 OOD 鲁棒性。

## 研究背景与动机

Swin Transformer 通过层级结构和移动窗口机制降低了 ViT 的二次复杂度，成为密集预测任务（目标检测、分割）的强力骨干网络。然而随着高分辨率图像的普及，自注意力操作的计算开销仍然是实时应用（如自动驾驶）的延迟瓶颈。

现有解决方案分两条路线：

**激活剪枝（Activation Pruning）**：如 SparseViT，利用自然图像中"并非所有像素都重要"的稀疏性，跳过不重要窗口/token 的计算。但存在三个问题——(a) 需要重训练；(b) 高剪枝率时精度显著下降（信息完全丢失）；(c) 依赖简单的窗口选择准则（如 L2 范数），面对分布外（OOD）输入时选择可能不准确，在安全关键应用中不可靠。

**量化（Quantization）**：降低权重和激活的精度表示。混合精度量化（MPQ）可以给不同层分配不同比特宽度。但现有 MPQ 方法大多关注**层间**比特分配，忽略了**层内**的激活稀疏性——同一层的不同窗口/区域重要性差异很大。

本文的核心洞察是：**从混合精度量化角度利用激活稀疏性**——对不重要区域不是跳过计算（剪枝），而是用更低精度计算。这样既保留了信息（不完全丢失），又减少了计算量。而且这种方法可以**无需训练**地与 PTQ 方法结合，且最坏情况下（窗口选择完全错误）性能仍有下界保证。

## 方法详解

### 整体框架

MixA-Q 替换 Swin Transformer 中的标准 Swin Block 为 **Two-Branch Swin Block**：根据窗口重要性分数（L2 范数）将窗口分为高精度组和低精度组，分别通过高/低精度分支处理，最后合并回特征图。压缩比率（低精度窗口比例）通过进化搜索优化。

### 关键设计

1. **从剪枝到混合精度执行（From Pruning to MP Execution）**:

    - 功能：将 SparseViT 的窗口剪枝替换为混合精度执行
    - 核心思路：SparseViT 跳过不重要窗口并转发上一层特征；MixA-Q 保留不重要窗口的计算但用更低精度（如 4bit→2bit），减少大约 50% 计算量而非 100% 节省。这带来四大优势：(a) 可免训练集成 PTQ；(b) 高压缩率时避免信息完全丢失；(c) OOD 输入下性能有下界保证（最坏情况退化为全低精度而非丢失信息）；(d) 动态激活蒸馏可降低量化误差
    - 设计动机：剪枝和混合精度不是非此即彼——剪枝效率更高（省 100% 计算）但代价更大（信息完全丢失）；混合精度效率略低（省约 50%）但更安全（保留信息）

2. **双分支 Swin Block（Two-Branch Swin Block）**:

    - 功能：替代原始 Swin Block 以支持混合精度窗口计算
    - 核心思路：(a) 每个 stage 首个 block 前计算一次窗口重要性分数（L2 范数），整个 stage 共享；(b) 按压缩比率将窗口分为高/低精度两组并 gather；(c) LayerNorm **独立复制**给两个分支（处理分布偏移）；(d) MHA 和 FFN 创建为 **shadow layers**——共享权重和偏置，但维护独立的 step size 和 zero point；(e) 训练时两个分支的梯度通过同一组参数累加
    - 设计动机：独立 LayerNorm 处理高/低精度分支的分布差异；共享 MHA/FFN 权重避免参数膨胀，shadow layers 仅维护量化参数的独立性

3. **进化搜索压缩比率（Evolutionary Search of Compression Ratios）**:

    - 功能：为每对连续 Swin Block 找到最优的低精度窗口比例
    - 核心思路：Swin-Tiny 有 12 个 block，每对共享一个压缩比率，共 6 个变量。将其形式化为双目标优化（最大化 mAP + 最小化 BOPs），用 NSGA-II 搜索。压缩比率离散化为 {0%, 10%, ..., 80%}
    - 设计动机：不同层对压缩的敏感度不同，需要自适应分配

4. **稀疏感知量化适应（Sparsity-Aware Quantization Adaptation, SAQA）**:

    - 功能：让 QAT 模型适应混合精度配置
    - 核心思路：先用均匀 QAT（如 OFQ W4A4）量化模型，然后在 SAQA 中随机采样压缩比率训练（类似 SparseViT 的 SAA）。关键改进是 **Uniform-sum 压缩比率采样**——先均匀采样目标总和 $S$，再用 Dirichlet 分布采样各层比率使总和为 $S$，拒绝任何超过 80% 的样本。这避免了 naive 独立采样导致的 Irwin-Hall 分布偏向中心的问题
    - 设计动机：SAQA 后模型在不同压缩比率下都能给出合理性能估计，无需逐配置重训练。均匀和采样提高搜索效率

5. **动态激活蒸馏（Dynamic Activation Distillation）**:

    - 功能：SAQA 过程中自动降低重要区域的量化误差
    - 核心思路：SAQA 训练中只有重要窗口的梯度经过高精度（4-bit）分支，引导模型聚焦于降低重要窗口的量化噪声。实验发现 SAQA 后早期阶段前景窗口 SQNR 升高、背景窗口 SQNR 降低；后期阶段整体 SQNR 升高（因为早期重要窗口的改善传播到后续特征）
    - 设计动机：这是一个意外发现——MixA-Q 的 SAQA 不仅适应混合精度，还反过来提升了基础量化模型的精度（W4A4 mAP 从 43.1 提升到 43.8）

6. **融合激活剪枝（Activation Pruning Incorporation）**:

    - 功能：在高加速目标下结合剪枝和混合精度
    - 核心思路：最不重要的窗口直接剪枝（省 100%），次不重要但仍相关的窗口低精度处理（省约 50%），重要窗口高精度处理。每对 Swin Block 分配一个剪枝比率和一个压缩比率
    - 设计动机：在 ≤0.7 相对成本时，纯混合精度效果不如纯剪枝（因为剪枝效率更高）；但激进剪枝的信息丢失可以通过保留中间重要度窗口在低精度来弥补

### 损失函数 / 训练策略

- QAT 基础方法：OFQ（W4A4），低精度分支为 W4A2
- PTQ 基础方法：RepQ（W4A8），低精度分支为 W4A4
- SAQA 采用随机压缩比率采样 + Uniform-sum 采样策略
- 进化搜索用 NSGA-II，在 COCO val 上评估 mAP

## 实验关键数据

### 主实验（COCO 目标检测，Mask R-CNN + Swin-Tiny）

| 方法 | Act Bit | BOPs(T) | 加速 | mAP |
|------|---------|---------|------|-----|
| Full-Precision | 32 | 88.42 | - | 46.0 |
| OFQ W4A4 | 4 | 11.05 | 1.0× | 43.1 |
| OFQ W4A3 | 3 | 8.29 | 1.33× | 41.4 |
| SparseViT 1.24× | 3.2* | 8.92 | 1.24× | 43.1 |
| **MixA-Q** (无损) | 4 | 11.05 | 1.0× | **43.8** (+0.7) |
| **MixA-Q** 1.24× | 3.2* | 8.92 | 1.24× | 43.2 |
| **MixA-Q** 1.35× | 2.94* | 8.21 | 1.35× | 42.3 |
| SparseViT 1.53× | 2.6* | 7.24 | 1.53× | 41.4 |
| **MixA-Q+Prun** 1.53× | 2.6* | 7.24 | 1.53× | **42.1** |
| SparseViT 1.82× | 2.2* | 6.08 | 1.82× | 40.1 |
| **MixA-Q+Prun** 1.82× | 2.2* | 6.08 | 1.82× | **40.5** |

### 消融实验

| 实验 | 结果 | 说明 |
|------|------|------|
| SAQA vs 无 SAQA (W4A4 OFQ) | 43.8 vs 43.1 mAP | SAQA 意外提升基础精度 +0.7% |
| SAQA Stage 0/1 mSQNR | 降低 (13.70→13.60 / 9.70→9.50) | 背景窗口量化噪声增大 |
| SAQA Stage 2/3 mSQNR | 升高 (6.90→6.96 / 7.81→8.14) | 重要特征传播导致整体改善 |
| OOD 鲁棒性 (COCO-O Weather) | MixA-Q 退化率 26.5% vs SparseViT 30.2% | 混合精度更鲁棒 |
| 最坏情况 (反向窗口选择) | MixA-Q 退化率 30.6% vs SparseViT 43.4% | 剪枝的灾难性失败 |
| SAQA from fp vs from W4A4 | 42.8 vs 43.4 mAP (3.36* bit) | 从已量化模型开始更好 |
| 窗口提升 (低→高精度) | -0.1% 或持平 | stage 内精度一致性重要 |
| PTQ (RepQ W4A8, 无训练) | 1.35× 无损加速 | 完全免训练的加速 |

### 关键发现

- **意外精度提升**：MixA-Q 的 SAQA 不仅实现混合精度适应，还通过动态激活蒸馏将 W4A4 mAP 从 43.1 提升到 43.8（降低量化退化 24%）
- **MixA-Q 在高相对成本（≥0.8）时优于 SparseViT**，在低相对成本（≤0.7）时结合剪枝后仍优于纯 SparseViT
- **OOD 鲁棒性显著优于剪枝**：SparseViT 在雾天场景中窗口选择被严重干扰，剪枝掉重要窗口导致灾难性退化（43.4%）；MixA-Q 最坏情况退化仅 30.6%
- **PTQ 无训练 1.35× 加速**是一个重要实用结果——比均匀降到 W4A3（退化 1.7 mAP）更好

## 亮点与洞察

- **视角转换的价值**：将激活稀疏性从"剪枝"视角转向"混合精度"视角，解锁了免训练集成 PTQ、OOD 鲁棒性、最坏情况保证等一系列优势
- **动态激活蒸馏的发现**：SAQA 的训练动态自然形成了一种隐式蒸馏——重要窗口的梯度引导模型聚焦于降低这些区域的量化噪声，这是一个优雅的副产品
- **Uniform-sum 采样**：简单但有效地解决了随机采样导致的 Irwin-Hall 分布偏差问题
- **安全关键场景的适用性**：在自动驾驶等安全关键场景中，OOD 鲁棒性比平均精度更重要

## 局限与展望

- 仅在 Swin Transformer（窗口注意力）上验证，对非窗口架构的适用性未知
- 窗口重要性仅用 L2 范数，更复杂的重要性度量可能改善选择质量
- 进化搜索仍需多次推理 COCO val，搜索成本不可忽略
- 未在更新的检测框架（如 DETR 系列）上验证
- 实际硬件加速依赖于混合精度 GEMM 支持（需要 4-bit 和 2-bit 同时可用）

## 相关工作与启发

- **SparseViT**：MixA-Q 的直接基线，用窗口剪枝实现稀疏加速。MixA-Q 将其"剪枝"替换为"降精度"，解决了鲁棒性和免训练问题
- **PMQ**：按注意力分数分配 MLP token 比特宽度，但仅关注 token 级别且需注意力指导
- **Granular-DQ**：在超分辨率中按信息密度动态量化 patch，思路类似但应用领域不同
- **启发**：模型压缩中"降级"比"丢弃"更安全，尤其在安全关键应用中

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts](../../ACL2025/model_compression/moqae_mixed_precision_kv_cache.md)
- [\[ICCV 2025\] OuroMamba: A Data-Free Quantization Framework for Vision Mamba](ouromamba_a_data-free_quantization_framework_for_vision_mamba.md)
- [\[CVPR 2025\] L-SWAG: Layer-Sample Wise Activation with Gradients for Zero-Shot NAS on Vision Transformers](../../CVPR2025/model_compression/l_swag_zero_shot_nas_vision_transformers.md)
- [\[CVPR 2026\] BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers](../../CVPR2026/model_compression/binaryattention_one-bit_qk-attention_for_vision_and_diffusion_transformers.md)
- [\[NeurIPS 2025\] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](../../NeurIPS2025/model_compression/quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)

</div>

<!-- RELATED:END -->
