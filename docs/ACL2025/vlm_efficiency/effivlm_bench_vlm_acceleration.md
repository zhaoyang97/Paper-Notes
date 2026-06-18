---
title: >-
  [论文解读] EffiVLM-Bench: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models
description: >-
  [ACL 2025][多模态VLM][VLM加速] 提出 EffiVLM-Bench 统一评估框架，从性能、泛化性、忠实度和效率四个维度系统评估 LVLM 免训练加速方法（token 压缩 + 参数压缩），覆盖 3 个前沿模型和 17 个基准任务，揭示各方法在不同压缩率下的 Pareto 最优权衡。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "VLM加速"
  - "免训练压缩"
  - "Token压缩"
  - "参数量化"
  - "Pareto最优"
---

# EffiVLM-Bench: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models

**会议**: ACL 2025  
**arXiv**: [2506.00479](https://arxiv.org/abs/2506.00479)  
**代码**: [项目页面](https://effivlm-bench.github.io/)  
**领域**: 多模态VLM / 模型加速  
**关键词**: VLM加速, 免训练压缩, Token压缩, 参数量化, Pareto最优

## 一句话总结

提出 EffiVLM-Bench 统一评估框架，从性能、泛化性、忠实度和效率四个维度系统评估 LVLM 免训练加速方法（token 压缩 + 参数压缩），覆盖 3 个前沿模型和 17 个基准任务，揭示各方法在不同压缩率下的 Pareto 最优权衡。

## 研究背景与动机

**领域现状**：大型视觉语言模型（LVLM）在多模态 AI 任务上取得了卓越成功，但巨大的计算和内存开销严重阻碍了实际部署。为提升效率，研究者探索了免训练加速方法——无需重新训练即可降低推理成本，主要分为 token 压缩（消除冗余 token）和参数压缩（剪枝/量化减少参数量）两大类。

**现有痛点**：现有加速方法的评估存在三个关键不足：(1) **模型架构过时**——评估常停留在 LLaVA/LLaVA-v1.5 等旧模型，未考虑具有动态分辨率处理机制的最新 LVLM（如 Qwen2-VL、InternVL2.5）；(2) **基准有限**——通常仅使用通用 VQA 任务，忽略了 OCR、长文本生成等挑战性更高的任务；(3) **指标单一**——仅关注绝对性能（准确率），忽视了泛化性、忠实度等维度，也缺乏性能-效率 Pareto 最优权衡的系统探索。

**核心矛盾**：缺乏统一的评估框架来全面理解不同加速技术在多样场景下的表现和权衡，阻碍了方法选型和实际部署。

**本文目标** → 为 LVLM 免训练加速领域提供一个统一、全面、可扩展的评估基准。

**切入角度**：定义四个正交评估维度（性能/泛化/忠实度/效率），覆盖最新模型架构和多样任务场景，系统对比主流加速方法。

**核心 idea**：用四维度统一框架终结 LVLM 加速方法各自为政的评估乱象，通过 Pareto 前沿分析为实际部署提供选型依据。

## 方法详解

### 整体框架

EffiVLM-Bench 定义了四个核心评估维度，覆盖 17 个基准任务（从文档理解到数学推理）、3 个前沿 LVLM（LLaVA-OneVision-7B、Qwen2-VL-7B、InternVL2.5-38B），以及两大类加速方法（token 压缩和参数压缩），在 1%/10%/40% 等不同压缩率下进行全面对比。

### 关键设计

1. **四维评估指标体系**:

    - 功能：从互补的四个角度全面衡量压缩方法的质量
    - 核心思路：
        - **整体性能 (OP)**：$OP^{m,c} = \sqrt{\frac{1}{B}\sum_{b=1}^{B}\mathbb{E}\left(\frac{EM_b^{m,c}}{EM_b^m}\right)^2}$，压缩模型与原始模型在各基准上的评估指标比值的均方根平均
        - **泛化性 (OG)**：跨基准和模型的性能变异系数，值越低表示压缩方法的行为越稳定
        - **忠实度 (OL)**：$OL^c = \mathbb{E}_{b,m}[\mathbb{I}(P_b^{m,c}, P_b^m)]$，压缩模型与原始模型预测的一致程度，确保压缩不引入新偏差
        - **效率 (OE)**：基于实际推理时间的加速比（非 FLOPs 等理论指标），直接反映真实延迟
    - 设计动机：仅看准确率会掩盖压缩带来的行为偏移——高准确率但低忠实度意味着模型"答对了但原因不同"

2. **全面的方法覆盖与压缩率阶梯**:

    - 功能：在统一条件下对比两大类方法在多个压缩率下的表现
    - 核心思路：**Token 压缩**包括 token 剪枝（FastV 动态剪枝、VisionZip 视觉编码器级剪枝、PruMerge+ 先剪后合）和 KV cache 压缩（StreamingLLM 滑窗注意力、H2O 重度命中者淘汰、SnapKV/PyramidKV 分层策略、LOOK-M/VL-Cache 多模态感知压缩）；**参数压缩**包括权重剪枝（EcoFLAP、Wanda、SparseGPT）和量化（AWQ int4、GPTQ int4）。所有方法在 1%/10%/40%/100% token 保留率或对应参数压缩级别下测试
    - 设计动机：不同方法的优劣在不同压缩率下可能翻转，需要全谱对比才能找到 Pareto 最优前沿

### 损失函数 / 训练策略

EffiVLM-Bench 评估的均为**免训练方法**，不涉及额外训练过程。被评估方法通过以下策略实现压缩：token 剪枝基于注意力分数或文本-视觉相关性选择保留的 token；KV cache 压缩利用注意力稀疏性选择较少的 key-value 对减少内存；权重剪枝基于权重重要性度量（如 Wanda 的权重×激活度量）移除冗余参数；量化将全精度权重转换为 int4/int8 低精度格式。

## 实验关键数据

### 主实验（LLaVA-OneVision-7B 上 Token 压缩对比）

| 方法 | 保留率 | DocVQA | ChartQA | OCRBench | MMMU | MMBench | OP |
|------|--------|--------|---------|----------|------|---------|-----|
| Original | 100% | 87 | 80.00 | 595 | 45.44 | 83.12 | 1.00 |
| FastV | 40% | 80 | 69.20 | 488 | 46.33 | 81.22 | 0.94 |
| VisionZip | 40% | 72 | 67.04 | 500 | 46.11 | 80.43 | 0.93 |
| PruMerge+ | 40% | 49 | 51.40 | 382 | 45.55 | 80.88 | 0.88 |
| FastV | 10% | 48 | 43.16 | 190 | 45.33 | 70.12 | 0.76 |
| VisionZip | 10% | 56 | 49.88 | 352 | 44.11 | 78.86 | 0.84 |
| FastV | 1% | 8 | 14.00 | 27 | 40.89 | 27.69 | 0.48 |
| VisionZip | 1% | 35 | 35.16 | 194 | 42.56 | 74.10 | 0.75 |

### 消融实验（跨模型泛化性对比，40% 保留率）

| 方法 | LLaVA-OV-7B OP | Qwen2-VL-7B OP | InternVL2.5-38B OP |
|------|---------------|----------------|-------------------|
| FastV | 0.94 | 0.92 | 0.91 |
| VisionZip | 0.93 | 0.93 | 0.89 |
| PruMerge+ | 0.88 | 0.91 | 0.85 |

### 关键发现

- **Token 压缩性能高度依赖任务和模型**：在 40% 保留率下三种 token 剪枝方法均保持 OP > 0.88，但在 1% 极端压缩下差异巨大——VisionZip 仍保持 OP=0.75 而 FastV 暴跌至 0.48
- **文档理解类任务最脆弱**：DocVQA 和 ChartQA 对 token 压缩最敏感（1% 时 FastV 的 DocVQA 从 87 降至 8），因为这类任务需要精确的局部视觉信息
- **量化是最实用的加速方案**：AWQ int4 量化在保持接近原始性能（OP > 0.95）的同时提供约 2× 加速，且对任务类型不敏感
- **KV cache 压缩的多模态感知策略优于 LLM 方法**：VL-Cache 和 LOOK-M 通过区分视觉和文本 token 的不同角色，在同等压缩率下优于直接移植自 LLM 的通用 cache 压缩方法
- **Pareto 前沿分析**：在性能-效率权衡中，token 剪枝方法在低压缩率（40%+）时主导 Pareto 前沿，而量化方法在高压缩率下更优

## 亮点与洞察

- **四维评估框架**的设计理念非常高明——特别是忠实度（OL）指标弥补了传统评估的盲区：一个方法可能在某些样本上"蒙对新答案"但在其他样本上"丢失正确答案"，单看准确率无法发现
- **Pareto 前沿分析**为实际部署提供了直接的选型建议——根据延迟预算从 Pareto 曲线上直接选方法
- **1% 极端压缩的对比**揭示了各方法的鲁棒性底线，VisionZip 在极低预算下的优势来自其视觉编码器级别的压缩而非 LLM 层面
- **模块化设计**支持后续扩展新模型和新方法

## 局限与展望

- 仅覆盖三个 LVLM 模型，未包含更大规模模型（如 72B/110B 级别）或更多架构变体
- 未评估多种压缩方法的组合效果——token 压缩和参数压缩联合使用可能产生协同或冲突效应
- 效率仅以推理时间衡量，未报告内存占用和吞吐量等同样重要的部署指标
- 缺乏对生成质量的细粒度评估——LVLM 的长文本生成场景（如详细描述）的质量可能比准确率下降更早劣化
- 未覆盖最新的注意力架构变体（如 GQA、MQA）对压缩方法效果的影响

## 相关工作与启发

- Token 压缩方面，LLaVA-PruMerge（Shang 2024）首次在视觉编码器阶段进行剪枝合并，VL-Cache（Tu 2024）引入模态感知 cache 分配策略；参数压缩方面，SparseGPT（Frantar 2023）和 AWQ（Lin 2024）是两种主流路径
- 本文的统一评估框架思路可推广到其他领域——如 LLM 推理加速、语音模型压缩等
- **启发**：(1) 实用部署应优先考虑 AWQ int4 量化作为基线加速方案；(2) 高压缩场景下多模态感知的 token 压缩策略是关键差异化因素；(3) 评估框架应多维度化，单一指标容易误导方法选型

## 评分

⭐⭐⭐⭐ 填补了 LVLM 加速领域缺乏统一评估框架的空白，四维指标设计合理、实验覆盖全面，Pareto 前沿分析对实际部署有直接指导价值，但模型覆盖范围有限且未探索组合压缩策略。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service](vlminferslow_evaluating_the_efficiency_robustness_of.md)
- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [\[AAAI 2026\] VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)
- [\[ACL 2025\] JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games](jarvis-vla_post-training_large-scale_vision_language_models_to_play_visual_games.md)
- [\[AAAI 2026\] CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/crossvid_a_comprehensive_benchmark_for_evaluating_cross-vide.md)

</div>

<!-- RELATED:END -->
