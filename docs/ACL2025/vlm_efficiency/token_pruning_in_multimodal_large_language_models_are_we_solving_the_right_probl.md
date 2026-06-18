---
title: >-
  [论文解读] Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?
description: >-
  [ACL 2025][多模态VLM][剪枝] 通过大规模基准实验揭示了当前MLLM视觉token剪枝方法的多个根本性问题：精心设计的剪枝策略（FastV、SparseVLM）在多数基准上甚至不如随机选择和池化等朴素方法，原因在于注意力评分的位置偏差、对语言信息的误用、重要性与冗余性的失衡以及评估指标的不可靠。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "剪枝"
  - "多模态大模型"
  - "视觉token压缩"
  - "注意力偏差"
  - "推理加速"
---

# Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?

**会议**: ACL 2025  
**arXiv**: [2502.11501](https://arxiv.org/abs/2502.11501)  
**作者**: Zichen Wen, Yifeng Gao (SJTU), Weijia Li (中山大学), Conghui He (上海AI Lab), Linfeng Zhang (SJTU)
**代码**: 未公开  
**领域**: 多模态VLM  
**关键词**: token pruning, 多模态大模型, 视觉token压缩, 注意力偏差, 推理加速

## 一句话总结

通过大规模基准实验揭示了当前MLLM视觉token剪枝方法的多个根本性问题：精心设计的剪枝策略（FastV、SparseVLM）在多数基准上甚至不如随机选择和池化等朴素方法，原因在于注意力评分的位置偏差、对语言信息的误用、重要性与冗余性的失衡以及评估指标的不可靠。

## 研究背景与动机

### 问题背景
多模态大语言模型（MLLM）在跨模态理解和生成上表现优异，但推理开销巨大。视觉编码器产生大量token（LLaVA-1.5生成576个，LLaVA-NeXT高达2880个），远超文本prompt长度。Token剪枝作为无需训练的加速手段，通过识别冗余视觉token并删除/合并来降低计算和KV缓存开销，吸引了大量关注。

### 已有工作的不足
- FastV利用LLM早期层的注意力分数裁剪低分token，SparseVLM引入文本-视觉交叉注意力引导剪枝，MustDrop在编码-预填充-解码全生命周期压缩token
- 然而这些方法**与随机选择和均匀池化相比并无优势甚至更差**，这一反直觉现象从未被正视
- 已有评估普遍依赖FLOPs和token数量衡量加速效果，忽视实际延迟
- 未考虑Qwen2-VL等模型在训练阶段已执行的token合并压缩

### 核心动机
本文不提出新方法，而是系统性追问五个根本问题：(1) 为什么现有方法不敌随机选择？(2) 注意力评分是否可靠？(3) 语言信息何时有用？(4) 重要性和冗余性如何平衡？(5) 现行评估协议是否全面公正？

## 方法详解

### 整体框架
本文是一项实证分析工作，不提出新架构，而是通过控制变量实验和对比分析，逐一拆解token剪枝的关键设计选择。实验覆盖LLaVA-1.5-7B/13B、LLaVA-Next-7B、Qwen2-VL-7B/72B五个模型，在GQA、MMBench、MME、POPE、ScienceQA、TextVQA、VizWiz、RefCOCO、Visual Haystack等9+数据集上，对比FastV、SparseVLM、MustDrop与Random、Pooling等基线。

### 关键发现1：位置偏差导致注意力选择失效
通过对POPE数据集8910个样本的统计分析发现，FastV依赖最后一个token对视觉token的注意力分数来评估重要性，但**序列末尾的视觉token获得显著更高的注意力分数和保留频率**，即注意力评分存在严重的位置偏差。相比之下，Random和Pooling天然保持空间均匀分布。

为验证"空间均匀性优于位置偏差"这一假说，作者提出**Window FastV**：在FastV中引入滑动窗口机制，每个窗口内按固定比例选择token，确保保留token的空间均匀性。结果表明：
- 在75%裁剪率下，Window FastV比Vanilla FastV的平均性能下降少3.4个百分点
- 在88.9%裁剪率下，差距扩大到9个百分点
- 在RefCOCO空间定位任务上，空间均匀方法（Window FastV、Random、Pooling）显著优于非均匀方法（FastV、SparseVLM）

### 关键发现2：语言信息的作用依赖任务类型
将FastV改造为FastV_VIS（用最后一个视觉token替代最后一个文本token计算注意力），在Visual Haystack（视觉大海捞针）任务上测试：
- 去除文本引导后性能显著下降，确认了**强文本相关任务中语言信息至关重要**
- SparseVLM（文本引导）在77.8%压缩率下几乎保持了无压缩模型的准确率
- 但在常规VQA基准上，纯视觉剪枝方法（FasterVLM等）反而表现更好
- 结论：语言信息并非普遍有益，仅当任务与文本高度相关时才值得引入

### 关键发现3：重要性vs冗余性的$\alpha$困境
从信息论视角形式化token剪枝的两种准则：
- **冗余性准则**：最大化$I(\mathbf{X};\mathbf{X'})$，保持输入结构完整性（与信息瓶颈原理对应）
- **重要性准则**：最大化$I(\mathbf{X'};\mathbf{Y})$，保留对预测关键的token

引入参数$\alpha$平衡两者：$\text{Score}(x_i) = \alpha \cdot \text{预测关键性} + (1-\alpha) \cdot \text{模式唯一性}$

实验发现最优$\alpha$因任务而异：
- 感知型任务（MME、POPE）：$\alpha=0.0\sim0.1$最优，偏好冗余性优先
- 知识密集型任务（SQA、TextVQA）：$\alpha=0.8\sim0.9$最优，偏好重要性优先

### 关键发现4：FLOPs不能反映实际加速
在LLaVA-Next-7B上，SparseVLM与FastV的FLOPs仅差2.8%，但延迟高出26.8%。原因是SparseVLM在4层中执行剪枝（vs FastV仅1层），每层都需完整注意力图而无法使用Flash Attention。这揭示了**与硬件高效算子（Flash Attention）的兼容性**才是决定实际加速的关键因素。

### 关键发现5：训练感知压缩被忽视
Qwen2-VL在训练阶段已将4个相邻patch合并为1个token（TACR=4）。如果在评估token剪枝时考虑这一训练阶段压缩（即实际token缩减率 = TACR × 推理阶段缩减率），即使在88.9%的总缩减率下，性能也几乎与原始模型一致（99.6% vs 84.0%未考虑TACR时）。

## 实验关键数据

### 表1：LLaVA-1.5-7B上朴素方法vs精心设计方法（保留144 tokens，↓75%）

| 方法 | GQA | MMB | MME | POPE | SQA | TextVQA | 平均保留率 |
|------|-----|-----|-----|------|-----|---------|-----------|
| Vanilla（满token） | 61.9 | 64.7 | 1862 | 85.9 | 69.5 | 58.2 | 100% |
| Random | 59.0 | 62.2 | 1736 | 79.4 | 67.8 | 51.7 | 95.0% |
| Pooling | 59.1 | 62.5 | 1763 | 81.4 | 69.1 | 53.4 | 96.4% |
| Window FastV | 59.2 | 59.3 | 1737 | 80.3 | 66.4 | 50.8 | 93.2% |
| Vanilla FastV | 56.5 | 59.3 | 1689 | 71.8 | 65.3 | 53.6 | 89.8% |
| SparseVLM | 55.1 | 59.5 | 1711 | 77.6 | 69.3 | 54.9 | 93.5% |

Random和Pooling在约2/3的基准上超越FastV和SparseVLM。

### 表2：实际延迟vs FLOPs对比（LLaVA-Next-7B，保留320 tokens）

| 方法 | Token数↓ | 延迟 | FLOPs↓ | KV Cache↓ | POPE |
|------|---------|------|--------|-----------|------|
| Vanilla | 2880 | 36:16 | 100% | 1512 MB | 86.5 |
| FastV | 320 | 18:17 | 12.8% | 168 MB | 78.3 |
| SparseVLM | 320 | 23:11 | 15.6% | 168 MB | 82.3 |
| MustDrop | 320 | 23:40 | 11.5% | 168 MB | 82.1 |

FLOPs最低的MustDrop（11.5%）延迟反而最高，FastV FLOPs稍高但延迟最低。

### 表3：$\alpha$参数对不同任务的影响（LLaVA-1.5-7B，保留144 tokens）

| $\alpha$ | MME | POPE | SQA | TextVQA |
|----------|-----|------|-----|---------|
| 0.0（纯冗余） | 1707 | **82.8** | 64.8 | 53.6 |
| 0.1 | **1714** | 82.6 | 65.2 | 53.8 |
| 0.5 | 1711 | 81.6 | 65.3 | 54.3 |
| 0.8 | 1699 | 79.7 | 65.2 | 54.5 |
| 0.9 | 1680 | 75.6 | **65.7** | 54.2 |
| 1.0（纯重要性） | 1689 | 71.8 | 65.3 | 53.6 |

感知任务偏好低$\alpha$（冗余优先），知识任务偏好高$\alpha$（重要性优先）。

## 亮点与洞察

- **破除迷信**：用大量实验数据证明"精心设计不如随机选择"，迫使社区重新审视token剪枝的基本假设
- **位置偏差的发现**：揭示注意力评分的系统性偏差来源，Window FastV的简单修复即带来显著改善，说明问题根源在于评分机制而非剪枝框架
- **信息论框架**：用互信息和信息瓶颈原理统一了重要性和冗余性两种准则，$\alpha$参数实验证明没有万能配方，需要任务自适应
- **评估标准纠偏**：指出FLOPs与实际延迟的脱节（2.8% FLOPs差异→26.8%延迟差异），提醒社区关注Flash Attention兼容性
- **训练-推理联合视角**：Qwen2-VL的实验表明，训练阶段的压缩使得推理阶段剪枝几乎无损，暗示研究重心应转向训练感知压缩

## 局限性

- 实验主要基于LLaVA系列和Qwen2-VL，未覆盖InternVL、MiniCPM-V、Phi-Vision等更广泛架构
- 未在视频理解任务上验证结论（视频场景下token冗余模式可能不同）
- Window FastV仅作为验证假说的工具，未进一步优化为完整方法
- $\alpha$参数分析仅用FastV的注意力分数近似互信息，实际互信息计算更复杂
- 未探讨token剪枝与量化、蒸馏等其他压缩手段的组合效果
- 对训练感知压缩的分析仅限于Qwen2-VL的PatchMerger，缺少对其他训练压缩方案（如Q-Former）的详细对比

## 相关工作与启发

- **FastV (Chen et al., 2024)**：本文的主要分析对象，利用LLM第2层注意力分数剪枝，被证明存在严重位置偏差
- **SparseVLM (Zhang et al., 2024)**：引入文本引导的跨模态注意力选择，在文本相关任务（Visual Haystack）上表现好，但常规任务上不敌随机
- **MustDrop (Liu et al., 2024)**：全生命周期多阶段压缩，FLOPs最低但延迟反而最高，暴露了多层剪枝的硬件不友好问题
- **ToMe (Bolya et al., 2023)**：ViT中的token合并，空间相邻合并的思路与本文的空间均匀性发现一致
- **Qwen2-VL (Wang et al., 2024)**：训练阶段4:1 patch合并，为"训练感知压缩"提供了关键案例

**启发**：未来token剪枝应(1)保证保留token的空间均匀分布，(2)根据任务类型自适应平衡重要性和冗余性，(3)优先在浅层执行简单剪枝以兼容Flash Attention，(4)将研究重心转向训练阶段的感知压缩而非推理阶段的后处理。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 不提新方法但系统性揭示现有方法的根本缺陷，视角独到
- 实验充分度: ⭐⭐⭐⭐⭐ — 5个模型、9+数据集、多种剪枝率的全面对比，控制变量实验扎实
- 写作质量: ⭐⭐⭐⭐ — 以五个问题为线索层层展开，逻辑清晰，图表丰富
- 价值: ⭐⭐⭐⭐⭐ — 对MLLM推理加速社区有重要的纠偏意义，值得该领域每位研究者阅读

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models](unsolvable_problem_detection.md)
- [\[ICCV 2025\] Pi-GPS: Enhancing Geometry Problem Solving by Unleashing the Power of Diagrammatic Information](../../ICCV2025/multimodal_vlm/pi-gps_enhancing_geometry_problem_solving_by_unleashing_the_power_of_diagrammati.md)
- [\[ECCV 2024\] IVTP: Instruction-Guided Visual Token Pruning for Large Vision-Language Models](../../ECCV2024/multimodal_vlm/ivtp_instruction-guided_visual_token_pruning_for_large_vision-language_models.md)
- [\[ACL 2025\] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art](godbench_a_benchmark_for_multimodal_large_language_models_in_video_comment_art.md)
- [\[CVPR 2026\] TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model](../../CVPR2026/multimodal_vlm/transprune_token_transition_pruning_for_efficient_large_vision-language_model.md)

</div>

<!-- RELATED:END -->
