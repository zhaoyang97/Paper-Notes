---
title: >-
  [论文解读] VLsI: Verbalized Layers-to-Interactions from Large to Small Vision Language Models
description: >-
  [CVPR 2025][多模态VLM][知识蒸馏] VLsI 提出了一种基于自然语言的层间蒸馏方法，通过在大小 VLM 的中间层引入 "verbalizer" 将特征映射到语言空间，并采用自适应层匹配策略对齐推理过程，使 2B/7B 小模型在 10 个 VL 基准上平均超过 GPT-4V 达 11.0%/17.4%，无需改变架构或增加参数。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "知识蒸馏"
  - "层间对齐"
  - "自然语言蒸馏"
  - "小模型VLM"
  - "高效推理"
---

# VLsI: Verbalized Layers-to-Interactions from Large to Small Vision Language Models

**会议**: CVPR 2025  
**arXiv**: [2412.01822](https://arxiv.org/abs/2412.01822)  
**代码**: [https://github.com/byungkwanlee/VLsI](https://github.com/byungkwanlee/VLsI)  
**领域**: 多模态VLM  
**关键词**: 知识蒸馏, 层间对齐, 自然语言蒸馏, 小模型VLM, 高效推理

## 一句话总结
VLsI 提出了一种基于自然语言的层间蒸馏方法，通过在大小 VLM 的中间层引入 "verbalizer" 将特征映射到语言空间，并采用自适应层匹配策略对齐推理过程，使 2B/7B 小模型在 10 个 VL 基准上平均超过 GPT-4V 达 11.0%/17.4%，无需改变架构或增加参数。

## 研究背景与动机
开源 VLM（如 Qwen2-VL、LLaVA-OneVision）通过扩大模型规模不断提升性能，但大模型的计算开销使其难以部署到移动设备和机器人等资源受限平台。现有解决方案包括添加额外模块（如多视觉编码器融合）或修改架构（如 TroL 的双重前向传播），但这些方法引入工程复杂度和兼容性问题。传统知识蒸馏方法通常直接模仿最终输出，容易导致训练不稳定，且忽略了大小模型在中间层的推理过程差异。**核心 idea**：既然自然语言是人类之间传递知识的高效媒介，能否也用自然语言作为大小 VLM 之间知识传递的桥梁？通过将中间层特征"语言化"，实现层间推理轨迹的对齐。

## 方法详解

### 整体框架
VLsI 的训练分三步走：(1) Verbalization Step：在大小 VLM 的中间目标层各自训练一个 verbalizer，将该层的隐藏状态映射到自然语言空间；(2) Interaction Step：通过自适应层匹配找到大小 VLM 之间的最优层对应关系，用 KL 散度蒸馏小模型使其模仿大模型各层的推理进展；(3) SFT Step：对蒸馏后的小模型做全量参数微调，通过自回归损失进一步优化指令跟随能力。

### 关键设计
1. **Verbalizer（语言化器）**:
    - 功能：将 Transformer 中间层的隐藏状态投影到自然语言空间，使其可被解读为文本响应
    - 核心思路：每个目标中间层配备一个 verbalizer，由一个精简版 FFN（verb-FFN，不做维度扩展/压缩）+ backbone VLM 的 language head 组成。在冻结 backbone 参数的条件下，用自回归损失训练 verb-FFN，使中间层输出经过 verbalizer 后能产生与目标响应对齐的文本。由于 backbone 权重冻结，各层 verbalizer 的梯度更新相互独立
    - 设计动机：借鉴 speculative decoding 的思路——小 LLM 可以通过复用大 LLM 的 word embedding 和 language head 来模仿大模型。Verbalization 使得我们可以在自然语言空间中追踪"关键推理进展"，而非在难以解释的特征空间中对齐

2. **自适应层匹配 (Adaptive Layer Matching)**:
    - 功能：解决大小 VLM 层数不同时的层间对应问题
    - 核心思路：用多项式采样策略动态匹配层。对于小模型的第 $i_s$ 层，在大模型的搜索范围内遍历所有候选层，计算各候选层与小模型当前层的 KL 散度，然后用 softmax(-KLD/T) 生成采样分布进行采样。关键约束有两条：**顺序保持**——小模型第 $i$ 层匹配到大模型第 $j$ 层后，第 $i+1$ 层必须匹配到 $j+1$ 之后的层；**搜索范围限制**——避免大模型的早期层与小模型的晚期层匹配
    - 设计动机：固定均匀映射忽略了不同层"推理进展"不同的事实。反 KL 散度采样使匹配偏向对齐较好的层对，自适应温度使之随训练动态调整

3. **三步训练策略**:
    - 功能：分阶段最大化知识迁移效果
    - 核心思路：Verbalization 阶段为大小模型各自建立"层到语言"的映射（独立训练）；Interaction 阶段用 KL 散度损失对齐层间分布，此时只更新小模型的 verbalizer 和 LoRA 参数；SFT 阶段解冻全部参数做自回归微调
    - 设计动机：如果在 Interaction 阶段同时优化 KL 散度和自回归损失，会导致性能下降（消融实验验证）。SFT 阶段类似于剪枝后的 recovery training，帮助小模型充分吸收蒸馏知识

### 损失函数 / 训练策略
- Verbalization：自回归损失 $\mathcal{L}_{AR}$ 独立训练每层 verbalizer
- Interaction：各层 KL 散度损失之和（包括中间层和最后一层）
- SFT：标准自回归损失对小模型全量微调
- 训练配置：8×A100 80GB，LoRA rank=64，AdamW + cosine schedule，2.9M 多样化视觉指令数据

## 实验关键数据

### 主实验（7B 模型对比）

| Benchmark | Qwen2-VL-7B | VLsI-7B | GPT-4V | 提升 vs GPT-4V |
|-----------|------------|---------|--------|---------------|
| MM-Vet | 62.0 | **75.2** | 67.5 | +7.7 |
| MMMU | 54.1 | **69.3** | 61.7 | +7.6 |
| MathVista | 58.2 | **74.7** | 54.7 | +20.0 |
| MMB | 83.0 | **86.3** | - | - |
| AI2D | 77.5 | **87.3** | 78.6 | +8.7 |

### 小模型对比（2B）

| Benchmark | Qwen2-VL-2B | VLsI-2B | 提升 |
|-----------|------------|---------|------|
| MM-Vet | 49.5 | **64.8** | +15.3 |
| MMMU | 41.1 | **51.4** | +10.3 |
| MathVista | 43.0 | **68.4** | +25.4 |
| AI2D | 60.2 | **89.0** | +28.8 |

### 消融实验

| 配置 | MMB | MM-Vet | MMMU | 说明 |
|------|-----|--------|------|------|
| CE (Interaction) + 无 Last Layer | 79.2 | 64.5 | 56.5 | 用交叉熵做中间层蒸馏 |
| KLD + 无 Last Layer | 83.0 | 69.5 | 61.0 | KL散度+无末层蒸馏 |
| KLD + KLD (完整) | **86.3** | **75.8** | **69.3** | 中间层+末层都用 KLD |
| 无 SFT | 82.1 | 60.5 | 52.9 | 跳过 SFT 阶段 |
| 有 SFT | **86.3** | **75.8** | **69.3** | SFT 带来显著提升 |

### 关键发现
- VLsI-7B 在 MM-Vet、MMMU、MathVista 等困难基准上**超过 GPT-4V 达 17.4%**
- VLsI-2B 甚至超过许多 7B-13B 模型（如 LLaVA-NeXT-13B、Eagle-13B）
- KL 散度比交叉熵更适合层间蒸馏；中间层和末层同时蒸馏效果最好
- 自适应层匹配中各组件（顺序保持、搜索范围、自适应温度）均有贡献
- verbalizer 结构：verb-FFN（269M 参数）在性能/效率上达到最优平衡

## 亮点与洞察
- **自然语言作为蒸馏媒介**是非常优雅的 idea——让特征空间对齐问题转化为语言空间的分布匹配，更直观且有效
- 通过 verbalization 可以可视化每一层的推理进展（如 Fig. 3），具有可解释性价值
- 不修改模型架构、不增加推理成本，蒸馏完成后可直接使用小模型
- VLsI-0.5B 在 LLaVA-OV-72B 指导下也能达到不错效果（MMB=72.5），展示了极端规模差距下蒸馏的可行性

## 局限与展望
- 大小模型必须共享同一 tokenizer 和词表索引，限制了跨模型族的迁移（如 Qwen 系列内可以但不能跨到 LLaMA）
- 训练需要同时加载大小两个模型，显存开销较大（72B teacher + 7B student）
- verbalization 阶段为每个目标层独立训练，层数多时训练成本增加
- 实验中 visual instruction 数据量较大（2.9M），数据效率有待探索
- 目前只在 Qwen2-VL 和 LLaVA-OV 两个模型族上验证，更广泛的架构支持需要进一步研究

## 相关工作与启发
- **vs LLaVA-KD / LLaVA-MoD**: 这些方法只做最终层蒸馏，忽略了层间推理过程的对齐
- **vs TroL / Phantom**: 通过修改架构（双重前向/扩展隐维度）增强小模型，但引入 KV-cache 和兼容性问题
- **vs Align-KD / MoVE-KD**: 在特征空间做对齐，而 VLsI 在语言空间做对齐，更自然且有效
- **vs DistiLLM / MiniLLM**: 关注蒸馏中 KL 散度方向的选择，VLsI 则关注层间对齐这一正交维度

## 补充说明
- VLsI-7B 在 LLaVA-Wilder 上达到 92.0 分，超过 GPT-4o 的 85.9 和 Qwen2-VL-72B 的 84.1
- 中间目标层选择：小模型每隔 4 层选一个（第 2/6/10/.../26 层），大模型同样间隔选取

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用自然语言做层间蒸馏是非常新颖的范式，verbalizer + 自适应层匹配设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 10个基准、大量消融、跨backbone验证、与闭源模型对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，verbalization 可视化效果好
- 价值: ⭐⭐⭐⭐⭐ 为高效 VLM 提供了新的蒸馏范式，2B 模型即可超过许多 7B-13B 模型

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Towards Understanding How Knowledge Evolves in Large Vision-Language Models](towards_understanding_how_knowledge_evolves_in_large_vision-language_models.md)
- [\[CVPR 2025\] Calico: Part-Focused Semantic Co-Segmentation with Large Vision-Language Models](calico_part-focused_semantic_co-segmentation_with_large_vision-language_models.md)
- [\[CVPR 2025\] Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves?](can_large_vision-language_models_correct_semantic_grounding_errors_by_themselves.md)
- [\[CVPR 2025\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[CVPR 2025\] Synthetic Data is an Elegant GIFT for Continual Vision-Language Models](synthetic_data_is_an_elegant_gift_for_continual_vision-language_models.md)

</div>

<!-- RELATED:END -->
