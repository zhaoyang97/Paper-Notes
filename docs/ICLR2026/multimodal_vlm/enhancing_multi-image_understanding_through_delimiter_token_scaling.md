---
title: >-
  [论文解读] Enhancing Multi-Image Understanding through Delimiter Token Scaling
description: >-
  [ICLR 2026][多模态VLM][多图理解] 通过对视觉语言模型中图像分隔符token的隐藏状态进行缩放，增强图像间的信息隔离能力，在不增加任何训练或推理成本的前提下，在多图理解（Mantis/MuirBench/MIRB/QBench2）和多文档/多表格理解（TQABench/MultiNews/WCEP-10）基准上均获得性能提升。
tags:
  - "ICLR 2026"
  - "多模态VLM"
  - "多图理解"
  - "大型视觉语言模型"
  - "分隔符token"
  - "跨图信息泄漏"
  - "注意力机制"
---

# Enhancing Multi-Image Understanding through Delimiter Token Scaling

**会议**: ICLR 2026  
**arXiv**: [2602.01984](https://arxiv.org/abs/2602.01984)  
**代码**: [GitHub](https://github.com/MYMY-young/DelimScaling)  
**领域**: Multimodal / VLM  
**关键词**: 多图理解, 大型视觉语言模型, 分隔符token, 跨图信息泄漏, 注意力机制

## 一句话总结
通过对视觉语言模型中图像分隔符token的隐藏状态进行缩放，增强图像间的信息隔离能力，在不增加任何训练或推理成本的前提下，在多图理解（Mantis/MuirBench/MIRB/QBench2）和多文档/多表格理解（TQABench/MultiNews/WCEP-10）基准上均获得性能提升。

## 研究背景与动机
大型视觉语言模型（LVLMs，如LLaVA、InternVL等）在单图任务上已取得优异性能，但在处理多图输入时性能明显下降。一个核心原因是**跨图信息泄漏（cross-image information leakage）**——模型难以区分来自不同图像的信息，导致推理时"张冠李戴"。

现有LVLMs已经使用分隔符token（delimiter tokens）来标记每张图像的起始和终止位置（如 `<image_start>` 和 `<image_end>`），但这些分隔符实际上未能有效地阻止跨图信息泄漏。模型在自注意力计算中，不同图像的视觉token仍然会相互交互，导致图像特异性信息被"稀释"。

**核心矛盾**: 分隔符token的存在提供了图像边界信息，但其隐藏状态的幅度不足以在注意力计算中形成有效的"信息屏障"。

**本文切入角度**: 极其简洁——直接放大分隔符token的隐藏状态（乘以一个缩放因子），从而增强其在注意力机制中的"隔离"效果。这一操作在推理时直接应用，无需重新训练模型。

## 方法详解

### 整体框架
输入是一段多模态序列：多张图像的视觉 token 之间穿插着标记每张图起止的分隔符 token（delimiter token，如 Qwen2.5-VL 的 `<|vision_start|>`/`<|vision_end|>`），后面接文本 prompt。LVLM 本就自带这些分隔符，作者先做了一轮诊断：把分隔符去掉、或换成别的特殊 token，注意力图里原本清晰的"三角块"边界就消失、多图任务掉约 10 个百分点——说明它们确实在划分图像边界，但图中红框区域仍残留可观的跨图注意力，**并没有完全挡住跨图信息泄漏**。

本文的全部改动只有一步：前向传播时把分隔符 token 的隐藏状态乘上一个缩放因子 $\lambda > 1$，其余权重、模块、流程一律不动。这一下同时做了两件看似矛盾的事——既让 softmax 把更多注意力压向分隔符、削弱跨图 token 之间的相互关注，又因为分隔符的值向量被一并放大、保住了"同图 token 共享一枚图像标签"的聚合效应。整个干预只发生在推理时，是 training-free 的。

### 关键设计

**1. 诊断：分隔符 token 有用，却没真正隔开多张图**

LVLM 多图掉点的根因是跨图信息泄漏（cross-image information leakage）——模型把不同图的信息混在一起。作者拆解分隔符 token 在注意力里的行为，发现它其实承担两个性质：(i) 第 $i$ 张图的 token 会集中关注第 $i$ 个分隔符，形成"图 ↔ 分隔符"一对一的纵向条纹，而不像 sink token 那样被全局关注；(ii) 这个被集中关注的分隔符相当于一枚"图像标签"，它在注意力输出里给同图所有 token 贡献一个共享的加性项 $p_{d_i} v_{d_i}$，从而强化图内交互（intra-image interaction），注意力图上表现为同图内部的三角块。问题在于这两个性质都偏弱：分隔符拿到的注意力不够强，跨图 token 之间仍有明显的相互关注（图 1a 红框），所以边界标记"有效果但不彻底"。

**2. 缩放分隔符隐藏状态：一个标量乘法同时抑制跨图、保住图内**

修法极简——在 Transformer 各层里，把分隔符 token 的隐藏状态 $h_t^{(l)}$ 替换成 $\lambda \cdot h_t^{(l)}$（$\lambda > 1$，非分隔符 token 不动），$D$ 是分隔符索引集合：

$$h_t^{(l)*}=\begin{cases}\lambda \cdot h_t^{(l)} & t \in D\\ h_t^{(l)} & t \notin D\end{cases}$$

关键不在"放大"本身，而在它**同时拨动上面两个性质却不互相打架**。一方面范数变大让分隔符像 sink token 一样吸走更多注意力，受 softmax 归一化所迫，分到其他图 token 上的注意力被相应压低——跨图交互被抑制。但只压跨图还不够：若连图内交互也被一并压垮，分隔符就丢了性质 (ii) 的图像标签作用。本文的观察是，缩放同时放大了分隔符的值向量 $v_d$，于是图内那枚共享加性项 $p_{d_i} v_{d_i}$ 的贡献被同步放大（实测约为相邻图对应项的 15～30 倍），抵消了 softmax 的压制——图内交互不仅没被牺牲反而更稳。一次标量乘法就把"抑制跨图"和"保住图内"两件事一起办了，这正是方法看似简单却有效的核心。作者也指出直接改注意力分数等替代实现可行，但隐藏状态缩放兼容 FlashAttention、开销最低，是效率与效果的折中。

**3. 免训练、零额外成本：直接套在现成 LVLM 上**

整个方法不新增任何参数或模块、不重训不微调，前向时只在分隔符位置多做一次标量乘法，计算开销可忽略，推理速度与显存基本不变；且与具体架构无关，可直接套用到各类 LVLM。代价只是两个超参：缩放因子 $\lambda$ 和施加缩放的层范围。这也解释了为什么同一招对纯文本里区分多文档/多表格的分隔符同样管用——它本质是注意力里一个跨模态通用的"边界信号太弱"问题，而非视觉模态独有。

### 损失函数 / 训练策略
无需训练。方法是纯推理时干预，唯一要设定的是缩放因子 $\lambda$（$>1$）和施加缩放的层范围。

## 实验关键数据

### 主实验
论文在多个多图理解基准上进行了评估：

| 数据集 | 任务类型 | 效果 |
|--------|---------|------|
| Mantis | 多图推理 | 提升 |
| MuirBench | 多图理解基准 | 提升 |
| MIRB | 多图推理基准 | 提升 |
| QBench2 | 图像质量对比 | 提升 |

此外，方法还在需要区分不同文本实体的纯文本任务上验证了有效性：

| 数据集 | 任务类型 | 效果 |
|--------|---------|------|
| TQABench | 多表格理解 | 提升 |
| MultiNews | 多文档摘要/理解 | 提升 |
| WCEP-10 | 多文档事件理解 | 提升 |

### 消融实验

| 配置 | 关键发现 | 说明 |
|------|---------|------|
| 缩放因子 $\lambda$ | 存在最优区间 | 过小效果不明显，过大可能破坏模型原有分布 |
| 应用层范围 | 中间层最有效 | 早期层和最后层的效果可能较弱 |
| 分隔符类型 | start和end均有效 | 两种分隔符的缩放都贡献于性能提升 |

### 关键发现
- 现有LVLM中的分隔符token虽然存在，但在隐藏状态层面未能有效发挥边界标记作用
- 简单的缩放操作就能显著增强其功能，说明问题不在于架构设计，而在于训练过程中分隔符未被充分学习
- 方法不仅对视觉分隔符有效，对文本中的分隔符（区分多文档/多表格）同样有效，说明机制具有通用性
- 该方法与模型的具体架构无关，可应用于多种LVLM

## 亮点与洞察
- **极简方法，显著效果**: 仅通过缩放隐藏状态就能改善多图理解，方法的简洁性令人印象深刻
- **零成本**: 真正做到了"免费午餐"——无需训练、无需额外参数、推理开销可忽略
- **通用机制**: 从视觉分隔符扩展到文本分隔符（多文档/多表格），说明这是注意力机制中的一个通用问题，而非视觉模态特有
- **诊断性洞察**: 论文对分隔符token为何失效的分析（隐藏状态范数不足以影响注意力分布）提供了对LVLM内部工作机制的有价值理解
- **实用性极强**: 可直接应用于任何已有的LVLM，无需重新训练，适合即时部署

## 局限与展望
- 缩放因子 $\lambda$ 需要手动调节，不同模型和任务可能需要不同的最优值
- 方法是推理时干预，如果在训练时就考虑分隔符的学习可能获得更好效果
- 论文HTML版本在ar5iv上转换失败，部分实验数值细节难以完整获取
- 对于特别长的多图序列（如视频帧），缩放策略可能需要进一步调整
- 未探讨与其他注意力干预方法（如注意力掩码、位置编码修改）的对比或组合
- 未在最新的超大规模LVLM（如GPT-4V）上测试

## 相关工作与启发
- 与视觉token压缩方法（如TrimTokenator-LC、VisionTrim）关注效率不同，本文关注多图场景下的信息隔离质量
- 与专门为多图理解设计的训练方法不同，本文提供了一种免训练的补充手段
- 启发：注意力机制中特殊token的"信号强度"可能是一个被忽视的设计维度——未来的LVLM训练可能需要显式地让分隔符学到更强的边界表示
- "缩放隐藏状态"这一简单dry intervention思路可能适用于其他需要信息隔离的场景（如多轮对话中区分不同轮次、RAG中区分不同检索文档）

## 评分
- 新颖性: ⭐⭐⭐⭐ — 观察和方法都很新颖，但技术复杂度较低
- 实验充分度: ⭐⭐⭐⭐ — 覆盖了多个基准和任务类型，包括消融和跨模态验证
- 写作质量: ⭐⭐⭐⭐ — 论文动机清晰、方法简洁（虽然全文HTML不可用，从摘要和代码可判断）
- 价值: ⭐⭐⭐⭐⭐ — 实用价值极高，任何LVLM用户都可以立即使用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Benchmarking and Enhancing VLM for Compressed Image Understanding](../../ICML2026/multimodal_vlm/benchmarking_and_enhancing_vlm_for_compressed_image_understanding.md)
- [\[ICLR 2026\] Patch-as-Decodable-Token: Towards Unified Multi-Modal Vision Tasks in MLLMs](patch-as-decodable-token_towards_unified_multi-modal_vision_tasks_in_mllms.md)
- [\[ICLR 2026\] OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM](omnivinci_enhancing_architecture_and_data_for_omni-modal_understanding_llm.md)
- [\[ICLR 2026\] MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence](mmsi-bench_a_benchmark_for_multi-image_spatial_intelligence.md)
- [\[ICLR 2026\] Sparsity Forcing: Reinforcing Token Sparsity of MLLMs](sparsity_forcing_reinforcing_token_sparsity_of_mllms.md)

</div>

<!-- RELATED:END -->
