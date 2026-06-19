---
title: >-
  [论文解读] MoSE: Hierarchical Self-Distillation Enhances Early Layer Embeddings
description: >-
  [AAAI 2026][代码智能][自蒸馏] 提出 ModularStarEncoder（MoSE），一个 10 亿参数的多出口编码器，通过新颖的自蒸馏机制（高层引导低层训练）显著增强早期层表示，在 CodeSearchNet 等代码理解任务上超越所有开源模型，同时支持灵活的计算-精度权衡部署。 领域现状 大语言模型在 NL…
tags:
  - "AAAI 2026"
  - "代码智能"
  - "自蒸馏"
  - "多出口网络"
  - "代码检索"
  - "早期退出"
  - "模块化部署"
---

# MoSE: Hierarchical Self-Distillation Enhances Early Layer Embeddings

**会议**: AAAI 2026  
**arXiv**: [2503.03008](https://arxiv.org/abs/2503.03008)  
**代码**: [HuggingFace](https://huggingface.co/modularStarEncoder)  
**领域**: 代码智能  
**关键词**: 自蒸馏, 多出口网络, 代码检索, 早期退出, 模块化部署

## 一句话总结
提出 ModularStarEncoder（MoSE），一个 10 亿参数的多出口编码器，通过新颖的自蒸馏机制（高层引导低层训练）显著增强早期层表示，在 CodeSearchNet 等代码理解任务上超越所有开源模型，同时支持灵活的计算-精度权衡部署。

## 研究背景与动机

### 领域现状
大语言模型在 NLP 领域取得了显著进展，但其巨大的计算需求给部署带来了严峻挑战。学界已提出多种策略：量化减少数值精度、知识蒸馏训练小模型、剪枝去除低影响参数。LLaMA、Qwen、Mistral 等系列代表了向更高效架构的转变。

### 现有痛点

**传统蒸馏成本高**：需要分别训练教师和学生模型，成本随模型数量线性增加

**固定推理成本**：标准模型必须经过所有层才能输出，无法根据任务难度调整计算量

**中间层价值未被利用**：SimCLR 和 Valeriani 等研究表明中间层而非最终层往往持有最语义丰富的表示

**NSP 损失的局限**：传统 Next Sentence Prediction 损失微调后几乎无益，且对代码输入的长上下文利用效率低

### 核心矛盾
如何在单一模型中同时训练多个不同计算量的子模型，并确保早期层也能产生高质量表示？

### 切入角度
在单个 Transformer 中设置多个出口点，通过多层损失让高层的训练信号传播到低层（自蒸馏），无需额外的教师模型。同时用 In-Context Classification（ICC）损失替代 NSP，提高上下文窗口利用率。

## 方法详解

### 整体框架
基于 StarCoder-2 架构的 1B 参数双向编码器，36 层 → 在第 4、9、18、27、36 层设置出口头 → 每个出口同时计算 MLM 和 ICC 损失 → 加权求和（自蒸馏）→ 推理时用户可选择任意出口点（最小 160M 参数）。

### 关键设计

1. **自蒸馏（Self-Distillation）多层损失**:

    - 在选定层 $\iota = \{4, 9, 18, 27, 36\}$ 上分别计算损失
    - 各层共享分类头，但加入层索引的位置编码以区分
    - 加权系数 $\alpha = i/|I|$，其中 $I = \{1, ..., 36\}$，越深的层权重越大
    - 总损失：$\mathcal{L} = \sum_{i \in \iota} \mathcal{L}_i \cdot \alpha$
    - 效果：高层的训练信号自然传播到共享的低层参数，促使低层学到更好的表示
    - 设计动机：在一次训练中得到多个性能各异的模型，避免多次蒸馏的冗余成本

2. **In-Context Classification（ICC）损失**:

    - 替代传统的 Next Sentence Prediction
    - 随机拼接代码片段（用 [SEP] 分隔），50% 概率来自不同仓库
    - 分类：拼接的输入是否来自同一仓库
    - 优势：
        - 提高输入密度：平均输入长度从 630 token 增加到 1,300 token
        - 仓库天然模块化，包含多语言文件，有助于跨语言理解
    - 总预训练目标：$\mathcal{L} = \mathcal{L}_{MLM} + \mathcal{L}_{ICC}$
    - 设计动机：NSP 对微调后几乎无益，且浪费上下文窗口

3. **架构改进**:

    - 基于 StarCoder-2，36 隐藏层，1B 参数
    - GQA（16 注意力头，4 KV 头）+ RoPE（θ=10⁶）
    - 隐藏维度 1024，中间维度 12288
    - 关键修改：
        - 去除因果掩码 → 双向注意力
        - 滑动窗口注意力 → 全注意力（避免感受野限制，确保模块化）
        - 集成 FlashAttention V2
    - 上下文长度 2048 token

4. **SynthCoNL 数据集**:

    - 基于 CodeSearchNet 种子数据集
    - 使用 Qwen2.5-Coder-7B-Instruct 进行代码翻译
    - 生成 1,071,367 个（自然语言, 代码A, 代码B）三元组
    - 代码B 横跨 Go, Ruby, JS, Python, C++, PHP, C, Java
    - 近去重：LSH + Jaccard 相似度阈值 0.7 + 256 排列 + 字符级 5-gram
    - 设计动机：扩展文本-代码基准，增加跨语言代码-代码检索能力

### 损失函数 / 训练策略

**预训练**：
- 批次大小 4M tokens，最大上下文 2048 tokens
- 245,000 步，处理约 1T tokens（TheStackV2 数据集）
- AdamW optimizer（β₁=0.9, β₂=0.95, ε=1e-6, weight decay=0.1）
- 多步学习率调度器：4000 warmup 步，在 120K/185K/220K/230K/240K 步阶梯衰减
- 512 × NVIDIA Ampere (64GB) GPU，约 450,000 GPU 小时

**微调（检索）**：
- CLIP-style 损失 + 多层方法
- 5 个不同的投影头（对应 5 个出口点）
- batch 2048，文本-代码和代码-代码均匀分布
- 数据增强：30% 概率随机替换代码中的高频词
- lr=1e-5，温度参数 10.0

**微调（分类 - BigCloneBench）**：
- 输入格式：[SEP] snippet-1 [SEP] snippet-2 [CLS]
- 用最终 [CLS] token 做分类
- lr=1e-5，2000 warmup 步，batch=64，14000 步

## 实验关键数据

### 主实验

| 模型 | Ruby | JS | Go | Python | Java | PHP | avg MRR | avg NDCG | POJ104 mAP |
|------|------|----|----|--------|------|-----|---------|----------|------------|
| **MoSE** | 74.1 | 74.0 | 82.5 | **92.5** | 78.7 | 84.5 | **81.0** | **84.2** | **75.9** |
| CodeT5+ | 78.0 | 71.3 | 92.7 | 75.8 | 76.2 | 70.1 | 77.4 | - | 24.5 |
| UniXcoder | 74.0 | 68.4 | 91.5 | 72.0 | 72.6 | 67.6 | 74.4 | - | 41.0 |
| ModernBERT-large | - | - | - | - | - | - | - | 59.5 | 27.3 |
| OpenAI Embedding | 84.7 | 85.3 | 95.9 | 99.8 | 90.1 | 95.6 | 91.9 | 93.3 | 82.9 |

| 模型 | BigCloneBench F1 | 备注 |
|------|-----------------|------|
| MoSE (L4) | 93.0 | 最浅出口 |
| MoSE (L9) | 93.4 | - |
| MoSE (L18) | **94.2** | 最佳层 |
| MoSE (L27) | 94.1 | - |
| MoSE (L36) | 94.1 | - |
| CodeT5+ (770M) | 95.1 | 仅高 0.9 |
| UniXcoder | 95.2 | - |

### 消融实验

| 配置 | avg Recall@1 增益 | 说明 |
|------|------------------|------|
| 单出口基线 L4 | 基线 | 仅在第 4 层训练 |
| 单出口基线 L9 | 基线 | 仅在第 9 层训练 |
| Self-Distilled L9 | +4.36% | 多层损失带来最大提升 |
| Self-Distilled L18-36 | 稳定高性能 | 从第 18 层起性能差距小 |
| ICC vs NSP (CoIR CSN-CCR) | 10.1 vs 6.2 | ICC 显著更好 |
| ICC vs NSP (CoIR CT-DL) | 32.0 vs 31.0 | ICC 稍优或持平 |

### 关键发现

1. **开源模型 SOTA**：MoSE 在 CodeSearchNet 上 MRR=81.0，超越 CodeT5+ 3.6%，大幅缩小与 OpenAI 闭源模型的差距
2. **90% 计算量减少，仅 6.4% 性能损失**：从第 36 层到第 4 层，FLOPs 减少约 90%，但文本-代码检索 MRR 仅下降 6.4%
3. **任务相关的最优层**：文本-代码检索最优层在第 18 层，代码-代码检索最优层在浅层，印证了语义信息在不同层的不均匀分布
4. **自蒸馏 vs 单出口**：多层损失在所有层上持续优于单出口基线，第 9 层提升最大（+4.36% Recall@1）
5. **ICC > NSP**：在跨上下文检索任务上 ICC 显著优于 NSP，同时提供更密集的表示
6. **跨架构泛化**：在 AlexNet、VGG11、ResNet18 上均表现良好
7. **置换检验**：不同层产生显著不同的相似度评分（p < 0.001），证明模型各层确实学到了不同的表示

## 亮点与洞察
1. 自蒸馏思路优雅：无需教师模型，高层自然引导低层，一次训练得到多个模型变体
2. ICC 损失设计巧妙：既提高了上下文利用率（630→1300 token），又利用仓库级信息提升跨语言理解
3. 发现了任务类型与最优出口层之间的关系：跨模态任务需要更深层，同模态任务浅层即可
4. SynthCoNL 数据集的构建方法可推广：利用代码翻译模型低成本生成跨语言训练对
5. 模块化设计给用户极大的灵活性：从 160M 到 1B 参数按需选择

## 局限与展望
1. 计算资源限制了超参调优和预训练消融实验的充分性
2. 使用合成代码翻译数据微调的影响尚不明确
3. 与 OpenAI Embedding 差距仍大（81.0 vs 91.9 MRR），闭源模型规模和训练数据不透明
4. 出口点选择（4,9,18,27,36）为经验设定，可探索更优的出口配置
5. 可研究不同出口点的组合使用和任务类型与深度的更系统关系
6. 仅在代码理解任务上验证，可扩展到自然语言任务

## 相关工作与启发
- **BranchyNet**：早期退出架构 → 本文在大规模 Transformer 上实现
- **Matryoshka 表示学习**：嵌入维度裁剪 → 本文裁剪层深度
- **SimCLR / Valeriani**：最终层不一定最优 → 本文的核心假设
- **CodeT5+**：多任务代码预训练 → 本文用自蒸馏+ICC 替代
- **知识蒸馏（Sanh 2019）**：教师-学生范式 → 本文消除对教师模型的依赖

## 评分
- 新颖性: ⭐⭐⭐⭐ (自蒸馏+ICC 的组合在代码编码器上是首创)
- 实验充分度: ⭐⭐⭐⭐⭐ (CodeSearchNet+POJ104+BigCloneBench+CoIR，消融充分)
- 写作质量: ⭐⭐⭐⭐⭐ (结构清晰，图表直观，论证严密)
- 价值: ⭐⭐⭐⭐⭐ (实际部署价值高，开源模型+数据集贡献大)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Behavioral Embeddings of Programs: A Quasi-Dynamic Approach for Optimization Prediction](../../ICLR2026/code_intelligence/behavioral_embeddings_of_programs_a_quasi-dynamic_approach_for_optimization_pred.md)
- [\[ACL 2026\] DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation](../../ACL2026/code_intelligence/deepguard_secure_code_generation_via_multi-layer_semantic_aggregation.md)
- [\[ACL 2025\] Revisit Self-Debugging with Self-Generated Tests for Code Generation](../../ACL2025/code_intelligence/revisit_self-debugging_with_self-generated_tests_for_code_generation.md)
- [\[ACL 2026\] EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents](../../ACL2026/code_intelligence/eet_experience-driven_early_termination_for_cost-efficient_software_engineering_.md)
- [\[ACL 2025\] SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL](../../ACL2025/code_intelligence/share_text_to_sql_correction.md)

</div>

<!-- RELATED:END -->
