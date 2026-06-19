---
title: >-
  [论文解读] ELMO: Efficiency via Low-precision and Peak Memory Optimization in Large Output Spaces
description: >-
  [ICML 2025][推荐系统][极端多标签分类] 提出 ELMO 框架，通过纯 BFloat16/Float8 低精度训练结合梯度融合、分块策略等峰值显存优化，将 300 万标签的 XMC 模型训练显存从 39.7 GiB 降至 6.6 GiB，且不损失分类精度。 极端多标签分类（Extreme Multilabel C…
tags:
  - "ICML 2025"
  - "推荐系统"
  - "极端多标签分类"
  - "低精度训练"
  - "峰值显存优化"
  - "Float8"
  - "大规模输出空间"
---

# ELMO: Efficiency via Low-precision and Peak Memory Optimization in Large Output Spaces

**会议**: ICML 2025  
**arXiv**: [2510.11168](https://arxiv.org/abs/2510.11168)  
**代码**: [xmc-aalto/elmo](https://github.com/xmc-aalto/elmo)  
**领域**: 推荐系统  
**关键词**: 极端多标签分类, 低精度训练, 峰值显存优化, Float8, 大规模输出空间

## 一句话总结

提出 ELMO 框架，通过纯 BFloat16/Float8 低精度训练结合梯度融合、分块策略等峰值显存优化，将 300 万标签的 XMC 模型训练显存从 39.7 GiB 降至 6.6 GiB，且不损失分类精度。

## 研究背景与动机

极端多标签分类（Extreme Multilabel Classification, XMC）广泛应用于大规模标签推荐、商品推荐、维基百科标签等场景，其标签空间可达数十万到数百万级别。在这种设定下，线性分类头（而非编码器）成为计算和显存的主要瓶颈——embedding 维度 768、300 万标签时，仅分类器权重就需 ~8 GiB，加上梯度和优化器状态则膨胀到 ~32 GiB。

当前 SOTA 方法 Renee 虽然通过跳过显式 loss 计算避免了部分中间变量的显存占用，但仍存在三大问题：

**混合精度低效**：FP16-FP32 混合精度训练需要维护全精度参数副本，且要求将分类器梯度上转为 FP32，显存浪费严重

**峰值显存堆积**：计算图的执行顺序导致显存密集操作在同一时间点叠加，峰值显存飙升至 39.7 GiB

**分类器权重无压缩**：Renee 和其他标签短列表方法均未对分类层权重进行显存压缩

此外，FP16 在 XMC 场景中存在固有不稳定性：大标签空间导致梯度累加时上溢，同时小梯度又面临下溢。这种上溢与下溢并存的特性使得标准 loss scaling 难以同时应对。

## 方法详解

### 整体框架

ELMO 的核心思路是将 XMC 模型训练从 FP16-FP32 混合精度转向纯低精度（BF16 或 FP8），并配合一系列峰值显存优化策略。整体方案分为三个递进层次：

1. **纯 16-bit 训练**（BF16）：消除多精度副本的冗余显存
2. **架构级显存优化**：通过计算流重组、分块策略、梯度融合降低峰值显存
3. **8-bit 训练**（FP8 E4M3）：进一步将分类器权重和编码器压缩到 FP8

### 关键设计

#### 1. 纯 BF16 训练 + 精度补偿

直接将权重从 FP32 降为 BF16 会导致优化器更新被取消（round-to-nearest 会把小于相邻可表示数距离一半的更新抹掉）。ELMO 对此采用两种互补策略：

- **分类器权重 → 随机舍入（Stochastic Rounding）**：显存效率优先，随机舍入是无偏估计，能防止小梯度累加时的灾难性舍入误差
- **编码器权重 → Kahan 求和**：通过额外的补偿项 $c$ 跟踪舍入误差，在后续加法中修正：

$$
y \leftarrow v - c, \quad c \leftarrow ((s+y) - s) - y, \quad s \leftarrow s + y
$$

虽然 Kahan 求和需要额外的补偿缓冲区，但省去了高精度权重的额外拷贝。编码器参数量小，这点开销可忽略。

#### 2. 去除分类器动量

实验表明分类器层不需要动量缓冲区，直接使用大学习率的纯 SGD 即可。这一步直接省掉了 ~8 GiB（300 万标签时）的动量存储。

#### 3. 计算流重组 + 分块策略（Chunking）

Renee 的显存峰值源于分类器的前向输出、梯度和权重副本同时驻留显存。ELMO 的解决方案：

- **解耦编码器和分类器更新**：先完成编码器前向传播，再对分类器分块处理
- **标签分块**：将标签空间分为 $k$ 个等大的块（实验中 $k=3 \sim 8$），依次完成每个块的前向、反向和参数更新，将瞬态显存需求降低 $k$ 倍
- 分块对训练延迟无明显影响

#### 4. FP8 分类器训练

通过对不同指数位和尾数位组合的 sweep 实验发现：

- **权重**：3 位指数即够（E4M3 的动态范围足够），尾数低于 6 位时性能下降，但随机舍入可完全恢复
- **梯度**：E5M2 下仍有 ~20% 梯度为零，必须保持 BF16
- **输入**：FP8 E4M3 即可覆盖分类器输入范围

具体流程：将 BF16 的分类器输入 cast 为 FP8 E4M3 → 与 FP8 权重矩阵乘得到 BF16 logits → 在 BF16 下计算梯度。

#### 5. 梯度融合 Triton 内核

ELMO 将梯度计算与 SGD 更新融合进单个 Triton 内核，全程在 SRAM 中执行：

1. 从 HBM 加载分类器权重、logits、输入到 SRAM
2. 在 SRAM 中完成 matmul 计算梯度
3. 直接在 SRAM 中用 SGD + 随机舍入更新权重
4. 写回 HBM

这彻底消除了分类器梯度在 GPU 显存（HBM）中的存储需求。

#### 6. FP8 编码器（torchao）

集成 torchao 框架对 Transformer 编码器进行 FP8 训练，进一步降低激活显存（从 BF16 的 4.6 GiB 降至 3 GiB）。

### 损失函数 / 训练策略

- 损失函数沿用 Renee 的设计（BCE loss），不显式计算 loss 值，仅需 logit 梯度
- 分类器使用大学习率纯 SGD（无动量），编码器使用 AdamW + Kahan 求和
- 分类器权重使用随机舍入（BF16/FP8），编码器权重使用 Kahan 求和补偿（BF16）
- 不使用任何 tensor scaling 技术

## 实验关键数据

### 主实验

在 Wiki-500K、AmazonTitles-670K、Amazon-670K、Amazon-3M 等标准 XMC 数据集上与多种 SOTA 方法对比：

| 数据集 | 方法 | P@1 | P@3 | P@5 | 峰值显存 (GiB) |
|--------|------|-----|-----|-----|---------------|
| Wiki-500K | LightXML | 76.19 | 57.22 | 44.12 | 15.72 |
| Wiki-500K | CascadeXML | 77.0 | 58.3 | 45.1 | 18.8 |
| AmazonTitles-670K | LightXML | 41.7 | 37.3 | 34.2 | 13.99 |
| AmazonTitles-670K | CascadeXML | 42.1 | 37.5 | 34.1 | 22.3 |
| Amazon-3M | Renee | — | — | — | 39.7 |
| Amazon-3M | ELMO-BF16 | 相当 | 相当 | 相当 | 10.3 |
| Amazon-3M | ELMO-FP8 | 相当 | 相当 | 相当 | **6.6** |

### 显存对比（3M 标签，batch=128，BERT-base，embed=768）

| 配置 | 初始化显存 | 峰值显存 | 相对 Renee 降幅 |
|------|-----------|---------|----------------|
| Renee (FP16-FP32混合) | 17.9 GiB | 39.7 GiB | — |
| ELMO-BF16 | 5.2 GiB | 10.3 GiB | **74%** |
| ELMO-FP8 | 3.2 GiB | 6.6 GiB | **83%** |

### 消融实验

| 配置 | 关键指标 / 效果 | 说明 |
|------|---------------|------|
| 去除动量 | 显存减 ~8 GiB，精度不变 | 分类器不需要 momentum |
| 随机舍入 vs 最近舍入 | 随机舍入恢复 FP32 精度 | 在尾数 < 6 bit 时差异显著 |
| 分块数 k=3~8 | 训练延迟无明显影响 | 显存线性降低 |
| 梯度融合 Triton kernel | 梯度显存→≈0 | 全 SRAM 执行 |
| E4M3 vs E5M2 权重 | E4M3 足够 | 3 位指数即可覆盖权重范围 |
| BF16 vs FP8 梯度 | 必须 BF16 | FP8 E5M2 下 20% 梯度为零 |
| torchao FP8 编码器 | 激活 4.6→3 GiB | 额外引入 0.5 GiB buffer |

### 关键发现

1. **FP16 混合精度在 XMC 中是不稳定的**：大标签空间导致梯度累加时上溢，同时小梯度又面临下溢，loss scaling 无法兼顾
2. **BF16 的扩展动态范围完美解决上溢问题**，配合随机舍入/Kahan 求和可弥补精度损失
3. **FP8 训练不需要 tensor scaling**：E4M3 的原生动态范围足以覆盖分类器权重和输入
4. **分类器梯度必须保持 BF16**：FP8 下即使用 E5M2 格式仍有 ~20% 梯度归零
5. **峰值显存优化的关键在于计算流重组**，而非单纯的精度降低

## 亮点与洞察

- **问题导向的系统设计**：从 memory profiling 出发，逐层分析显存瓶颈，针对性地设计解决方案，是系统优化的典范
- **Triton 梯度融合内核**：将梯度计算和参数更新全部在 SRAM 中完成，彻底消除梯度存储，是非常优雅的工程创新
- **分精度策略**：不是一刀切地用同一精度，而是为权重（FP8）、梯度（BF16）、编码器（Kahan BF16）、分类器（SR BF16/FP8）各选最佳精度
- **8.6M 标签新数据集**：LF-Paper2Keywords-8.6M 是目前最大的公开 XMC 基准，有助于推动领域发展
- **6x 显存降低且精度不损**：6.6 GiB vs 39.7 GiB，意味着从 A100-80G 降级到消费级 GPU 即可训练百万标签模型

## 局限与展望

1. **仅验证了 BERT-base 编码器**：更大的编码器（如 BERT-large、DeBERTa-v3）下的表现未知
2. **FP8 训练依赖 Hopper/Ada/Blackwell GPU**：E4M3 硬件加速仅限较新的 GPU 架构
3. **分块策略可能限制并行度**：标签分块是串行处理的，在极端大标签空间下可能成为时间瓶颈
4. **未探索稀疏训练的结合**：与动态稀疏训练（如 ELIAS）结合可能进一步降低显存
5. **Triton 内核的可移植性**：自定义 Triton 内核增加了维护成本和迁移难度

## 相关工作与启发

- **Renee**（Jain et al., 2023）：全 end-to-end XMC 训练的基石，ELMO 在此基础上做显存优化
- **DEXML**（Gupta et al., 2024）：通过双编码器消除分类器，但代价是更大的计算和显存开销
- **torchao**：通用 FP8 训练框架，ELMO 将其应用于 XMC 编码器
- **启发**：低精度训练 + 自定义融合内核的思路可推广到其他大输出空间任务（如语言模型的大词表 softmax）

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 首次实现 XMC 纯 FP8 训练，Triton 梯度融合很巧妙 |
| 技术深度 | 5 | 从浮点表示到内核编程，涵盖多层次技术栈 |
| 实验充分度 | 4 | 多数据集覆盖 + 消融 + 新基准，但大编码器实验缺失 |
| 实用价值 | 5 | 6x 显存降低直接降低硬件门槛，代码开源 |
| 写作质量 | 4 | 逻辑清晰，memory trace 可视化分析很直观 |
| 总分 | **4.4** | 系统优化工作的优秀范例 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] GoalRank: Group-Relative Optimization for a Large Ranking Model](../../ICLR2026/recommender/goalrank_group-relative_optimization_for_a_large_ranking_model.md)
- [\[ACL 2026\] GraphLoRA: Structure-Aware Low-Rank Adaptation for Large Language Model Recommendation](../../ACL2026/recommender/graphlora_structure-aware_low-rank_adaptation_for_large_language_model_recommend.md)
- [\[AAAI 2026\] Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models](../../AAAI2026/recommender/inference-aware_prompt_optimization_for_aligning_black-box_large_language_models.md)
- [\[AAAI 2026\] AutoPP: Towards Automated Product Poster Generation and Optimization](../../AAAI2026/recommender/autopp_towards_automated_product_poster_generation_and_optimization.md)
- [\[NeurIPS 2025\] R²ec: Towards Large Recommender Models with Reasoning](../../NeurIPS2025/recommender/r2ec_towards_large_recommender_models_with_reasoning.md)

</div>

<!-- RELATED:END -->
