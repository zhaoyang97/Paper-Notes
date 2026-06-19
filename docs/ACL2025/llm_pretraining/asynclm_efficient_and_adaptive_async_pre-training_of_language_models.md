---
title: >-
  [论文解读] AsyncLM: Efficient and Adaptive Async Pre-training of Language Models
description: >-
  [ACL 2025][预训练][异步预训练] 本文提出AsyncLM，一种高效的异步预训练框架，通过自适应梯度补偿和动态批量调度策略解决异步分布式训练中的梯度过期问题，在保持与同步训练相当的模型质量的同时，将大规模语言模型预训练的吞吐量提升了1.4-1.8倍。 领域现状：大语言模型的预训练需要消耗巨量的计算资源…
tags:
  - "ACL 2025"
  - "预训练"
  - "异步预训练"
  - "语言模型"
  - "分布式训练"
  - "梯度延迟"
  - "自适应学习率"
---

# AsyncLM: Efficient and Adaptive Async Pre-training of Language Models

**会议**: ACL 2025  
**代码**: 无  
**领域**: LLM预训练  
**关键词**: 异步预训练, 语言模型, 分布式训练, 梯度延迟, 自适应学习率

## 一句话总结
本文提出AsyncLM，一种高效的异步预训练框架，通过自适应梯度补偿和动态批量调度策略解决异步分布式训练中的梯度过期问题，在保持与同步训练相当的模型质量的同时，将大规模语言模型预训练的吞吐量提升了1.4-1.8倍。

## 研究背景与动机

**领域现状**：大语言模型的预训练需要消耗巨量的计算资源，通常在数百甚至数千个GPU/TPU上进行分布式训练。当前主流的分布式训练方案采用同步SGD（Synchronous SGD），所有计算节点在每一步都需要等待最慢的节点完成梯度计算，然后进行全局梯度同步。数据并行（Data Parallelism）、张量并行（Tensor Parallelism）和流水线并行（Pipeline Parallelism）是三种常见的并行策略。

**现有痛点**：同步训练存在严重的"掉队者"（Straggler）问题——在大规模集群中，个别节点因硬件差异、网络波动或内存不足等原因导致计算速度显著慢于其他节点，所有快节点都必须等待慢节点，导致GPU利用率下降。实测数据显示，在1000+GPU的集群中，同步训练的有效GPU利用率可能低于60%。

**核心矛盾**：异步训练（Asynchronous SGD）可以消除等待开销，但传统异步方法存在梯度过期（Staleness）问题——当一个慢节点的梯度最终到达参数服务器时，模型参数已经被其他节点更新了多次，过期的梯度可能导致训练发散或收敛到次优解。在LLM预训练这种对训练稳定性要求极高的场景中，这一问题尤为突出。

**本文目标**：设计一种异步预训练框架，在大规模LLM训练中实现高吞吐量的同时保持训练质量（困惑度、下游任务性能与同步训练持平）。

**切入角度**：作者观察到梯度过期的影响是非均匀的——不同参数组（如attention层vs FFN层、浅层vs深层）对过期梯度的敏感性不同。基于此观察设计自适应补偿机制。

**核心 idea**：通过参数组级别的自适应梯度补偿和动态任务调度，将异步训练的质量损失控制在可接受范围内，从而充分释放异步训练的吞吐量优势。

## 方法详解

### 整体框架
AsyncLM采用异步数据并行+流水线并行的混合架构。各计算节点独立完成前向和反向计算后，不等待其他节点即刻发送梯度到参数服务器。参数服务器收到梯度后立即应用更新。在此基础上，加入三个核心组件：自适应梯度补偿器（AGC）、动态批量调度器（DBS）和异步感知学习率调节器（AALR）。输入为预训练数据流，输出为训练好的语言模型。

### 关键设计

1. **自适应梯度补偿器（Adaptive Gradient Compensator, AGC）**:

    - 功能：对过期梯度进行补偿修正，减少异步训练的质量损失
    - 核心思路：对每个到达的梯度 $g_t^{(k)}$（来自节点$k$，基于第$t-\tau$步的参数计算，$\tau$为延迟步数），估计一个补偿项 $\Delta g$，使得补偿后的梯度 $\hat{g} = g_t^{(k)} + \Delta g$ 近似于使用当前参数计算得到的梯度。AGC使用一阶泰勒展开来估计补偿项：$\Delta g \approx \tau \cdot H \cdot \bar{v}$，其中 $H$ 是近似Hessian矩阵（通过指数移动平均的梯度差分估计），$\bar{v}$ 是近期参数更新的平均方向。关键创新是按参数组（attention-Q/K/V/O、FFN-up/down、embedding等）分别维护补偿统计量，因为不同参数组的梯度动态差异显著。
    - 设计动机：之前的异步训练方法通常使用全局统一的延迟补偿，忽略了不同层/参数组对过期梯度的敏感性差异。实验表明attention层的Q/K参数对过期梯度特别敏感

2. **动态批量调度器（Dynamic Batch Scheduler, DBS）**:

    - 功能：通过智能的数据分配减少异步训练中的梯度延迟
    - 核心思路：根据各计算节点的实时处理速度动态分配不同大小的micro-batch。具体而言，维护一个节点速度估计器，对每个节点的计算耗时进行滑动窗口估计。速度较慢的节点分配较小的batch（减少其每步计算时间），速度较快的节点分配较大的batch（充分利用其计算能力）。约束条件是所有节点的梯度在"有效样本数"上等权（通过梯度缩放实现：$g_i^{scaled} = g_i \cdot B_{base}/B_i$，其中$B_i$为节点$i$的实际batch大小）。这样可以将最大延迟从 $O(B_{slow}/B_{fast})$ 降低到接近常数。
    - 设计动机：传统异步训练中所有节点使用相同batch大小，导致慢节点的延迟非常大；自适应batch分配可以从源头减少延迟，而不仅仅是事后补偿

3. **异步感知学习率调节器（Async-Aware Learning Rate, AALR）**:

    - 功能：根据当前的异步延迟程度动态调整学习率
    - 核心思路：定义一个全局延迟指标 $\bar{\tau}_t$，表示最近一个窗口内所有接收到的梯度的平均延迟步数。当$\bar{\tau}_t$超过阈值时，自动降低学习率：$lr_t = lr_{base} \cdot \min(1, \tau_{thresh}/\bar{\tau}_t)$。在训练运行正常（延迟小）时使用完整学习率，在系统繁忙（延迟大）时自动降速保稳定。配合warmup和cosine decay调度器使用，AALR修正因子作为乘法调制项叠加在原始调度上。
    - 设计动机：传统固定学习率在异步训练中可能过大（当延迟突增时）或过小（当系统运行顺畅时），自适应调节可以在效率和稳定性之间动态平衡

### 损失函数 / 训练策略
预训练目标沿用标准的next-token prediction损失（交叉熵损失）。优化器使用AdamW，基础学习率3e-4，cosine decay调度。gradient clipping设为1.0。AGC补偿项的Hessian估计使用EMA系数0.99，DBS的速度估计窗口为100步。整个框架与现有的DeepSpeed/Megatron训练栈兼容。

## 实验关键数据

### 主实验

| 模型规模 | 指标 | AsyncLM | 同步训练 | 朴素异步 | 加速比(vs同步) |
|---------|------|---------|---------|---------|-------------|
| 1.3B | PPL (val) | 14.82 | 14.65 | 16.21 | 1.43x |
| 7B | PPL (val) | 8.93 | 8.87 | 10.34 | 1.58x |
| 13B | PPL (val) | 7.45 | 7.38 | 9.12 | 1.76x |
| 7B | MMLU (5-shot) | 52.1 | 52.8 | 46.3 | - |
| 7B | HellaSwag | 74.6 | 75.1 | 68.7 | - |

### 消融实验

| 配置 | 7B PPL | 吞吐量提升 | 说明 |
|------|--------|----------|------|
| Full AsyncLM | 8.93 | 1.58x | 完整框架 |
| w/o AGC | 9.67 | 1.58x | 去掉梯度补偿，PPL上升0.74 |
| w/o DBS | 9.21 | 1.38x | 去掉动态调度，吞吐量下降 |
| w/o AALR | 9.15 | 1.58x | 去掉自适应LR，PPL上升0.22 |
| AGC (全局) | 9.31 | 1.58x | 全局补偿替代参数组补偿 |
| 朴素异步 | 10.34 | 1.62x | 无任何补偿策略 |
| 同步训练 | 8.87 | 1.00x | 基准 |

### 关键发现
- AsyncLM在7B模型上只损失0.06 PPL的代价下获得了1.58倍的吞吐量提升，而朴素异步方法虽然吞吐量略高（1.62x），但PPL损失高达1.47
- AGC是保证质量的最关键组件（去掉后PPL上升0.74），而DBS主要通过减少延迟间接影响质量并直接影响吞吐量
- 参数组级别的补偿（AGC）显著优于全局补偿（PPL: 8.93 vs 9.31），验证了不同参数组对过期梯度敏感性不同的假设
- 随着模型规模增大，AsyncLM的加速比提升（1.43x→1.76x），这是因为大模型训练中通信开销占比更高，异步训练的优势更明显

## 亮点与洞察
- 参数组级别的自适应梯度补偿是一个精致的设计——它利用了Transformer内部不同模块的梯度动态特性差异，这一观察本身就是一个有价值的发现。未来可以将这种参数组感知的思路迁移到其他优化问题中（如混合精度训练、选择性参数冻结）
- DBS通过从源头减少延迟来解决问题，比事后补偿更加根本。这种"预防优于治疗"的思路在系统设计中很常见但在异步训练中之前没有被充分利用
- 模型规模越大异步训练越有优势这一发现有重要的实践意义：在LLM预训练持续向更大规模发展的趋势下，异步方案的吸引力会越来越大

## 局限与展望
- 当前实验最大规模为13B参数，在100B+的超大模型上的表现还需验证
- AGC中的Hessian近似在训练不稳定阶段（如学习率warmup初期）可能不准确
- 框架与流水线并行的结合存在一些工程挑战，如微批次调度与流水线调度的耦合
- 未来可以探索将AsyncLM与其他效率优化技术（如混合精度训练、稀疏Attention）结合

## 相关工作与启发
- **vs Local SGD (Lin et al.)**: Local SGD通过减少通信频率实现加速，但每次通信都是全同步的；AsyncLM完全消除了同步等待，在异构集群中优势更大
- **vs DC-ASGD (Zheng et al.)**: DC-ASGD也使用了梯度延迟补偿，但只做全局补偿；AsyncLM的参数组级别补偿更加精细
- **vs 流水线并行 (GPipe, PipeDream)**: 流水线并行通过切分模型来提高利用率，但仍需要micro-batch间的同步；AsyncLM与流水线并行是正交的，可以组合使用

## 评分
- 新颖性: ⭐⭐⭐⭐ 参数组级别自适应补偿和动态批量调度的组合新颖
- 实验充分度: ⭐⭐⭐⭐ 多种规模模型评估，消融实验详尽
- 写作质量: ⭐⭐⭐⭐ 技术描述清晰，系统设计合理
- 价值: ⭐⭐⭐⭐ 对大规模LLM训练的工程实践有直接参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] Pre-Training Curriculum for Multi-Token Prediction in Language Models](pre-training_curriculum_for_multi-token_prediction_in_language_models.md)
- [\[ACL 2025\] Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](metarater_a_multidimensional_data_selection_method.md)
- [\[ACL 2025\] LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models](leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg.md)
- [\[ACL 2025\] Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack](stealing_training_data_from_large_language_models_in_decentralized_training_thro.md)

</div>

<!-- RELATED:END -->
