---
title: >-
  [论文解读] LoPrune: Efficient Data Pruning for LoRA-Based Fine-Tuning of Vision Transformer
description: >-
  [CVPR 2026][模型压缩][数据剪枝] 针对端侧 LoRA 微调中"数据冗余"这个被忽视的瓶颈，LoPrune 提出把样本影响函数投影到 LoRA 可训练子空间打分（TSA Score），并用 K-FAC 曲率近似实现单 epoch 高效评分，在 ViT/DeiT/Swin/DETR 等模型上把微调开销最多降 72.9%、训练提速最多 3.69×，准确率反而最高提升 3.50%。
tags:
  - "CVPR 2026"
  - "模型压缩"
  - "数据剪枝"
  - "LoRA 微调"
  - "影响函数"
  - "可训练子空间"
  - "K-FAC"
---

# LoPrune: Efficient Data Pruning for LoRA-Based Fine-Tuning of Vision Transformer

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/He_LoPrune_Efficient_Data_Pruning_for_LoRA-Based_Fine-Tuning_of_Vision_Transformer_CVPR_2026_paper.html)  
**代码**: 待确认  
**领域**: 模型压缩 / 高效微调  
**关键词**: 数据剪枝, LoRA 微调, 影响函数, 可训练子空间, K-FAC

## 一句话总结
针对端侧 LoRA 微调中"数据冗余"这个被忽视的瓶颈，LoPrune 提出把样本影响函数投影到 LoRA 可训练子空间打分（TSA Score），并用 K-FAC 曲率近似实现单 epoch 高效评分，在 ViT/DeiT/Swin/DETR 等模型上把微调开销最多降 72.9%、训练提速最多 3.69×，准确率反而最高提升 3.50%。

## 研究背景与动机

**领域现状**：视觉模型大量部署在 IoT/边缘设备上，需要在设备端持续微调以适应光照、天气、用户行为等动态变化。受算力、内存、电量约束，端侧通常用参数高效微调（PEFT），其中 LoRA 通过给权重矩阵加低秩增量、冻结主干，是主流方案。

**现有痛点**：LoRA 把"参数层面"的开销压下去了，但**数据层面的冗余依旧存在**——本地数据里大量样本对模型提升贡献甚微，却照样消耗训练资源。数据剪枝本可以挑出最有信息量的子集来省算力，但现有剪枝方法直接套到 LoRA 微调上水土不服。

**核心矛盾**：现有剪枝有两条路且都和 LoRA 冲突。其一，基于学习动态（EL2N、Forgetting、GraNd）的方法假设"越难学越重要"，需要跨多个 epoch 累积统计；但 LoRA 微调往往几个 epoch 就收敛，warm-up 太短打分就不准（甚至不如随机剪枝），而专门多加 epoch 去记录动态又与"省开销"的初衷相悖——打分成本可能反超训练本身。其二，基于影响函数的方法在**全参数空间**评估样本重要性，但 LoRA 只更新低秩 adapter、冻结主干，大量梯度能量被投到永不更新的冻结方向上，造成对样本重要性的系统性高估或误判。

**本文目标**：设计一种**与 LoRA 可更新方向对齐、且能在第一个 epoch 内完成**的低成本数据剪枝，从数据和模型两个层面同时压低端侧微调开销。

**切入角度**：既然 LoRA 只在低秩子空间更新，那么样本重要性就该在这个"可训练子空间"里衡量，而不是在全参数空间里——只奖励那些真能被更新的方向。

**核心 idea**：用 Jacobian 把影响函数从全参数空间投影到 LoRA 子空间得到 TSA Score（只算可更新方向、顺带降维），再用 K-FAC 曲率近似把 Hessian 求逆变成近线性时间，实现单 epoch 剪枝。

## 方法详解

### 整体框架
LoPrune 把数据剪枝建模为：从训练集 $D=\{z_i\}_{i=1}^N$ 中用二值向量 $m$ 选出保留子集 $S(m)$，在尽量多删样本的同时让微调性能损失最小。流程分三步：第一步用 Jacobian 把样本对模型的影响从全参数空间 $\theta$ 映射到 LoRA 可训练子空间 $\phi$，得到精确 TSA Score；第二步用 K-FAC 近似，从每个 LoRA 层的输入协方差 $\Sigma_x$ 与输出梯度协方差 $\Sigma_\delta$ 算出该层分数并聚合为最终 TSA Score；第三步按分数降序保留高分样本，得到子集 $S(m)$ 用于微调。整个打分所需的 $x_t$、$\delta_t$ 都来自正常的前/反向传播，不引入额外计算，因此能在第一个 epoch 内一次完成。

### 关键设计

**1. TSA Score：把影响函数投影到 LoRA 可训练子空间，只奖励真能更新的方向**

这一步直击"全参数空间打分会把冻结方向的梯度能量也算进去、从而误判样本重要性"的痛点。LoPrune 先把所有 LoRA 因子拼成一个可训练参数向量 $\phi = (\mathrm{vec}(A_1), \mathrm{vec}(B_1), \dots)$，记 $\theta(\phi)$ 为注入低秩增量后的等效全模型权重，$J=\partial\,\mathrm{vec}(\theta)/\partial\phi$ 为该映射的 Jacobian。用链式法则就能不显式构造 $J$ 而直接得到子空间里的梯度与曲率：$g_\phi(z)=J^\top\mathrm{vec}(g_\theta(z))$，$H_\phi=J^\top H_\theta J$。于是精确 TSA Score 为

$$s(z) = g_\phi(z)^\top H_\phi^{-1} g_\phi(z)$$

它等价于在 $H_\theta$ 度量下把影响方向从全空间投到 LoRA 子空间。这样做有两重好处：一是只奖励"实际可被更新"的方向，不会因冻结主干上的大梯度而高估那些预训练已学好的样本；二是把逆 Hessian-向量积（iHVP）问题从全空间降到低秩子空间，避免对冻结权重算梯度/曲率，开销大幅下降。消融显示，加上这个投影（L）比不加在准确率上提升约 1.33%–3.38%、剪枝时间还降 82.4%–93.2%。

**2. K-FAC 近似评分：把 Hessian 求逆变成近线性时间，实现单 epoch 打分**

即便降到子空间，直接解 Hessian 仍然昂贵。LoPrune 用 Kronecker 因子化曲率（K-FAC）近似：假设跨层参数项约为零、只看层内关系，并利用"网络梯度可分解为输入与输出贡献"，把 $H_\theta \approx \Sigma_x \otimes \Sigma_\delta$，其中 $\Sigma_x=\mathbb{E}[xx^\top]$ 是输入激活协方差、$\Sigma_\delta=\mathbb{E}[\delta\delta^\top]$ 是输出梯度协方差。由 Kronecker 积定理 $H_\theta^{-1}=\Sigma_x^{-1}\otimes\Sigma_\delta^{-1}$，TSA Score 可写成对所有可训练块求平均：

$$s(z) = \frac{1}{T}\sum_{t=1}^{T}\left(x_t^\top \Sigma_{x,t}^{-1} x_t\right)\left(\delta_t^\top \Sigma_{\delta,t}^{-1} \delta_t\right)$$

其中 $t$ 索引所有可训练的 $A$、$B$ 块。把 Jacobian 映射代入后，B 块映射让输入侧落到 $A$ 张成的子空间、A 块映射让输出侧落到 $B$ 张成的子空间（对应公式 14、15）。由于 $x_t$、$\delta_t$ 本就来自前/反向传播，整个打分不额外增加计算。这一步把打分压成近线性时间，是 LoPrune 能"单 epoch、低开销"的关键。消融中 L+K 相比仅用 L+F（传统 Hessian 求逆近似）时间再降 91.9%–93.3%、准确率几乎不变（约 0.05% 差距）。

## 实验关键数据

### 主实验
四模型三数据集、剪枝率 0.5（除 No pruning），报告准确率 / 总耗时（含打分+训练，秒）；分类用 Top-1、检测用 AP50：

| 方法 | ViT-S/CIFAR100 | ViT-S/Tiny-IN | Swin-T/COCO | DETR/COCO |
|------|----------------|---------------|-------------|-----------|
| No pruning | 90.05 (561.1) | 84.91 (992.1) | 67.18 (5403.1) | 60.23 (7220.2) |
| Random | 88.85 (343.3) | 83.86 (561.4) | 66.11 (3348.2) | 59.13 (4464.3) |
| MoSo 2023 | 89.25 (893.3) | 84.17 (2394.3) | 66.40 (8532.7) | 59.74 (11160.3) |
| TFDP 2025 | 88.96 (544.2) | 83.78 (783.2) | 65.82 (5670.3) | 59.54 (7416.1) |
| **LoPrune** | **89.59 (371.5)** | **84.78 (622.1)** | **66.70 (3620.0)** | **59.92 (4804.2)** |

LoPrune 在准确率和总耗时上同时领先：相比最优 baseline 分类 Top-1 提升 0.34%–6.62%、总开销降 1.07×–5.72×；相比不剪枝通常省 25%–37% 时间而准确率仅掉 0.10%–1.21%。Tiny-ImageNet 上剪 50% 数据仅掉 0.22% Top-1、省 37.4% 时间。

### 消融实验
TSAS（L）与 K-FAC（K）的组件消融，F 为传统 Hessian 求逆近似 baseline：

| L | K | F | CIFAR-100 Acc | Time(s) |
|---|---|---|---------------|---------|
| ✓ | - | - | 89.64 | 5542.3 |
| - | ✓ | - | 86.88 | 2114.4 |
| ✓ | - | ✓ | 89.13 | 489.2 |
| - | - | ✓ | 86.21 | 4793.3 |
| ✓ | ✓ | - | **89.59** | **371.5** |

### 关键发现
- **TSAS（L）是精度主力**：去掉 L 后准确率明显下降，L+K / L+F 相比 K / F 提升 2.33%–2.92%、剪枝时间降 82.5%–89.8%，证明"在可训练子空间打分"是核心。
- **K-FAC（K）是效率主力**：精度上只带来微小提升，但把打分压成近线性时间，L+K 是准确率与耗时的最佳组合。
- **高剪枝率下优势最明显**：剪枝率超过 0.5 后 LoPrune 比其他方法高 0.24%–13.72%；低剪枝率（如 10%）时多数剪枝方法甚至优于全量训练，因为删掉了带负贡献的噪声样本。
- 有意思的现象：EL2N、GraNd 等需多 epoch warm-up 的方法在短程微调下难收敛，高剪枝率时比 LoPrune 低 5.08%–10.34%，且额外打分开销常反超剪枝节省。

## 亮点与洞察
- **"在哪个空间打分"比"用什么打分函数"更关键**：把影响函数投到 LoRA 子空间这一步，单独就贡献了大部分精度增益——它点破了"PEFT 时代沿用全参数空间重要性度量会失真"这一被忽视的错配。
- **借力前/反向传播自带的统计量实现零额外开销打分**：$x_t$、$\delta_t$ 本就在训练时算好，K-FAC 顺手复用，使"单 epoch 剪枝"成为可能，这种"白嫖中间量"的思路可迁移到其他需要二阶信息的轻量算法。
- **首个专为 LoRA 设计的数据剪枝**：把"数据冗余"和"参数冗余"两类压缩第一次在 PEFT 框架下打通，为端侧低成本微调提供了数据×模型双层路径。

## 局限与展望
- K-FAC 依赖"跨层参数项约为零、层内可 Kronecker 分解"的假设，在极小 rank 或异常协方差结构下近似误差的影响未充分讨论（⚠️ 以原文为准）。
- 实验集中在 ViT 系视觉模型与 LoRA，是否推广到其他 PEFT（如 Adapter、Prefix-tuning）或 CNN/多模态结构仍待验证。
- 低剪枝率（<30%）下各方法差距不大，LoPrune 的优势主要体现在高剪枝率场景，端侧实际能省多少取决于可接受的剪枝比例。
- 剪枝起始 epoch 等超参对结果有影响（Fig. 显示第 2 epoch 起剪较优），需按任务调参（⚠️ 以原文为准）。

## 相关工作与启发
- **vs EL2N / GraNd / Forgetting**: 它们靠多 epoch 学习动态打分，短程 LoRA 微调下难收敛且打分开销可能反超训练；LoPrune 单 epoch、近线性时间打分，避开了"为打分而加 epoch"的悖论。
- **vs Optimization-based / MoSo**: 同样用影响函数/经验风险变化衡量重要性，但在全参数空间评估、计算昂贵；LoPrune 投到 LoRA 子空间，既对齐可更新方向又大幅降维。
- **vs TFDP 2025**: TFDP 基于样本形状-类别的复杂度打分，是训练无关启发式；LoPrune 直接对齐 LoRA 实际更新方向，高剪枝率下准确率更稳。

## 评分
- 新颖性: ⭐⭐⭐⭐ "在 LoRA 子空间衡量样本重要性"角度新颖，是首个面向 LoRA 的数据剪枝。
- 实验充分度: ⭐⭐⭐⭐ 覆盖四模型三数据集、分类+检测两类任务，组件消融与剪枝率扫描清晰。
- 写作质量: ⭐⭐⭐⭐ 动机与推导链条扎实，部分 K-FAC 假设的适用边界交代偏简。
- 价值: ⭐⭐⭐⭐ 对端侧/IoT 低成本持续微调有实际意义，思路可迁移到其他 PEFT 场景。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] TaskIT: Memory-Efficient Fine-Tuning of Multi-LoRA LLMs via Cross-Task Importance Transfer](taskit_memory-efficient_fine-tuning_of_multi-lora_llms_via_cross-task_importance.md)
- [\[CVPR 2026\] Edge-RecViT: Efficient Vision Transformer via Semantic-Refined Dynamic Recursion](edge-recvit_efficient_vision_transformer_via_semantic-refined_dynamic_recursion.md)
- [\[CVPR 2026\] TAS-LoRA: Transformer Architecture Search with Mixture-of-LoRA Experts](tas-lora_transformer_architecture_search_with_mixture-of-lora_experts.md)
- [\[CVPR 2026\] ThinkingViT: Matryoshka Thinking Vision Transformer for Elastic Inference](thinkingvit_matryoshka_thinking_vision_transformer_for_elastic_inference.md)
- [\[CVPR 2026\] Mining Attribute Subspaces for Efficient Fine-tuning of 3D Foundation Models](mining_attribute_subspaces_for_efficient_fine-tuning_of_3d_foundation_models.md)

</div>

<!-- RELATED:END -->
