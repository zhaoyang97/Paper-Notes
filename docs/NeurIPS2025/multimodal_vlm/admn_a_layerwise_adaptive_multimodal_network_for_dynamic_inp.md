# ADMN: A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources

**会议**: NeurIPS 2025  
**arXiv**: [2502.07862](https://arxiv.org/abs/2502.07862)  
**代码**: [https://github.com/nesl/ADMN](https://github.com/nesl/ADMN)  
**领域**: 多模态VLM / 模型压缩  
**关键词**: multimodal, adaptive depth, LayerDrop, quality-of-information, dynamic compute budget, layer allocation, sensor corruption  

## 一句话总结
提出 ADMN（Adaptive Depth Multimodal Network），通过两阶段训练——(1) Multimodal LayerDrop 微调使 backbone 适应任意层配置，(2) QoI感知控制器动态分配层预算给各模态——在严格计算约束下根据每个模态的信息质量(QoI)自适应分配层数，匹配全量模型精度同时减少 75% FLOPs 和 60% 延迟。

## 背景与动机
多模态系统部署在动态环境中面临双重挑战：
1. **计算资源变化**：多租户、设备异构、热节流导致可用计算预算随时间变化，且需严格遵守（不能超过上限）
2. **输入质量变化**：传感器损坏、天气变化等导致各模态的信号质量动态波动——被严重损坏的模态不应消耗与正常模态相同的计算资源

现有方法的局限：静态多模态网络无法适配变化的计算预算；现有动态网络（Early Exit、DynMM等）优化平均case效率但无法处理严格预算约束；且几乎所有方法都忽略了各模态QoI的影响。

## 核心问题
如何构建一个**同时适应动态计算约束和动态输入质量**的多模态网络——按计算预算总量控制总层数，并根据各模态QoI按需分配层？

## 方法详解

### 整体框架
两阶段训练：Stage 1 构建层自适应的多模态backbone，Stage 2 训练QoI感知控制器分配层预算。

### 关键设计
1. **Multimodal LayerDrop（Stage 1）**: 
   - 先在 MAE 预训练阶段引入 LayerDrop（率0.2），使ViT backbone对缺失层具有鲁棒性
   - 再在多模态任务微调时保持 LayerDrop，使融合层和输出层适应各种backbone层配置
   - **全backbone Dropout**：10%概率丢弃某模态backbone的全部层——模拟该模态完全不可用的极端情况
   - 结果：单套权重可在任意层预算下工作

2. **QoI感知控制器（Stage 2）**:
   - 轻量级架构：下采样输入→模态特定卷积→Transformer融合→MLP输出层分配logits
   - **腐蚀感知监督（ADMN）**：用额外的腐蚀预测loss $\mathcal{L}_{corr}$ 显式教控制器注意模态QoI
   - **自编码器初始化（ADMN_AE）**：无腐蚀标注时，先用AE预训练控制器的感知层——重建目标迫使latent space按QoI聚类（t-SNE可视化验证）
   - 消融证明：纯task loss无法学到QoI感知分配

3. **可微分层选择**:
   - Gumbel-Softmax采样（温度1）+ Top-L离散化 + Straight-through estimator
   - 实现在总预算L层约束下对C个backbone层的可微分选择

### 损失函数 / 训练策略
Stage 1: Task loss + LayerDrop(0.2) + Full-backbone dropout(10%)。Stage 2: $\mathcal{L}_{total} = \mathcal{L}_{model} + \mathcal{L}_{corr}$（或AE初始化+$\mathcal{L}_{model}$）。

## 实验关键数据

| 数据集 | 任务 | 方法 | 6层 | 8层 | 12层 | 16层 | 上界(24层) |
|--------|------|------|-----|-----|------|------|----------|
| GDTM(高斯噪声) | 定位(cm↓) | Naive Alloc | 112.5 | 97.6 | 46.9 | 31.0 | 29.6 |
| | | **ADMN** | **51.4** | **39.0** | **33.1** | **30.3** | |
| | | **ADMN_AE** | **53.6** | **38.4** | **33.5** | **29.4** | |
| GDTM(低光) | 定位(cm↓) | Naive Alloc | 90.3 | 67.3 | 27.1 | 17.7 | 18.8 |
| | | **ADMN** | **49.5** | **23.9** | **18.0** | **17.3** | |
| MM-Fi | 分类(↑) | Naive Alloc | 5.56% | 12.96% | 29.01% | 42.90% | 44.44% |
| | | **ADMN** | **35.03%** | **39.25%** | **41.92%** | **43.31%** | |
| AVE | 分类(↑) | Naive Alloc | 36.16% | 46.89% | 65.71% | 67.95% | 71.19% |
| | | **ADMN** | **57.07%** | **62.95%** | **67.48%** | **66.60%** | |

GDTM(Blur, 8层)：ADMN定位误差~11cm，接近上界(9.4cm)，而减少75% FLOPs + 60%延迟。

### 消融实验要点
- **LayerDrop阶段**：MAE预训练+微调双阶段加LayerDrop效果最好（Fig 6）
- **QoI监督必要性**：纯task loss控制器无法学到QoI感知（Table 5: task loss 46.2% vs ADMN 57.07%）
- **AE latent space**：t-SNE证实AE自动将不同corruption级别聚类（Fig 5）
- **全backbone dropout**：必要，否则模型无法处理某模态完全缺失的极端情况
- **三模态泛化**：RGB+Depth+mmWave三模态实验验证通用性（Fig 7）
- **不等计算模态**：视觉backbone 3倍于audio的FLOPs场景下，ADMN正确分配（Table 4）
- **6 seeds稳定性**：标准差<5%，大预算更稳定

## 亮点
- **双重自适应**是核心创新——同时适应计算预算变化和输入质量变化，前所未有
- LayerDrop从单模态文本Transformer扩展到多模态ViT是非平凡的工程贡献——需要全backbone dropout等特殊处理
- ADMN_AE不需要任何QoI标注就能学到感知分配——实用价值极高
- 控制器仅占总FLOPs ~1%，额外开销极小
- 6 seeds × 3数据集 × 3/4层预算 × 3-4种corruption = 大规模消融实验

## 局限性 / 可改进方向
- 每个层预算需单独训练控制器（通用控制器初步结果可行但待完善）
- 批推理不兼容（不同样本有不同层配置，难以batch）
- 可与Early Exit结合进一步提效
- 仅在嵌入级融合架构上验证，数据级/late fusion待探索

## 与相关工作的对比
- **vs DynMM/AdaMML（模型选择）**: 这些方法选择预定义的专家模型；ADMN在单个模型内部做层级分配，更灵活
- **vs PrefixKV（同系列笔记）**: PrefixKV按层分配KV cache预算；ADMN按模态分配层预算——两者都是"自适应跨维度分配"
- **vs ASF（同系列笔记）**: ASF在统一规范空间融合传感器并估计可用性；ADMN通过层分配实现更细粒度的资源控制

## 启发与关联
- 层分配的QoI感知思路可迁移到VLM：如低质量图像给视觉encoder少分配层，高质量文本给LLM多分配层
- 与 [ideas/model_compression/20260318_cross_layer_token_budget_allocation.md](../../../ideas/model_compression/20260318_cross_layer_token_budget_allocation.md) 直接相关——ADMN按模态分配层，该idea按层分配token预算
- AE初始化的QoI聚类可用于自动检测传感器退化——不需要显式标注

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 双重自适应（计算+QoI）是全新问题定义和解决方案
- 实验充分度: ⭐⭐⭐⭐⭐ 3数据集、3-4类corruption、6 seeds、大量消融+定性分析
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰、方法描述详尽、消融极为彻底
- 价值: ⭐⭐⭐⭐⭐ 解决了多模态部署的实际双重约束问题，AE方案无需QoI标注
