---
title: >-
  [论文解读] Power Ensemble Aggregation for Improved Extreme Event AI Prediction
description: >-
  [NeurIPS 2025][地球科学][极端事件预测] 提出基于幂均值的自适应集成聚合方法，通过对生成式天气预测模型的集成成员得分施加非线性聚合（幂指数$p1$），显著提升极端高温事件的分类性能，尤其在高分位数阈值下效果更佳。 深度学习天气预报模型（如GraphCast、PanguWeather）在常规天气预测上已超越物理…
tags:
  - "NeurIPS 2025"
  - "地球科学"
  - "极端事件预测"
  - "集成聚合"
  - "幂均值"
  - "热浪分类"
  - "气候预测"
---

# Power Ensemble Aggregation for Improved Extreme Event AI Prediction

**会议**: NeurIPS 2025  
**arXiv**: [2511.11170](https://arxiv.org/abs/2511.11170)  
**代码**: 暂无  
**领域**: 时间序列  
**关键词**: 极端事件预测, 集成聚合, 幂均值, 热浪分类, 气候预测

## 一句话总结

提出基于幂均值的自适应集成聚合方法，通过对生成式天气预测模型的集成成员得分施加非线性聚合（幂指数$p>1$），显著提升极端高温事件的分类性能，尤其在高分位数阈值下效果更佳。

## 研究背景与动机

深度学习天气预报模型（如GraphCast、PanguWeather）在常规天气预测上已超越物理模型，但**极端事件预测仍是短板**。其根本原因在于：

**稀有性**：极端事件定义上就是小概率事件，训练数据中占比极低，模型倾向于预测均值附近的结果

**均值聚合的内在偏差**：当使用集成模型时，标准做法是对成员预测取均值。但均值操作天然地"压制"极端值——即使有少数成员正确预测了极端事件，也会被多数保守预测稀释

**缺少自适应机制**：不同强度的极端事件需要不同程度的"偏向极端"策略，但传统聚合方法（均值或最大值）缺乏灵活性

核心动机：**在均值（过保守）和最大值（过激进）之间，是否存在一个最优的中间策略？** 幂均值提供了一个连续可调的折衷方案。

## 方法详解

### 整体框架

系统分为三个部分：1) 基于U-Net的确定性天气预测模型，使用立方球面网格避免极点奇异性；2) Perlin噪声注入使模型变为生成式，输出$n=50$个集成成员；3) 幂均值聚合将集成成员的得分转换为极端事件分类概率。

### 关键设计

1. **极端事件定义与分类框架**

   使用局部气候学定义极端事件：对每个位置和季节计算地表气温的局部气候均值和标准差，将温度转换为局部异常值 $x$。阈值 $q$ 定义为：温度异常 $x$ 满足 $\Phi(x) \geq q$ 时视为极端事件，其中 $\Phi$ 是标准正态CDF。

   这是关键的设计选择——使用局部定义而非全局阈值，避免了只能捕捉类似撒哈拉沙漠等常年高温地区的限制。

2. **幂均值聚合**

   对 $n$ 个集成成员的局部异常预测 $\{\hat{x}_i\}_{i=1}^n$，先计算每个成员的得分 $\hat{s}_i = \Phi(\hat{x}_i)$，然后进行幂均值聚合：

    $\hat{s} = \left(\frac{1}{n}\sum_{i=1}^n \hat{s}_i^p\right)^{1/p}$

   当 $p=1$ 时退化为算术均值；$p \to \infty$ 时趋近最大值。参数 $p \geq 1$ 控制对极端预测成员的"权重倾斜"程度。

   注意幂均值应用在得分 $\hat{s}_i$（正数）上而非异常值 $\hat{x}_i$（可负）上，因为幂运算要求正数输入。

3. **生成式模型构造**

   在确定性U-Net基线输入中注入Perlin噪声以创建集成多样性。Perlin噪声相比白噪声的优势在于其空间连贯性——天气场的扰动应该在空间上平滑而非像素级独立。

   具体改进：在3D立方体 $[0,1]^3$ 上生成Perlin噪声，取对应地球表面的2D切片以保证全球连续性；使用对数正态分布随机化梯度幅度以更好捕获极端值（默认Perlin噪声被限制在$[-1,1]$）；通过卷积层学习的振幅调制器将不同频率的噪声组合为分形噪声。

### 损失函数 / 训练策略

使用连续排序概率评分（CRPS）作为训练损失。训练数据为1990-2010年ERA5再分析资料，重采样至1.5°空间分辨率和日时间分辨率。使用立方球面网格（6×48×48）避免极点奇异性。模型在单个16GB GPU上训练数小时。

## 实验关键数据

### 主实验——不同分位数的AUC对比

| 分位数q | 预报时效 | 均值聚合AUC | 幂均值AUC | $p_{opt}$ | 相对提升RI |
|---------|---------|------------|----------|-----------|-----------|
| 0.80 | 7天 | 基线 | 略优 | ~2 | ~0.5% |
| 0.90 | 7天 | 基线 | 较优 | ~5 | ~1.0% |
| 0.98 | 7天 | 基线 | 显著优 | ~20 | ~2.5% |
| 0.80 | 12天 | 基线 | 优 | ~2 | ~1% |
| 0.98 | 12天 | 基线 | 大幅优 | ~20 | ~4% |

### 与GraphCast的对比（测试集2018年）

| 方法 | q=0.80 | q=0.90 | q=0.98 |
|------|--------|--------|--------|
| 持续性模型 | 低 | 低 | 低 |
| GraphCast (确定性) | 良好 | 良好 | 有限 |
| 集成均值 | 良好 | 良好 | 良好 |
| **集成幂均值** | **最佳** | **最佳** | **最佳** |

在高分位数(q=0.98)和长预报时效场景下，基于简单生成模型的幂均值聚合甚至超越了确定性GraphCast。

### 关键发现

1. **最优幂指数$p_{opt}$与分位数$q$呈指数关系**：$\log(p_{opt}) = f(q)$ 几乎完美线性，提供了跨分位数的简单预测规则
2. **改进随极端程度增加**：相对提升RI随着分位数阈值的提高而增大，正好说明方法对越极端的事件越有效
3. **改进随预报时效增加**：长期预报的不确定性更大，幂均值对极端成员的偏向效果更明显
4. 在验证集上优化的$p_{opt}$可直接迁移到不同预报时效，说明鲁棒性良好

## 亮点与洞察

1. **方法的优雅简洁**：仅引入一个超参数$p$，就实现了从均值到最大值的连续可调聚合策略
2. **$p_{opt}$的指数缩放规律**：为不同极端阈值提供了即用型的$p$选择指南
3. **模型无关性**：幂均值聚合可以应用于任何生成式预测模型，不需要修改模型架构
4. **"以简胜繁"的示范**：简单生成模型+幂均值聚合在极端事件上超越了复杂的GraphCast

## 局限与展望

- **极端事件定义简化**：仅使用单变量（地表气温）和静态气候学
- **未考虑气候变化**：基于固定气候态定义的异常可能在非平稳气候下失效
- **AUC指标的局限**：未结合社会经济损害的应用导向评估
- **仅测试了自建简单模型**：是否对更强大的基线模型（如GenCast）也有效尚未验证
- 未包含多变量极端事件（如复合型极端天气）

## 相关工作与启发

- **GraphCast**: DeepMind确定性天气预报模型，本文对比基线
- **WeatherBench2**: 通用天气预报评测基准和数据源
- **Perlin噪声**: 用于计算机图形学的连续噪声，被巧妙地用于气象集成扰动

## 评分

- 新颖性: ⭐⭐⭐☆☆ — 幂均值已有前人研究，创新在于气候领域的系统验证
- 实验充分度: ⭐⭐⭐⭐☆ — 多分位数、多时效对比充分，但缺少更强基线模型验证
- 写作质量: ⭐⭐⭐⭐☆ — 短小精悍，重点突出
- 价值: ⭐⭐⭐⭐☆ — 方法简单通用，对极端事件预测有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Predicting Public Health Impacts of Electricity Usage](predicting_public_health_impacts_of_electricity_usage.md)
- [\[NeurIPS 2025\] Reasoning With a Star: A Heliophysics Dataset and Benchmark for Agentic Scientific Reasoning](reasoning_with_a_star_a_heliophysics_dataset_and_benchmark_for_agentic_scientifi.md)
- [\[NeurIPS 2025\] Adaptive Online Emulation for Accelerating Complex Physical Simulations](adaptive_online_emulation_for_accelerating_complex_physical_simulations.md)
- [\[NeurIPS 2025\] A Probabilistic U-Net Approach to Downscaling Climate Simulations](a_probabilistic_unet_approach_to_downscaling_climate_simulat.md)
- [\[NeurIPS 2025\] ControlFusion: A Controllable Image Fusion Framework with Language-Vision Degradation Prompts](controlfusion_a_controllable_image_fusion_framework_with_language-vision_degrada.md)

</div>

<!-- RELATED:END -->
