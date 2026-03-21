# Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization

**会议**: CVPR 2025  
**arXiv**: 见CVF  
**代码**: 待确认  
**领域**: 对齐RLHF  
**关键词**: 待补充

## 一句话总结
> 注：本笔记基于论文摘要撰写，尚未阅读全文。待下载完整论文后补充详细方法和实验分析。

本文属于 对齐RLHF 方向。To improve aesthetics economically, this paper uses existing generic preference data and introduces step-by-step preference optimization (SPO) that discards the propagation strategy and allows fine-grained image details to be assessed.

## 研究背景与动机
1. **领域现状**：However, preference labels provided in existing datasets are blended with layout and aesthetic opinions, which would disagree with aesthetic preference.
2. **现有痛点**：现有方法在效率或效果方面存在提升空间。
3. **核心矛盾**：需要在性能与复杂度之间取得更好的平衡。
4. **本文要解决什么？** To improve aesthetics economically, this paper uses existing generic preference data and introduces step-by-step preference optimization (SPO) that discards the propagation strategy and allows fine-grained image details to be assessed.
5. **切入角度**：从新的技术视角切入。
6. **核心idea一句话**：To improve aesthetics economically, this paper uses existing generic preference data and introduces step-by-step preference optimization (SPO) that discards the propagation strategy and allows fine-gr

## 方法详解

### 整体框架
> 原文摘要（供参考）：
> Generating visually appealing images is fundamental to modern text-to-image generation models. A potential solution to better aesthetics is direct preference optimization (DPO), which has been applied to diffusion models to improve general image quality including prompt alignment and aesthetics. Popular DPO methods propagate preference labels from clean image pairs to all the intermediate steps along the two generation trajectories. However, preference labels provided in existing datasets are blended with layout and aesthetic opinions, which would disagree with aesthetic preference. Even if aesthetic labels were provided (at substantial cost), it would be hard for the two-trajectory methods to capture nuanced visual differences at different steps. To improve aesthetics economically, this paper uses existing generic preference data and introduces step-by-step preference optimization (SPO) that discards the propagation strategy and allows fine-grained image details to be assessed. Specifically, at each denoising step, we 1) sample a pool of candidates by denoising from a shared noise latent, 2) use a step-aware preference model to find a suitable win-lose pair to supervise the diffusion model, and 3) randomly select one from the pool to initialize the next denoising step. This strategy ensures that diffusion models focus on the subtle, fine-grained visual differences instead of layout aspect. We find that aesthetics can be significantly enhanced by accumulating these improved minor differences. When fine-tuning Stable Diffusion v1.5 and SDXL, SPO yields significant improvements in aesthetics compared with existing DPO methods while not sacrificing image-text alignment compared with vanilla models. Moreover, SPO converges much faster than DPO methods due to the use of more correct preference labels provided by the step-aware preference model. Code and models are available at https://github.com/RockeyCoss/SPO.

基于摘要理解，本文的方法可以概括为：To improve aesthetics economically, this paper uses existing generic preference data and introduces step-by-step preference optimization (SPO) that discards the propagation strategy and allows fine-grained image details to be assessed.

### 关键设计

1. **核心模块**:
   - 做什么：解决上述研究问题的关键技术组件
   - 核心思路：详见论文方法章节
   - 设计动机：针对现有方法的痛点设计

### 损失函数 / 训练策略
需阅读论文全文获取训练细节。

## 实验关键数据

### 主实验
A potential solution to better aesthetics is direct preference optimization (DPO), which has been applied to diffusion models to improve general image quality including prompt alignment and aesthetics.We find that aesthetics can be significantly enhanced by accumulating these improved minor differences.When fine-tuning Stable Diffusion v1.5 and SDXL, SPO yields significant improvements in aesthetics compared with existing DPO methods while not sacrificing image-text alignment compared with vanilla models.


| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 详见论文 | - | - | - | - |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整方法 | 最优 | 基线 |
| 去除核心模块 | 下降 | 验证各组件贡献 |

### 关键发现
- 本文方法在目标任务上有效
- 需阅读全文了解具体消融细节

## 亮点与洞察
- 问题定义和方法设计有针对性
- 需阅读全文进一步评估创新深度

## 局限性 / 可改进方向
- 需阅读全文深入分析
- 泛化性和可扩展性有待评估

## 相关工作与启发
- 在该领域既有方法基础上做出改进

## 评分
- 新颖性: ⭐⭐⭐ 基于摘要初评
- 实验充分度: ⭐⭐⭐ 需读全文评估
- 写作质量: ⭐⭐⭐ 基于摘要初评
- 价值: ⭐⭐⭐ 领域内有贡献
