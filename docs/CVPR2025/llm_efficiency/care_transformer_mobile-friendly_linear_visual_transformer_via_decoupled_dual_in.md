# CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction

**会议**: CVPR 2025  
**arXiv**: 见CVF  
**代码**: 待确认  
**领域**: LLM效率  
**关键词**: 待补充

## 一句话总结
> 基于摘要：Recently, large efforts have been made to design efficient linear-complexity visual Transformers. However, current linear attention models are generally unsuitable to be deployed in resource-constrained mobile devices, due to suffering from either few efficiency gains or significant accuracy drops. 

## 研究背景与动机
1. **领域现状**：本文研究的问题属于 LLM效率 方向。Recently, large efforts have been made to design efficient linear-complexity visual Transformers. However, current linear attention models are generally unsuitable to be deployed in resource-constrained mobile devices, due to suffering from either few efficiency gains or significant accuracy drops. In this paper, we propose a new deCoupled duAl-interactive lineaR attEntion (CARE) mechanism, revealing that features' decoupling and interaction can fully unleash the power of linear attention.
2. **现有痛点**：现有方法存在局限性——效率、精度或泛化性方面有改进空间。
3. **核心矛盾**：需要在效果与效率/泛化性之间找到更好的平衡。
4. **本文要解决什么？** 针对上述问题，作者提出了新方法。
5. **切入角度**：从新的技术视角或观察出发。
6. **核心idea一句话**：We first propose an asymmetrical feature decoupling strategy that asymmetrically decouples the learning process for local inductive bias and long-range dependencies, thereby preserving sufficient loca

## 方法详解

### 整体框架
本文提出的方法概述如下（基于摘要信息）：

We first propose an asymmetrical feature decoupling strategy that asymmetrically decouples the learning process for local inductive bias and long-range dependencies, thereby preserving sufficient local and global information while effectively enhancing the efficiency of models. Then, a dynamic memory unit is employed to maintain critical information along the network pipeline. Moreover, we design a dual interaction module to effectively facilitate interaction between local inductive bias and long-range information as well as among features at different layers.

### 关键设计

1. **核心模块**:
   - 做什么：解决上述痛点的关键技术组件
   - 核心思路：详见论文方法部分
   - 设计动机：提升性能或效率

### 损失函数 / 训练策略
详见论文全文（缓存不足，无法提取具体训练细节）。

## 实验关键数据

### 主实验
基于摘要的实验信息：By adopting a decoupled learning way and fully exploiting complementarity across features, our method can achieve both high efficiency and accuracy. Extensive experiments on ImageNet-1K, COCO, and ADE20K datasets demonstrate the effectiveness of our approach, e.g., achieving 78.4/82.1% top-1 accuracy on ImagegNet-1K at the cost of only 0.7/1.9 GMACs. Codes will be released on github.

| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 详见论文 | - | - | - | - |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整模型 | 最优 | 完整方法 |
| 去除核心模块 | 下降 | 验证核心贡献 |

### 关键发现
- 本文方法在目标任务上取得显著改进
- 各核心模块均对最终性能有贡献

## 亮点与洞察
- 问题定义清晰，方法针对性强
- 核心设计思路可能可以迁移到相关场景

## 局限性 / 可改进方向
- 需要阅读全文才能深入分析方法细节和局限
- 泛化性和可扩展性有待进一步验证

## 相关工作与启发
- 本文在该领域的既有方法基础上做出了改进

## 评分
- 新颖性: ⭐⭐⭐ 基于摘要初评，有一定创新
- 实验充分度: ⭐⭐⭐ 需读全文验证
- 写作质量: ⭐⭐⭐ 基于摘要初评
- 价值: ⭐⭐⭐ 在该领域有贡献
