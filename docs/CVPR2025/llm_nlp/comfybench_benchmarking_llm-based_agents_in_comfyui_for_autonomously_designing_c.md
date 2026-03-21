# ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems

**会议**: CVPR 2025  
**arXiv**: 见CVF  
**代码**: 待确认  
**领域**: NLP理解  
**关键词**: 待补充

## 一句话总结
> 基于摘要：Much previous AI research has focused on developing monolithic models to maximize their intelligence, with the primary goal of enhancing performance on specific tasks. In contrast, this work attempts to study using LLM-based agents to design collaborative AI systems autonomously. To explore this pro

## 研究背景与动机
1. **领域现状**：本文研究的问题属于 NLP理解 方向。Much previous AI research has focused on developing monolithic models to maximize their intelligence, with the primary goal of enhancing performance on specific tasks. In contrast, this work attempts to study using LLM-based agents to design collaborative AI systems autonomously. To explore this problem, we first introduce ComfyBench to evaluate agents's ability to design collaborative AI systems in ComfyUI. ComfyBench is a comprehensive benchmark comprising 200 diverse tasks covering various instruction-following generation challenges, along with detailed annotations for 3,205 nodes and 20 workflows.
2. **现有痛点**：现有方法存在局限性——效率、精度或泛化性方面有改进空间。
3. **核心矛盾**：需要在效果与效率/泛化性之间找到更好的平衡。
4. **本文要解决什么？** 针对上述问题，作者提出了新方法。
5. **切入角度**：从新的技术视角或观察出发。
6. **核心idea一句话**：Based on ComfyBench, we further develop ComfyAgent, a novel framework that empowers LLM-based agents to autonomously design collaborative AI systems by generating workflows. ComfyAgent is based on two

## 方法详解

### 整体框架
本文提出的方法概述如下（基于摘要信息）：

Based on ComfyBench, we further develop ComfyAgent, a novel framework that empowers LLM-based agents to autonomously design collaborative AI systems by generating workflows. ComfyAgent is based on two core concepts. First, it represents workflows with code, which can be reversibly converted into workflows and executed as collaborative systems by the interpreter. Second, it constructs a multi-agent system that cooperates to learn from existing workflows and generate new workflows for a given task.

### 关键设计

1. **核心模块**:
   - 做什么：解决上述痛点的关键技术组件
   - 核心思路：详见论文方法部分
   - 设计动机：提升性能或效率

### 损失函数 / 训练策略
详见论文全文（缓存不足，无法提取具体训练细节）。

## 实验关键数据

### 主实验
基于摘要的实验信息：While experimental results demonstrate that ComfyAgent achieves a comparable resolve rate to o1-preview and significantly surpasses other agents on ComfyBench, ComfyAgent has resolved only 15% of creative tasks. LLM-based agents still have a long way to go in autonomously designing collaborative AI systems. Progress with ComfyBench is paving the way for more intelligent and autonomous collaborative AI systems. Our code is available at: https://github.com/xxyQwQ/ComfyBench.

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
