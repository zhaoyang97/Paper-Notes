# coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation

**会议**: CVPR 2025  
**arXiv**: [2603.12829](https://arxiv.org/abs/2603.12829)  
**代码**: 待确认  
**领域**: 图像生成 / 组合式生成  
**关键词**: 多智能体, 组合生成, 布局规划, 文本到图像, 闭环推理

## 一句话总结

提出 coDrawAgents，由 Interpreter、Planner、Checker、Painter 四个专家 agent 组成的交互式多智能体对话框架，通过分而治之的增量布局规划、视觉上下文感知推理和显式错误纠正，在 GenEval 上达到 0.94（SOTA）、DPG-Bench 上 85.17（SOTA）。

## 研究背景与动机

1. **领域现状**：T2I 模型在复杂场景中仍难以正确组合多个对象并保持属性一致性。
2. **现有痛点**：(1) 全局布局规划面临 O(n^2) 关系复杂度；(2) 多数方法无视觉反馈做空间推理；(3) 扩散管线早期确定粗略结构后难以纠正。
3. **核心矛盾**：单 agent 能力瓶颈明显；现有多 agent 多为固定管线缺乏闭环推理。
4. **本文要解决什么**：通过多 agent 闭环对话解决复杂场景的组合式图像生成。
5. **切入角度**：将生成过程分解为四个专业角色，动态对话而非顺序流水线。
6. **核心idea一句话**：组合式图像生成需要"规划-检查-渲染"的闭环协作。

## 方法详解

### 整体框架

Interpreter 决定模式（layout-free/layout-aware）-> layout-aware 中：解析文本为结构化对象描述 -> 按语义优先级分组 -> 每组：Planner 增量规划 -> Checker 验证修正 -> Painter 渲染 canvas -> 循环到下组。

### 关键设计

1. **Interpreter**
   - 做什么：决定生成模式，将复杂 prompt 分解为结构化对象描述
   - 核心思路：LLM + CoT 三步（识别、排序分组、丰富属性）
   - 设计动机：简单 prompt 直接生成，自适应选择避免开销

2. **Planner + 可视化思维链（VCoT）**
   - 做什么：增量规划当前优先级对象的布局
   - 核心思路：GPT-5 做多模态 VCoT，输入全局文本+对象描述+历史布局+部分画面+对象定位
   - 三步推理：Canvas 状态分析 -> 上下文感知规划 -> 物理约束执行
   - 设计动机：分治降低复杂度，视觉上下文消除"凭空想象"

3. **Checker**
   - 做什么：两阶段检查修正（当前提案检查 + 全历史回溯）
   - 核心思路：对象级（尺寸、比例）+ 全局级（位置、遮挡）检查，回溯修正历史
   - 设计动机：显式纠正弥补扩散模型"一旦决定难改"的缺陷

4. **Painter**
   - 做什么：layout-free 调 Flux，layout-aware 调 3DIS
   - 设计为 plug-and-play，可替换任意 T2I/L2I 模型

### 损失函数 / 训练策略

Training-free 框架。Planner/Checker 用 GPT-5 推理。

## 实验关键数据

### 主实验

GenEval 对比：

| 模型 | Overall |
|------|---------|
| DALL-E 3 | 0.67 |
| FLUX.1-dev | 0.67 |
| SD3-Medium | 0.74 |
| GPT Image 1 | 0.84 |
| coDrawAgents | **0.94** |

DPG-Bench: Overall 85.17（SOTA），Relation 92.92 最佳。

### 消融实验

| 配置 | Overall |
|------|---------|
| Layout-free only | 77.60 |
| + Layout-aware | 82.61 |
| + Visual context | 84.51 |
| + Checker | **85.17** |

Agent 效率：平均每图仅 Planner 1.52 次、Checker 1.62 次，远少于对象数 2.79。

### 关键发现

- GenEval 0.94 比 GPT Image 1 高 10 个百分点
- Position 从 FLUX 0.20 到 0.95，增量规划+检查对空间定位极有效
- Counting 从 0.79 到 0.94，分治策略有效解决数量不准确
- 每个组件都有独立贡献（分治 +5pp、视觉上下文 +2pp、Checker +0.7pp）

## 亮点与洞察

- 闭环多 agent 对话，Checker 可回溯修正历史迭代错误
- 视觉上下文感知规划避免"凭空想象"
- 语义优先级分组高效减少 agent 调用次数
- GenEval 上 0.94 的成绩令人印象深刻

## 局限性 / 可改进方向

- 依赖 GPT-5，计算开销高且受幻觉影响
- Painter 受底层模型限制
- 仅限 2D 生成

## 相关工作与启发

- GoT 做一次性全局推理，coDrawAgents 做增量视觉感知推理
- 分治+视觉 CoT 范式可推广到视频/3D

## 评分

- **新颖性**: ⭐⭐⭐⭐ 闭环多 agent 对话框架是有意义创新
- **实验充分度**: ⭐⭐⭐⭐ 双基准全面评估，消融充分
- **写作质量**: ⭐⭐⭐⭐ 框架图清晰
- **价值**: ⭐⭐⭐⭐ GenEval 0.94 SOTA 具系统级意义
