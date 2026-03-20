# MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents

**会议**: NeurIPS 2025 (Datasets & Benchmarks)  
**arXiv**: [2505.20148](https://arxiv.org/abs/2505.20148)  
**代码**: 有  
**领域**: 多模态VLM / Agent / 空间规划  
**关键词**: spatial planning, Minecraft, open-world agent, MLLM benchmark, spatial intelligence  

## 一句话总结
在Minecraft中构建空间规划基准MineAnyBuild——要求AI Agent根据多模态指令生成可执行的建筑方案，包含4000个任务，从空间理解、空间推理、创造力和空间常识四个维度评估MLLM的空间规划能力，揭示现有模型的严重不足。

## 背景与动机
空间规划是空间智能的核心——理解并规划物体在3D空间中的排列。现有空间智能benchmark主要做VQA形式的推理（如"左边是什么？"），但抽象的空间理解和具体的任务执行之间存在gap。MineAnyBuild要求Agent不仅理解空间关系，还要生成可执行的建筑计划。

## 核心问题
MLLM-based agent能否将空间理解能力转化为具体的空间规划和执行能力？

## 方法详解

### 整体框架
- 基于Minecraft的3D建筑任务
- 输入：多模态指令（文本描述+参考图片）
- 输出：可执行的建筑方案（方块放置序列）
- 4000个精心策划的空间规划任务
- 支持通过玩家UGC无限扩展

### 四个评估维度
1. **空间理解**：能否正确解析空间关系描述
2. **空间推理**：能否推断新的空间关系
3. **创造力**：在给定约束下能否生成多样化方案
4. **空间常识**：是否具备基本的物理和几何常识

### 核心发现
- 现有MLLM-based agents在空间规划上存在严重局限
- VQA式的空间理解能力不能直接迁移到空间规划
- 建筑任务需要从理解到执行的完整能力链

## 亮点 / 我学到了什么
- Minecraft作为3D空间规划测试平台的独特价值——离散化简化了评估但保留了核心挑战
- 从"理解"到"执行"的gap是Agent研究的关键瓶颈
- UGC数据的可扩展性设计使benchmark有持续生命力

## 局限性
- Minecraft环境与真实世界3D空间有差距
- 离散方块世界简化了连续空间规划的复杂性

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统评估MLLM空间规划能力的benchmark
- 实验充分度: ⭐⭐⭐⭐ 4000任务×多模型评估
- 写作质量: ⭐⭐⭐⭐ 清晰系统
- 对我的价值: ⭐⭐⭐ Agent空间能力评估方向的参考
