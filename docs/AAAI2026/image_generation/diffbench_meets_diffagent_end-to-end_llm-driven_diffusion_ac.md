# DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation

**会议**: AAAI 2026  
**arXiv**: [2601.03178](https://arxiv.org/abs/2601.03178)  
**代码**: 未公开  
**领域**: Agent / 图像生成  
**关键词**: LLM Agent, 扩散模型加速, 代码生成, benchmark, 自动化优化  

## 一句话总结
提出DiffBench基准和DiffAgent框架，让LLM自动为扩散模型生成加速代码（如量化、剪枝、蒸馏策略），实现端到端的无人工干预扩散模型优化，首次将LLM Agent应用于扩散模型压缩加速的自动化流程。

## 背景与动机
扩散模型加速是一个活跃的研究领域（步数减少、蒸馏、量化等），但每种方法都需要大量专家知识来实现和调优。LLM Agent在代码生成方面展现出强大能力，能否让Agent自动为扩散模型选择和实现最优加速策略？

## 核心问题
如何构建一个可靠的benchmark来评估LLM在扩散模型加速代码生成上的能力？如何设计Agent使其能自动完成"理解加速需求→选择策略→生成代码→验证效果"的全链路？

## 方法详解
### 整体框架
DiffBench定义扩散模型加速任务的标准化评估体系，DiffAgent使用LLM进行端到端的加速代码生成。

### 关键设计
1. **DiffBench**: 涵盖多种加速策略（量化、步数减少、架构修改等）的标准化benchmark
2. **DiffAgent**: LLM驱动的pipeline——需求分析→策略选择→代码生成→自动验证
3. **端到端自动化**: 从自然语言描述到可执行加速代码的全流程自动化

## 实验关键数据
待查阅full paper。

## 亮点
- **极具前瞻性的方向** — 用AI优化AI的范式
- 首个将LLM Agent应用于扩散模型加速的系统性工作
- DiffBench为后续研究提供了标准化评估基础

## 局限性
- Agent生成的代码质量可能不及人工专家
- 仅基于abstract信息
- 复杂加速策略（如跨层知识蒸馏）可能超出当前LLM的代码生成能力

## 与相关工作的对比
- vs AutoTool(AAAI2026): AutoTool做工具选择，DiffAgent做代码生成——后者更复杂
- vs DICE(AAAI2026): DICE手工设计CFG替代方案，DiffAgent尝试让LLM自动发现类似方案

## 启发与关联
- "用Agent自动优化模型"的范式可以推广到VLM压缩——让Agent自动选择FiCoCo/HACK等方法的最优配置
- 是AutoML在生成模型领域的自然延伸

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统化地将Agent引入扩散模型加速
- 实验充分度: ⭐⭐⭐ 待补充
- 写作质量: ⭐⭐⭐⭐ 方向前瞻
- 价值: ⭐⭐⭐⭐ 对自动化ML有启示价值
