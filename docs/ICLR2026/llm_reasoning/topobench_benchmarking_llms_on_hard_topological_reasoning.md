# TopoBench: Benchmarking LLMs on Hard Topological Reasoning

**会议**: ICLR2026  
**arXiv**: [2603.12133](https://arxiv.org/abs/2603.12133)  
**代码**: [GitHub](https://github.com/mayug/topobench-benchmark)  
**领域**: llm_reasoning  
**关键词**: benchmark, topological reasoning, spatial reasoning, puzzle, error diagnosis, causal intervention  

## 一句话总结
构建TopoBench基准(6类拓扑谜题×3难度)评估LLM的全局空间推理能力，发现前沿模型hard tier仅解决<24%，并通过因果干预实验发现错误频率不等于因果影响——低频的约束遗忘比高频的重复推理更具破坏性。

## 背景与动机
1. LLM在代数/符号推理上表现强劲，但在需要维护全局空间不变量（连通性、闭环、对称性）的任务上能力不足
2. 现有谜题/推理基准多测试局部模式匹配或单元格级运算，不要求跨网格的全局约束维护
3. 拓扑约束在电路布局、路径规划、分子结构分析等实际应用中普遍存在
4. 现有评估仅报告准确率，无法区分模型失败源于推理本身还是空间信息提取/表示的局限
5. 需要将观察性错误分类与因果验证结合的诊断方法

## 方法
**TopoBench基准**: 6类谜题(FlowFree-路径连通, Bridges-网络连通, Loopy-闭环, Galaxies-旋转对称, Undead-反射可见性, Pattern-连续性)，每类3个难度(easy/medium/hard, 5×5→10×10)，900个实例，配专用验证器。

**诊断流程**:
1. **观察阶段**: 用LLM-as-Judge(GPT-5-mini)标注750条CoT推理链，按11类错误分类
2. **因果干预阶段**: 将4种错误模式注入部分金标准解题路径，测量下游准确率变化(每条件300题)

**4种干预错误**: RR(重复推理)、PC(过早承诺)、STF(状态追踪失败)、CF(约束遗忘)

**缓解策略**: cell-aligned网格表示、tool-augmented约束查询、提示级规划引导

## 实验
| 模型 | Easy Avg | Medium Avg | Hard Avg |
|------|:--------:|:----------:|:--------:|
| GPT-5-mini-high | **0.71** | **0.44** | **0.24** |
| Gemini-3-Flash | 0.60 | 0.35 | 0.09 |
| DeepSeek V3.2 | 0.58 | 0.37 | 0.10 |
| Qwen3-235B | 0.31 | 0.12 | — |
| Qwen3-32B | 0.07 | — | — |

**关键发现**: (1) Galaxies和Loopy在medium/hard上几乎所有模型准确率为0，全局不变量是最难的约束类型; (2) 约束遗忘(CF)仅在4%失败trace中出现，但注入后准确率下降~11pp，因果效应最大; (3) 重复推理(RR)在33%trace中出现，但注入后对准确率无显著影响——是搜索的良性副产品; (4) 工具增强提供结构化约束信息可提升Bridges hard 10%，但提供ASCII网格反而降低准确率; (5) 瓶颈在于从空间表示中提取约束，而非对约束进行推理。

## 亮点
- 错误频率≠因果影响的发现极具洞察力，挑战了常见假设
- 因果干预实验设计严谨：在金标准解题路径上注入控制变量
- 缓解策略实验区分了"空间表示解析"vs"约束推理"的瓶颈
- 6类谜题覆盖不同拓扑约束类型，设计全面

## 局限
- 仅在DeepSeek V3.2上做因果干预分析(其他模型不暴露完整CoT)
- 谜题虽控制良好但与真实工程任务有差距
- ASCII文本输入限制了多模态模型的潜力（虽有初步探索）
- hard tier大部分近零，区分度不足——可能需要更细粒度的难度梯度

## 相关工作
- 推理基准: GSM8K/MATH (代数), ARC (抽象), SATBench (逻辑), Sudoku-Bench (Latin square)
- 错误诊断: GridPuzzle (Tyagi et al. 2024) 观察性错误分类; LLM-as-judge (Liu et al. 2023)
- 空间推理: Othello-GPT (Li et al. 2023) 状态追踪; VGRP-Bench, Enigmata 视觉网格评估
- 工具增强: ReAct (Yao et al. 2023), Toolformer (Schick et al. 2023)

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (因果干预+拓扑推理诊断组合独特)
- 实验充分度: ⭐⭐⭐⭐⭐ (9模型+6谜题+3难度+因果实验+缓解策略)
- 写作质量: ⭐⭐⭐⭐⭐ (结构清晰，分析深入)
- 价值: ⭐⭐⭐⭐ (揭示LLM空间推理的根本瓶颈)
