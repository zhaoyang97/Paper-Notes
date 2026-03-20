# InnoGym: Benchmarking the Innovation Potential of AI Agents

**会议**: ICLR 2026  
**arXiv**: [2512.01822](https://arxiv.org/abs/2512.01822)  
**代码**: https://github.com/zjunlp/igym  
**领域**: AI评估 / Agent  
**关键词**: innovation benchmark, AI agent, novelty evaluation, agent-as-judge, research automation

## 一句话总结
提出 InnoGym 框架和 iBench/iGym 基准，首次从"创新性"维度评估 AI Agent——不仅衡量正确性还衡量方法论新颖性，发现当前 Agent 能产生新颖想法但无法转化为性能提升（平均归一化增益 -0.45）。

## 研究背景与动机

1. **领域现状**：现有 Agent benchmark（SWE-bench, MATH 等）仅衡量任务完成度/正确性，忽略方法论新颖性。
2. **现有痛点**：真正的创新需要"性能超越 SOTA + 方法论不同"——这一维度在评估中缺失。
3. **核心idea一句话**：用 (P,S,V,D) 形式化创新，$V = C(s) \cdot R(s)$ 衡量性能增益 G，$D$ = Agent-as-judge 按 6 维度评估新颖性 N。

## 方法详解

### 关键设计
1. **Performance Gain**: $G(s) = V(s) - V^*_{known}$
2. **Novelty**: $N(s) = \min$ distance to known solutions（6 维度评分 0-4）
3. **iBench**: 10 个 SOTA 可超越的任务（Circle Packing、对抗鲁棒性等）
4. **三种 Agent**: MLAB, CodeAct, AIDE

## 实验关键数据

### 主实验

| Agent | 平均归一化增益 | 平均新颖性 |
|-------|-------------|---------|
| MLAB | -0.45 | 56.55 |
| CodeAct | -0.69 | - |
| AIDE | -0.64 | - |

所有 Agent 性能增益为负——没有超越已知 SOTA。

### 关键发现
- **创新瓶颈在执行而非想法**：Agent 能产生新颖方案（novelty 56.55）但无法正确实现
- 温度折中：0.5-0.75 最优
- 显式"创新提示"提升新颖性（35→58）但降低性能
- 高新颖性往往伴随最差性能

## 亮点与洞察
- **创新 ≠ 新颖**：创新需要新颖性 AND 性能提升
- Agent-as-judge 的 6 维度新颖性评估方法具有可操作性

## 局限性 / 可改进方向
- 仅 10 个任务，覆盖面有限
- 每任务 12 小时计算限制

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次从创新维度评估 Agent
- 实验充分度: ⭐⭐⭐⭐ 多 Agent、多任务、温度分析
- 写作质量: ⭐⭐⭐⭐ 形式化框架优雅
- 价值: ⭐⭐⭐⭐ 揭示了 Agent 的"创新鸿沟"
