# Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure

**会议**: ICLR2026  
**arXiv**: [2602.08783](https://arxiv.org/abs/2602.08783)  
**代码**: [GitHub](https://github.com/J1mL1/causal-latent-cot)  
**领域**: llm_reasoning  
**关键词**: 隐式思维链, 因果分析, do-干预, 结构因果模型, 可解释性  

## 一句话总结
将隐式CoT建模为结构因果模型(SCM)，通过逐步do-干预分析Coconut和CODI两种范式，发现隐式推理步骤具有异质性因果杠杆、非局部跳跃传播结构、以及输出层早期偏向与表征层晚期提交之间的持续性差距。

## 背景与动机
1. 隐式/连续CoT用内部隐状态替代文本推理步骤，但中间计算难以评估和解释
2. 传统分析方法(步骤编辑/消融)无法直接应用于隐式CoT
3. 现有对隐式CoT的理解主要基于相关性探测，缺乏因果层面的分析
4. 隐式CoT的步骤预算是否均匀贡献？信息如何在步骤间传播？这些问题未被充分探索
5. 输出层面的"提交"(commitment)与表征层面是否同步，尚不清楚
6. 需要一个统一的干预协议来系统研究隐式推理的因果结构

## 方法详解
**因果框架**：将隐式CoT的隐状态序列$H_{1:T}$建模为SCM中的因果变量，通过$\mathrm{do}(h_t \leftarrow \tilde{h}_t)$干预单个步骤状态，观察对下游计算和最终输出的因果影响。

**三个研究问题**：
- **RQ1(必要性与充分性)**：用零干预(将$h_t$置零)测量flip rate——被干预后预测改变的样本比例；用early-stop解码测量正确答案何时可被解码
- **RQ2(传播与路由)**：结合单步干预与early readout，通过teacher-forced KL散度构建步间影响矩阵$W$，可视化为主导影响图(principal influence graph)
- **RQ3(叠加与提交)**：在StrategyQA上通过随机采样获得两模式prompt，用superposition score衡量中间步骤对竞争答案的支持程度

**实验范式**：Coconut(显式隐模式，隐状态回馈)和CODI(自蒸馏压缩离散CoT)；backbone: GPT-2, Llama3-1B, Qwen3-4B-Instruct。

## 实验关键数据
| 发现 | 细节 |
|------|------|
| RQ1: 步骤必要性异质 | flip rate随步骤变化显著，呈非均匀/中间步峰值模式 |
| RQ1: 任务依赖 | GSM8K flip rate 0.1-0.2+，CommonsenseQA普遍<0.1 |
| RQ1: 范式差异 | Coconut比CODI flip rate更高；更强backbone降低flip rate |
| RQ2: 显式CoT近链式 | CoT-SFT局部性≥0.6，相邻步骤间传播为主 |
| RQ2: 隐式CoT跳跃连接 | 隐式模型局部性显著更低、跨度更大，存在大量skip connection |
| RQ3: 提交不同步 | teacher-forced readout显示早期输出提交；probe readout显示中间步骤持续保留竞争假设 |

## 亮点
- 首次对隐式CoT进行因果层面的逐步分析，区分可用性(availability)与稳定性(stability)
- 揭示隐式推理步骤并非均匀"额外深度"，而是具有非局部路由的分阶段功能
- 发现输出层与表征层提交时机不同步——一个重要的设计洞察
- 统一的"干预+读出"协议，适用于不同隐式推理范式的对比

## 局限性 / 可改进方向
- 仅研究了两种隐式CoT范式(Coconut/CODI)，泛化性待验证
- 零干预(置零)作为唯一干预策略，可能引入分布外效应
- RQ3仅在StrategyQA(二元标签)上分析，未扩展到开放式任务
- 未提出具体的训练/解码改进方法，主要停留在分析层面
- 影响图的稀疏化阈值(α=0.1)选择较主观

## 与相关工作的对比
- 相比mechanistic interpretability(Elhage等)关注神经元/注意力头，本文以"步骤"为分析粒度
- 与Wu等(2025)认为连续推理是贪心/单线程的观点互补：probe readout显示确实保留了竞争假设
- 延续了因果干预分析(Pearl, Singh等)在LLM中的应用，首次聚焦隐式CoT

## 评分
- 新颖性: ⭐⭐⭐⭐ (首次因果分析隐式CoT，三个RQ层层递进)
- 实验充分度: ⭐⭐⭐⭐ (多范式/多backbone/多任务，分析全面)
- 写作质量: ⭐⭐⭐⭐⭐ (结构极清晰，现象→机制→本质)
- 价值: ⭐⭐⭐⭐ (对隐式推理设计有重要启发)
