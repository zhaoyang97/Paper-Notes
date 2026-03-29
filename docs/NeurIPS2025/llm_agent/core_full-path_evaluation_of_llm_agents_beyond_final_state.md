# CORE: Full-Path Evaluation of LLM Agents Beyond Final State

**会议**: NeurIPS 2025  
**arXiv**: [2509.20998](https://arxiv.org/abs/2509.20998)  
**代码**: https://github.com/Synkrasis-Labs/CORE  
**领域**: Agent  
**关键词**: Agent评估, 全路径评估, 确定有限自动机, 安全性, 工具调用

## 一句话总结
提出CORE框架：用确定有限自动机（DFA）编码Agent任务的合法工具调用路径，引入5个互补指标（路径正确性、顺序正确性、前缀危险性、有害调用率、效率）从全路径而非仅终态评估Agent行为，揭示了传统终态评估中不可见的安全和效率差异。

## 研究背景与动机

1. **领域现状**：LLM Agent通过函数调用序列执行真实任务（API调用、IoT控制、机器人操控等）。现有benchmark（BFCL等）主要通过终态是否正确来评判Agent——达到期望终态即"成功"。
2. **现有痛点**：终态评估严重不足——(a) 达到正确终态但中间有危险调用的Agent也被判为成功（如机械臂抓对了物体但过程中撞了其他东西）；(b) 冗余/低效的路径与最优路径分数相同；(c) "补偿对"问题——Agent先犯错再修正，终态正确但中间状态可能有害（如资金误转再撤回，但网络故障可能导致卡在错误状态）；(d) 不可观测危害——传感器粒度不够细，终态读数正确但过程中发生了有害操作。
3. **核心矛盾**：在边缘部署（机器人、电网、IoT）中，Agent"如何做到"和"是否做到"同等重要，但现有评估只关注后者。
4. **本文要解决什么？** 建立一个基于执行路径（而非终态）的评估框架，量化Agent行为的正确性、安全性和效率。
5. **切入角度**：将Agent任务建模为DFA——每个prompt诱导一组合法的工具调用路径（golden paths），Agent的实际执行路径与这些路径对比。
6. **核心idea一句话**：用DFA编码任务的合法执行路径空间，用5个互补指标从不同维度评估Agent的全路径质量。

## 方法详解

### 整体框架

CORE框架包含三个层次：
1. **任务建模**：将Agent world $W = (T, Q)$（工具集T，状态集Q）和任务 $\theta = (p, q_0, A)$ 编码为DFA，定义状态转移 $\alpha: Q \times A \to Q$，区分progress/self-loop/harmful三类转移
2. **路径处理**：对原始执行路径做condensation（去除不改变状态的自循环，保留progress和harmful步骤），并定义golden paths（无循环无危害的最优路径）
3. **五维度评估**：PC、PC-KTC、PrefixCrit、HarmRate、Efficiency

### 关键设计

1. **路径正确性 PC（Path Correctness）**:
   - 做什么：衡量Agent凝缩路径与golden path的对齐程度
   - 核心思路：使用归一化Levenshtein距离 $NLD(x,y) = \frac{2 \cdot LD(x,y)}{|x|+|y|+LD(x,y)}$，PC = $1 - NLD$，取所有golden paths中的最大值
   - 优势：容忍不等长路径、编辑操作对应有意义的Agent偏差、提供连续而非二值评分

2. **顺序感知复合指标 PC-KTC（Path Correctness - Kendall's Tau Composite）**:
   - 做什么：在PC基础上加入动作顺序的评估
   - 核心思路：$PC\text{-}KTC = \lambda \cdot PC + (1-\lambda) \cdot \tau^+$，其中 $\tau^+$ 是共有progress token的Kendall-τ顺序相关性的归一化版本
   - 设计动机：Agent可能执行了正确的工具集但顺序颠倒（如先浇水后打开阀门），PC无法捕捉这种问题

3. **前缀危险性 PrefixCrit（Prefix Criticality）**:
   - 做什么：对早期有害调用施加更重惩罚
   - 核心思路：$\text{PrefixCrit}_\beta = 1 - c(\beta, N)\sum_{k=0}^{N-1} m_k \beta^k$，用指数衰减 $\beta^k$ 加权，早期有害步骤权重更大
   - 设计动机：早期错误的因果影响更大——可能使后续所有步骤失效

4. **有害调用率 HarmRate**:
   - 做什么：统计凝缩路径中有害（DFA未定义转移）调用的占比
   - $\text{HarmRate} = \frac{1}{N} \sum m_k$

5. **效率 Efficiency**:
   - 做什么：衡量Agent使用的步骤数相对于最短合法路径的效率
   - $\text{Eff} = \ell^* / n$，其中 $\ell^*$ 是不超过n的最长golden path长度
   - 注意：效率使用原始路径（不凝缩），每个调用都计入成本

6. **Harm-Local Refinement (HLR)**:
   - 做什么：扩展参考路径集以减少因局部错误导致的过度惩罚
   - 在有害位置做局部修复（删除或替换为合法读操作），生成额外的合法参考路径

### 评估设置
- 在多个世界（农业巡逻车、银行助手等）上构建DFA
- 评估GPT-o4-mini、GPT-4o-mini、Qwen3系列(0.6B~8B)、Qwen2.5系列(0.5B~7B)

## 实验关键数据

### 主实验：多模型CORE指标对比

| 模型 | Harmful(总) | Eff | PC | PC-KTC | PrefixCrit | BFCL State% | PC+HLR |
|------|-------------|-----|-----|--------|------------|-------------|--------|
| GPT-o4-mini | 124 | 0.748 | 0.812 | 0.834 | 0.896 | 79.8 | 0.858 |
| GPT-4o-mini | 189 | 0.675 | 0.715 | 0.744 | 0.834 | 71.7 | 0.755 |
| Qwen3-8b | 111 | 0.591 | 0.744 | 0.777 | 0.897 | 80.5 | 0.775 |
| Qwen2.5-7b | 252 | 0.291 | 0.460 | 0.598 | 0.845 | 68.3 | 0.649 |
| Qwen2.5-0.5b | 高 | 低 | 低 | 低 | 低 | 低 | 低 |

### 消融：CORE vs BFCL终态评估

| 对比 | 发现 |
|------|------|
| BFCL State%相近的模型 | CORE指标可能差异很大——如Qwen3-8b和GPT-o4-mini BFCL约80%，但效率差距0.157 |
| 有害调用 | BFCL完全忽略中间有害调用，CORE的HarmRate和PrefixCrit暴露了隐藏风险 |
| HLR修正 | PC+HLR比PC更公平——对因局部错误偏离golden path的Agent给予合理信用 |

### 关键发现
- **终态评估严重高估Agent能力**：BFCL State%=80%的Agent，PC可能只有0.46（Qwen2.5-7b），说明虽然终态正确但过程严重偏离
- **模型规模不等于安全**：Qwen2.5-7b有害调用数(252)远多于更小的Qwen3-0.6b(157)
- **效率维度揭示部署适用性**：GPT-o4-mini不仅最准确，效率也最高（0.748），适合边缘部署
- **顺序很重要**：PC和PC-KTC的差异说明许多Agent虽然调用了正确的工具但顺序不对

## 亮点与洞察
- **DFA建模Agent任务**是将形式验证思维引入Agent评估的优秀尝试——将"正确执行"从模糊的终态判断变为可精确定义的路径匹配问题
- **五个指标形成互补矩阵**：正确性(PC/PC-KTC) × 安全性(PrefixCrit/HarmRate) × 效率(Eff)，每个维度独立有价值
- **补偿对和不可观测危害**的概念化很有启发——在Agent安全研究中可作为重要的失败模式分类

## 局限性 / 可改进方向
- DFA构建目前需要人工定义任务的状态空间和转移函数，扩展到任意任务很困难
- 只在较简单的世界模型上验证（农场巡逻车、银行助手），复杂真实世界任务（如Web Agent、代码Agent）的DFA定义不清晰
- golden paths是预定义的，无法处理创造性的正确解法
- 缺少与其他Agent benchmark（如SWE-Bench、WebArena）的对比
- 可探索从Agent trace自动学习DFA的方法

## 相关工作与启发
- **vs BFCL**: BFCL只检查终态和响应格式，CORE评估全路径。实验表明BFCL中"等价"的Agent在CORE下可能有巨大差异
- **vs SWE-Bench**: SWE-Bench用测试用例（也是终态评估），CORE的路径评估思想可以补充SWE-Bench
- **vs 形式验证**: 借鉴了DFA和状态机概念，但面向LLM Agent评估而非程序验证

## 评分
- 新颖性: ⭐⭐⭐⭐ DFA+五维度指标的框架设计新颖，填补了Agent评估的重要空白
- 实验充分度: ⭐⭐⭐ 多模型评估但世界模型较简单，缺少大规模真实任务验证
- 写作质量: ⭐⭐⭐⭐ 形式化定义清晰，有直觉性的例子辅助理解
- 价值: ⭐⭐⭐⭐ 对Agent安全部署有实际指导意义，尤其是边缘计算场景
