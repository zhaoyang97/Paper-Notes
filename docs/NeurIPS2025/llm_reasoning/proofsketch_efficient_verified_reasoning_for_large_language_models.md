# ProofSketch: Efficient Verified Reasoning for Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.24811](https://arxiv.org/abs/2510.24811)  
**代码**: https://github.com/tanishka66/ProofSketch  
**领域**: LLM推理  
**关键词**: verified reasoning, symbolic closure, sketch generation, token efficiency, logical reasoning

## 一句话总结
提出 ProofSketch 框架，通过符号闭包前向推理+短sketch生成+形式验证的多阶段pipeline，在降低token用量的同时提供逻辑推理的形式化正确性保证。

## 研究背景与动机

1. **领域现状**：CoT prompting和self-consistency等推理方法虽然提高了LLM准确率，但需要生成冗长的推理链，带来高token消耗和延迟。
2. **现有痛点**：(a) LLM经常为简单问题生成过长的推理链（overthinking），浪费计算资源；(b) 中间推理步骤未经验证，即使输出流畅也可能包含逻辑错误。
3. **核心矛盾**：需要在计算预算约束下同时优化准确率和可信度。
4. **本文要解决什么？** 设计一个推理框架，在减少token使用的同时通过形式化验证保证推理正确性。
5. **切入角度**：将LLM生成与符号推理结合——先用前向链接计算理论闭包，再让LLM生成短sketch，最后用闭包验证sketch中的原子声明。
6. **核心idea一句话**：用符号逻辑闭包作为验证oracle，筛选LLM生成的多个短推理sketch中验证覆盖率最高的那个。

## 方法详解

### 整体框架
输入为逻辑理论 $\mathcal{T}$（事实+规则）和True/False/Unknown分类问题 $Q$。Pipeline分四步：(1) 用前向链接计算符号闭包 $C(\mathcal{T})$；(2) 若闭包可直接回答则返回；(3) 否则生成 $K=4$个短sketch，每个包含答案+原子声明集合；(4) 用闭包验证各sketch的原子声明，按字典序评分选出最优sketch。

### 关键设计

1. **符号闭包计算**:
   - 做什么：解析理论为正事实 $F_+$、负事实 $F_-$ 和规则 $R$，用前向链接推导所有可导出事实
   - 核心思路：标准一阶逻辑前向推理，为后续验证提供ground truth
   - 设计动机：如果问题可纯逻辑回答则零成本解决，否则为sketch提供验证基础

2. **自适应sketch生成**:
   - 做什么：生成K=4个候选短sketch，每个在自适应token预算内
   - 核心思路：若查询实体已在闭包中则预算120 tokens，否则160 tokens。温度 $\tau=0.3$。每个sketch输出答案+原子声明（"e is a"或"e is not a"格式）
   - 设计动机：短sketch+多样本比长CoT+单样本更高效，且可并行验证

3. **字典序验证与选择**:
   - 做什么：对每个sketch验证其原子声明，按多目标字典序选出最优
   - 核心思路：优先级顺序为 (1) 完全认证 > (2) 部分验证覆盖率 > (3) token效率 > (4) 与闭包一致性。找到完全认证sketch时early stop
   - 设计动机：确保选出的结果有最大程度的形式化保证

## 实验关键数据

### 主实验
在300条ProofWriter数据上评估三个模型：

| 方法 | R1-Llama-8B Acc | R1-Llama Tokens | Mistral-7B Acc | Mistral Tokens | R1-Qwen-7B Acc | R1-Qwen Tokens |
|------|----------------|-----------------|----------------|----------------|----------------|----------------|
| Zero-shot | 0.37 | 122 | 0.33 | 7 | 0.39 | 10 |
| Short-CoT | 0.52 | 215 | 0.48 | 53 | 0.44 | 49 |
| Long-CoT | 0.52 | 219 | 0.41 | 102 | 0.47 | 101 |
| ProofSketch | **0.68** | 138 | **0.52** | **28** | **0.54** | **30** |

### 消融实验
| 配置 | 认证率 | 说明 |
|------|--------|------|
| ProofSketch (R1-Llama) | 42% | 最高准确率但认证率中等 |
| ProofSketch (Mistral) | **84%** | 最高认证率，token最少 |
| ProofSketch (R1-Qwen) | 42% | 准确率和认证率中等 |
| 所有baseline | 0% | 无验证能力 |

### 关键发现
- ProofSketch在所有模型上均为Pareto最优（准确率-token权衡曲线）
- Mistral-7B上token用量仅28，是Short-CoT的53%，且认证率高达84%
- R1-Llama上准确率提升16个百分点（0.52→0.68）
- 延迟方面表现参差：Mistral上显著降低，但R1-Llama上延迟翻倍（多次生成+验证的开销）

## 亮点与洞察
- **符号+神经混合推理的实用示例**：用符号闭包做"免费"验证，不需要训练额外的verifier模型
- **原子声明的规范化**：将LLM输出约束为标准格式的原子声明（"e is a"），使符号验证成为可能

## 局限性 / 可改进方向
- **实验规模太小**：仅300条ProofWriter数据，模型最大8B，结果说服力有限
- **仅限简单逻辑推理**：依赖一阶逻辑前向链接，无法处理数学证明、常识推理等复杂场景
- **延迟问题**：多次生成+验证在某些模型上带来显著延迟增加
- **认证率不稳定**：42%-84%的认证率范围很大，取决于模型和问题类型
- 未与更多验证方法（如LLM-as-verifier, tree search）对比

## 相关工作与启发
- **vs CoT/Self-Consistency**: ProofSketch追求更短的推理+形式验证，而非更长的推理+统计一致性
- **vs Sketch-of-Thought**: 都生成短sketch，但ProofSketch增加了符号验证层
- **vs atomic reasoning (zhang2025)**: 受原子声明分解的启发，但增加了闭包验证

## 评分
- 新颖性: ⭐⭐⭐ 将符号闭包与sketch生成结合的idea有意思，但技术贡献有限
- 实验充分度: ⭐⭐ 实验规模太小（300条），仅一个数据集，结论可信度不足
- 写作质量: ⭐⭐⭐ 结构清晰但深度不够，是workshop级别的论文
- 价值: ⭐⭐⭐ 方向有趣但验证不充分，需要更大规模的实验支撑
