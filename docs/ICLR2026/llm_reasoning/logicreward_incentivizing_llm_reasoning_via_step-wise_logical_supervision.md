# LogicReward: Incentivizing LLM Reasoning via Step-Wise Logical Supervision

**会议**: ICLR2026  
**arXiv**: [2512.18196](https://arxiv.org/abs/2512.18196)  
**代码**: [项目主页](https://llm-symbol.github.io/LogicReward)  
**领域**: llm_reasoning  
**关键词**: 逻辑推理, 定理证明器, 步骤级奖励, 自动形式化, 软统一  

## 一句话总结
提出LogicReward奖励函数，用Isabelle定理证明器做步骤级逻辑正确性验证，结合Autoformalization with Soft Unification减少自然语言歧义，训练出的8B模型在NLI和逻辑推理任务上超越GPT-4o 11.6%和o4-mini 2%。

## 背景与动机
1. 现有训练方法主要依赖结果反馈(outcome-based)，可能产生推理错误但答案正确的情况
2. 过程级监督(PRM等)仍缺乏逻辑正确性的形式化保证
3. 概率性反馈(token概率/学习的奖励模型)本质上非确定性，无法可靠检测逻辑错误
4. 符号验证方法目前主要限于结构化领域(数学/编程)，NLI领域缺失
5. 自然语言歧义性和隐含假设使形式化验证困难(如Dad≠Father但语义等同)
6. 高风险场景(医疗/法律)需要严格的逻辑一致性保证

## 方法详解
**LogicReward流水线**：
1. **Rollout**：用Qwen3-8B和GPT-4o生成8个候选响应，每个包含步骤序列$\{s_1,...,s_m\}$和最终答案A
2. **每步分解**：$s_i = (P_r, I)$，$P_r$=使用的前提，$I$=做出的推断

**三个验证维度**：
- **Premise Validity**：检查引用前提$P_r$是否在给定前提$P$中有Grounding(余弦相似度)
- **Logic Validity**：用Isabelle定理证明器验证推断$I$的逻辑有效性。语法正确+逻辑正确→1；语法正确+逻辑错误→0；语法错误→退回token概率置信度
- **Outcome Validity**：最终答案是否匹配ground truth

**Autoformalization with Soft Unification**：提示LLM在每个推理步骤中补充隐含但有效的假设(如同义词映射、隐含信息)，减少歧义提高形式化成功率。

**Refinement**：对Isabelle判定无效的推理步骤，用错误信息迭代refine Soft Unification过程。

**训练**：$LogicScore = avg(w_1 \cdot ReasoningValidity, w_2 \cdot OutcomeValidity)$，选最高分做SFT、最高/最低分对做DPO。

## 实验关键数据
| 模型 | M-LogiEval | FOLIO | ProverQA | LogiQA | 8任务平均 |
|------|-----------|-------|----------|--------|----------|
| GPT-4o | 68.0 | 63.5 | 78.4 | 69.3 | 73.9 |
| o4-mini | 82.0 | 80.8 | 78.8 | 65.7 | 83.5 |
| DeepSeek-R1-8B | 64.8 | 57.3 | 59.2 | 53.2 | 68.6 |
| **LogicReward-Qwen3-8B** | **82.0** | **79.5** | **81.2** | **72.3** | **85.5** |

- 8B模型超越GPT-4o(+11.6%)、GPT-4.1(+10%)、DeepSeek-R1-8B(+16.9%)
- 泛化到数学(CommonsenseQA/GSM8K)和常识推理任务
- 无ground-truth标签时仍可作为有效奖励信号(仅用ReasoningValidity)

## 亮点
- 首次将定理证明器引入NLI领域的步骤级奖励——跨越了符号验证从结构化→非结构化的鸿沟
- Soft Unification巧妙处理自然语言歧义，是让定理证明器在NL推理中可用的关键
- 8B模型超越o4-mini——证明逻辑正确的训练数据比模型规模更重要
- 无标签场景下仍有效——ReasoningValidity本身就是有价值的信号
- Refinement机制利用Isabelle错误信息迭代改进，闭环设计

## 局限性 / 可改进方向
- Isabelle形式化失败时退回token概率，失去了形式化保证
- 仅在NLI/逻辑推理任务上训练和主评估，数学/常识仅作泛化验证
- Soft Unification依赖LLM的能力，可能引入新的错误
- 训练数据仅~6000实例，扩展性待验证
- 形式化流水线成本高(需要Isabelle运行环境+多次LLM调用)
- $w_1=w_2=0.5$为简单等权，未探索最优权重

## 与相关工作的对比
- vs PRM(Lightman等): LogicReward提供确定性逻辑保证而非概率评估
- vs LINC/Logic-LM: 这些方法在推理时用prover，LogicReward在训练时用prover构建奖励
- vs DeepSeek-R1: 仅用outcome reward激励长推理，LogicReward额外监督推理过程的逻辑有效性
- vs SymbCoT/Aristotle: 让LLM扮演symbolic prover，LogicReward使用实际定理证明器

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (定理证明器+NLI训练奖励的首次结合)
- 实验充分度: ⭐⭐⭐⭐ (8 benchmark+多基线+泛化+无标签实验)
- 写作质量: ⭐⭐⭐⭐ (方法描述清晰，公式完整)
- 价值: ⭐⭐⭐⭐⭐ (8B超o4-mini，实用性和理论意义兼具)
