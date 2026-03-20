<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM 推理

**💬 ACL2025** · 共 **11** 篇

**[Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework](aristotle_logical_reasoning.md)**

:   提出 Aristotle 逻辑推理框架，将符号表达式和逻辑规则全面融入 Decompose-Search-Resolve 的每个阶段，通过逻辑分解器、搜索路由器和消解器三大组件实现逻辑完备的推理，在多个逻辑推理基准上以 GPT-4 平均提升 4.5%、GPT-4o 平均提升 5.4% 超越 SOTA。

**[Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs](dcot_diverse_cot_refinement.md)**

:   提出 Diverse Chain of Thought (DCoT) 训练方法，通过在单次推理中生成多条串行推理链实现"推理内自修正"（within-inference refinement），在 1.3B–70B 模型上均超越标准 CoT 基线，尤其在大输出空间任务（数值/抽取型）上提升显著。

**[Dynamic and Generalizable Process Reward Modeling](dgprm_dynamic_process_reward.md)**

:   DG-PRM 提出了一种动态可泛化的过程奖励建模框架，通过奖励树存储多维度评估标准并动态选择步骤相关的奖励信号，用 Pareto 支配估计处理多面奖励，在 PRMBench 上达到 SOTA 且具有优异的跨领域泛化能力。

**[Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering](glore_long_cot_representation.md)**

:   从表示空间角度发现 LLM 将长 CoT 推理编码为一种与普通 CoT 明确区分的通用能力，提出 GLoRE（General Long CoT Reasoning via Representation Engineering）——通过对比推理模式注入和领域特定表示调整来解锁长 CoT 能力，无需训练即可在域内和跨域场景下超越 SFT 方法。

**[Improve Vision Language Model Chain-of-thought Reasoning](improve_vlm_cot_reasoning.md)**

:   通过GPT-4o蒸馏193k CoT数据做SFT + 基于答案正确性构建偏好对做DPO，显著提升VLM的CoT推理能力（LLaVA-Reasoner在8个benchmark上CoT平均提升12.6%），且CoT训练还能反哺直接预测性能。

**[LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning](logicpro_program_guided_reasoning.md)**

:   提出 LogicPro 数据合成方法，利用 LeetCode 算法题和 Python 代码解作为逻辑源，通过"问题生成→代码中间变量提取→程序引导推理生成"三步流水线，从 2360 道算法题合成 540K 高质量文本推理数据，在 BBH27、LogicBench、DROP 等多个 OOD 基准上显著超越现有推理数据集。

**[Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning](mclm_multilingual_test_time_scaling.md)**

:   提出 MCLM（55 语言的竞赛级数学基准），发现三种 test-time scaling 方法（ORM/PRM/Budget Forcing）在英语上提升显著（如 AIME +20 分），但在其他语言上平均仅提升 1.94 分，表明 test-time scaling 的多语言泛化能力严重不足。

**[Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](safe_math_reasoning.md)**

:   提出 Safe 框架，首次利用 Lean 4 形式化语言对 LLM 数学推理的每一步进行回顾性逐步验证，通过自动形式化+自动定理证明检测幻觉，并与前瞻性 PRM 分数融合，在多个数学数据集上取得 SOTA，同时发布包含 30,809 条形式化声明的 FormalStep 基准。

**[SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs](softcot_soft_chain_of_thought.md)**

:   提出 SoftCoT，用一个冻结的小型辅助模型（如 LLaMA-3.2-1B）生成实例特定的"软思维 token"（连续隐状态），通过可训练的投影模块映射到主 LLM 的表示空间作为推理前缀，实现参数高效的连续空间 CoT 推理，避免了全模型微调导致的灾难性遗忘问题。

**[Is That Your Final Answer? Test-Time Scaling Improves Selective Question Answering](test_time_scaling_selective_qa.md)**

:   首次评估 test-time scaling 模型（DeepSeek R1、S1）在选择性问答（允许拒绝回答）场景下的表现，发现增加推理计算不仅提高正确率还提升对正确答案的置信度，提出基于置信度阈值的选择函数和 "Jeopardy Odds" 效用函数来评估非零错误惩罚下的 test-time scaling 性能。

**[TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning](tract_regression_cot.md)**

:   提出 TRACT，一种两阶段回归感知微调方法，将 CoT 推理与回归损失（squared error）结合，用于提升 LLM-as-a-Judge 场景中的数值评分精度，显著优于仅用交叉熵训练或仅用回归损失的现有方案。
