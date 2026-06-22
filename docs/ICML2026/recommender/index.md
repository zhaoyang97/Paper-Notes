---
title: >-
  ICML2026 推荐系统论文汇总 · 11篇论文解读
description: >-
  11篇ICML2026的推荐系统方向论文解读，涵盖推荐系统、LLM、对抗鲁棒、自监督学习、个性化生成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "推荐系统"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "对抗鲁棒"
  - "自监督学习"
  - "个性化生成"
item_list:
  - u: "a_paired_testing_protocol_for_batch-conditioned_refusal_robustness_in_llm_servin/"
    t: "A Paired Testing Protocol for Batch-Conditioned Refusal Robustness in LLM Serving"
  - u: "can_recommender_systems_teach_themselves_a_recursive_self-improving_framework_wi/"
    t: "Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control"
  - u: "gcib_graph_contrastive_information_bottleneck_for_multi-behavior_recommendation/"
    t: "GCIB: Graph Contrastive Information Bottleneck for Multi-Behavior Recommendation"
  - u: "incentivized_exploration_with_stochastic_covariates_a_two-stage_mechanism_design/"
    t: "Incentivized Exploration with Stochastic Covariates: A Two-Stage Mechanism Design for Recommender System"
  - u: "learning_design_skills_as_memory_policies_for_agentic_photonic_inverse_design/"
    t: "Learning Design Skills as Memory Policies for Agentic Photonic Inverse Design"
  - u: "position_neglecting_the_sustainability_of_ai_is_fuelling_a_global_ai_arms_race/"
    t: "Position: Neglecting the Sustainability of AI is Fuelling a Global AI Arms Race"
  - u: "position_stop_preaching_and_start_practising_data_frugality_for_responsible_deve/"
    t: "Position: Stop Preaching and Start Practising Data Frugality for Responsible Development of AI"
  - u: "prompts_for_public-sector_llms_should_be_governed_as_commons/"
    t: "Prompts for Public-Sector LLMs Should Be Governed as Commons"
  - u: "rethinking_contrastive_learning_for_graph_collaborative_filtering_limitations_an/"
    t: "Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy"
  - u: "rgmem_renormalization_group-inspired_memory_evolution_for_language_agents/"
    t: "RGMem: Renormalization Group-Inspired Memory Evolution for Language Agents"
  - u: "t-pop_test-time_personalization_with_online_preference_feedback/"
    t: "T-POP: Test-Time Personalization with Online Preference Feedback"
item_total: 11
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎁 推荐系统

**🧪 ICML2026** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (24)](../../ICLR2026/recommender/index.md) · [💬 ACL2026 (22)](../../ACL2026/recommender/index.md) · [🤖 AAAI2026 (27)](../../AAAI2026/recommender/index.md) · [🧠 NeurIPS2025 (24)](../../NeurIPS2025/recommender/index.md) · [🧪 ICML2025 (17)](../../ICML2025/recommender/index.md) · [💬 ACL2025 (7)](../../ACL2025/recommender/index.md)

🔥 **高频主题：** 推荐系统 ×3

**[A Paired Testing Protocol for Batch-Conditioned Refusal Robustness in LLM Serving](a_paired_testing_protocol_for_batch-conditioned_refusal_robustness_in_llm_servin.md)**

:   本文把 LLM serving 中的 batch 条件当作安全评测的处理变量，提出安全提示与能力控制成对比较、人工/打分器校正、跨模型扩展、连续批处理组合和 batch-invariant kernel 消融组成的测试协议，结论是拒绝翻转真实存在但低频、模型特异且依赖具体服务栈。

**[Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control](can_recommender_systems_teach_themselves_a_recursive_self-improving_framework_wi.md)**

:   RSIR 让序列推荐模型用自身预测能力生成新的合成用户交互序列、再训练一个新模型，并用基于排名的"保真度检查"过滤掉偏离用户偏好流形的样本，防止 self-consuming model collapse；在 4 个数据集 × 3 个主流 backbone 上稳定提升 NDCG/Recall 4–11%，并理论上证明该过程等价于沿用户偏好流形切空间的隐式正则化。

**[GCIB: Graph Contrastive Information Bottleneck for Multi-Behavior Recommendation](gcib_graph_contrastive_information_bottleneck_for_multi-behavior_recommendation.md)**

:   GCIB 用"图信息瓶颈 + 跨行为对比学习"双管齐下，先在结构层把辅助行为图里与目标任务无关的边剪掉（最大化与目标行为的互信息、用 HSIC 替代项最小化与原始辅助图的互信息），再在特征层把去噪后的辅助表示和稀疏的目标表示做 InfoNCE 对齐，从而在四个多行为推荐基准上把 HR@10 / NDCG@10 相对最佳 baseline 再推高 7%–40%。

**[Incentivized Exploration with Stochastic Covariates: A Two-Stage Mechanism Design for Recommender System](incentivized_exploration_with_stochastic_covariates_a_two-stage_mechanism_design.md)**

:   RCB 把推荐系统里的"探索-利用"和"用户激励兼容"打包成一个动态贝叶斯激励兼容（DBIC）约束下的上下文 bandit 问题，提出冷启动 + IPGS 两阶段算法，在随机用户协变量场景下证明 $\tilde{O}(\sqrt{KdT})$ regret、可插入任意 offline learning oracle，并量化"激励价格"——冷启动样本量随 $\epsilon$ 收紧呈 $1/\epsilon^2$ 增长。

**[Learning Design Skills as Memory Policies for Agentic Photonic Inverse Design](learning_design_skills_as_memory_policies_for_agentic_photonic_inverse_design.md)**

:   SkillPCF 把光子晶体光纤（PCF）的反向设计重塑为"记忆策略学习"问题：用 PPO 训练的控制器在每个轨迹片段从可演化技能库里挑 Top-K 个 memory 操作，执行器把它们落到轨迹记忆里，再用 MEEP 电磁仿真奖励同时优化控制器与技能库本身，在多 LLM 后端和经典优化基线上都拿到更好的设计成功率与仿真预算权衡。

**[Position: Neglecting the Sustainability of AI is Fuelling a Global AI Arms Race](position_neglecting_the_sustainability_of_ai_is_fuelling_a_global_ai_arms_race.md)**

:   这篇 position paper 借 Karl Marx 的"基础-上层建筑"框架，主张当下"sustainable AI"的讨论被环境维度独占而忽略了经济与社会维度，呼吁同时拉高**气候意识**与**资源意识**两条轴，并提出 CARAML 五层行动框架（个人 / 社区 / 工业 / 政府 / 全球）以抑制正在升级的"全球 AI 军备竞赛"。

**[Position: Stop Preaching and Start Practising Data Frugality for Responsible Development of AI](position_stop_preaching_and_start_practising_data_frugality_for_responsible_deve.md)**

:   这篇 position paper 指出 ML 社区在"数据节俭"(data frugality)上长期"只说不做"——大家口头承认 coreset 能省能耗，却几乎没人真去汇报能耗和碳排放，并以 ImageNet-1K 为例算出下游训练 + 存储约 5.82 GWh / 2589 tCO2e 的保守下限，呼吁把数据节俭从口号变成可度量、可执行、可奖励的工程实践。

**[Prompts for Public-Sector LLMs Should Be Governed as Commons](prompts_for_public-sector_llms_should_be_governed_as_commons.md)**

:   这是一篇 position paper：作者主张公共部门用的 LLM 提示词应当像开源 commons 一样被版本化、有出处、可审计、可否决，并用一座北美城市的 443 条社区提示词（增强到 3,317 条）跑了一个含五种治理状态的 pilot benchmark，给出可证伪的三个预测——治理化提示能改变输出分布、提升可审计性、缩短故障修复时延。

**[Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy](rethinking_contrastive_learning_for_graph_collaborative_filtering_limitations_an.md)**

:   作者把 LightGCN 的前向预测打开成"多跳邻居对的可学习权重之和"，发现 Sampled Softmax 损失只按物品侧邻居的结构相似度来加权、且对 UU/II/UI/IU 四类邻居对一视同仁，于是提出 NT-SSM——把用户侧结构相似度也接入梯度、并按邻居对类型分别校准加权策略，在四个数据集和多种 GCF 主干上稳定优于 SSM。

**[RGMem: Renormalization Group-Inspired Memory Evolution for Language Agents](rgmem_renormalization_group-inspired_memory_evolution_for_language_agents.md)**

:   RGMem 借统计物理里的重整化群思想，把语言 agent 的长期对话记忆建模成"事件层 → 关系层 → 概念层"的多尺度系统，通过阈值触发的非线性算子把零散对话粗粒化成稳定的用户画像，从而打破"稳定 vs 可塑"权衡。

**[T-POP: Test-Time Personalization with Online Preference Feedback](t-pop_test-time_personalization_with_online_preference_feedback.md)**

:   T-POP 把"测试时对齐"和"神经决斗赌博机"拼在一起，在不动 LLM 参数的前提下，用每轮一对回复的在线偏好反馈在线学习个性化奖励函数，从而解决新用户个性化的冷启动问题。
