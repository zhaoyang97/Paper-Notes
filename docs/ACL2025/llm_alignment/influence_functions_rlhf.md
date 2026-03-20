# Understanding Impact of Human Feedback via Influence Functions

**会议**: ACL 2025
**arXiv**: [2501.05790](https://arxiv.org/abs/2501.05790)
**代码**: [https://github.com/mintaywon/IF_RLHF](https://github.com/mintaywon/IF_RLHF)
**机构**: KAIST, Columbia University
**领域**: LLM对齐 / RLHF 数据质量分析
**关键词**: influence functions, RLHF, reward model, bias detection, scalable oversight, OPORP

## 一句话总结

首次将影响函数系统性应用于 RLHF 奖励模型的人类反馈审计，通过 DataInf 近似 + OPORP 梯度压缩（160MB→256KB）实现 2.5 倍加速，在长度偏差和谄媚偏差检测上显著超越 GPT-4o 等基线（AUC 0.8 vs ~0.6），并在 Anthropic-HH 原始数据集中发现 47% 的错标样本，同时展示了影响函数指导非专家标注者向专家策略对齐的能力。

## 研究背景与动机

1. **RLHF 的数据依? Understanding Impact of Human Feedback via Influence Functions

**会议**: ACL 2025
**arXiv**: [2501.05790](https://arxi??**会议**: ACL 2025
**arXiv**: [2501.05790](https://arxiv.org???*a??这些问题会?*代码**: [https://github.com/mintaywon/IF_RLHF](https?*机构**: KAIST, Columbia University
**领域**: LLM?*：当前缺乏系统性的方?*领域**: LLM对齐 / RLHF 数据贡?**关键词**: influence functions, RLHF, rewa??
## 一句话总结

首次将影响函数系统性应用于 RLHF 奖励模型的人类反馈方法
首次将?is、KNN
## 研究背景与动机

1. **RLHF 的数据依? Understanding Impact of Human Feedback via Influence Functions

**会议**: ACL 2025
**arXiv**: [2501.05790](https://arxi??**会议**: ACL 2025
**arXiv**: [2501.05790](https://arxiv.org???*a??这些问题会?*代码**: [https://github.com/mintaywon/IF_RLHF](https?*机构**: KAIST, Columbia University
**领域**: LLM?*：当前缺乏系统性的於??1. **RLHF 的数据依???**会议**: ACL 2025
**arXiv**: [2501.05790](https://arxi??**会议**: ACL 2025
**ar. ***arXiv**: [2501.05??**arXiv**: [2501.05790](https://arxiv.org???*a??这些问??**领域**: LLM?*：当前缺乏系统性的方?*领域**: LLM对齐 / RLHF 数据贡?**关键词**: influence functions, RLHF, rewa??
## 一句话总结

馯?## 一句话总结

首次将影响函数系统性应用于 RLHF 奖励模型的人类反馈方法
首次将?is、KNN
## 研究背景丄?首次将影响??首次将?is、KNN
## 研究背景与动机

1. **RLHF 的数据依? Unders??数自然地对长?1. **RLHF 的数据依???**会议**: ACL 2025
**arXiv**: [2501.05790](https://arxi??**会议**: ACL 2025
**ar
?*arXiv**: [2501.05?*arXiv**: [2501.05790](https://arxiv.org???*a??这些问??*领域**: LLM?*：当前缺乏系统性的於??1. **RLHF 的数据依???**会议**: ACL 2025
**arXiv**: [2501.05790](https://arxi??**会议**: ACL 2025
*??*arXiv**: [2501.05790](https://arxi??**会议**: ACL 2025
**ar. ***arXiv**: [2501.05??**arXiv**60**ar. ***arXiv**: [2501.05??**arXiv**: [2501.05790](htB）?# 一句话总结

馯?## 一句话总结

首次将影响函数系统性应用于 RLHF 奖励模型的人类反馈方法
首次将?is、KNN
## 研究背景丄?首次将影响??首次将?is、KNN
## 研究背景与动?
馯?## 一句话???首次将影响函敀???正影响分数（增大验证损失）的样本为"负贡献"样本，即可## 研究背景丄??# 研究背景与动机

1. **RLHF 的数据依? Unders??
1. **RLHF 的数据依

?*arXiv**: [2501.05790](https://arxi??**会议**: ACL 2025
**ar
?*arXiv**: [2501.05?*arXiv**: ??*ar
?*arXiv**: [2501.05?*arXiv**: [2501.05790](https:\t?a)**arXiv**: [2501.05790](https://arxi??**会议**: ACL 2025
*??*arXiv**: [2501.05790](https://arxi??**?theta(x,y)$ 为 RM 对 prompt $x$ 和回复 $y$ 的奖励分数，$z$ 为标*??*arXiv**: [2501.05790](https://arxi??**会议**: ACL f{**ar. ***arXiv**: [2501.05??**arXiv**60**ar. ***arXiv**: [250l}
馯?## 一句话总结

首次将影响函数系统性应用于 RLHF 奖励模型的人类反馈方法
首次将?ixt{tr}}; \theta)^{-1} \nabl首次将?is、KNN
## 研究背景丄?首次将影响??首次将?is、KN}(\mathbf{d}_i) > 0$ ?# 研究背景与动?
馯?## 一句话???首次将影??馯?## 一句???验证?1. **RLHF 的数据依? Unders??
1. **RLHF 的数据依

?*arXiv**: [2501.05790](https://arxi??**会议**: ACL 2025
**ar
?*arXiv**: [2501.05?*arXiv**: ??*ar??. **RLHF 的数据依

?*arXiv??阵：

$$H_{\text{pref}**ar
?*arXiv**: [2501.05?*arXiv**: ??*ar
?*arXiv**: la?a}?*arXiv**: [2501.05?athcal{D}_{\text{*??*arXiv**: [2501.05790](https://arxi??**?theta(x,y)$ 为 RM 对 prompt $x$ 和回复 $y$ 的ight)$$

其中 $v_\mat馯?## 一句话总结

首次将影响函数系统性应用于 RLHF 奖励模型的人类反馈方法
首次将?ixt{tr}}; \theta)^{-1} \nabl首次将?is、KNN
## 研究背景丄?首次将影响??首次将?is、KN}(\mathbf{d}_i) > 0$ ?# ??
首次将影响函敜????次将?ixt{tr}}; \theta)^{-1} \nabl首次将?is、KNN
## 研究背景丄 ?# 研究背景丄?首次将影响??首次将?is、KNm-馯?## 一句话???首次将影??馯?## 一句???验证?1. **RLHF 的数据依? Unders??
1. **RLH??1. **RLHF 的数据依

?*arXiv**: [2501.05790](https://arxi??**会议**: ark 4.1 的核心??*arXiv**: [2501.05数**ar
?*arXiv**: [2501.05?*arXiv**: ??*ar??. **RLHF ???偏??*arX?**：构建 Concise 验证集（2,629 样本），chosen ?$$H_{\text??帮??**arXiv**: [2501.??**arXiv**: la?a}?*arXiv**: [2501.??其中 $v_\mat馯?## 一句话总结

首次将影响函数系统性应用于 RLHF 奖励模型的人类反馈方法
首次将?ixt{tr}}; \theta)^{-1} \nabl首次???
首次将影响函数系统性应??首次将?ixt{tr}}; \theta)^{-1} \nabl首次将?is、KNN
## 研究背景丄??# 研究背景丄?首次将影响??首次将?is、KNSt首次将影响函敜????次将?ixt{tr}}; \theta)^{-1} \nabl首次将?is、KNN
#??# 研究背景丄 ?# 研究背景丄?首次将影响??首次将?is、KNm-?e1. **RLH??1. **RLHF 的数据依

?*arXiv**: [2501.05790](https://arxi??**会议**: ark 4.1 的核心??*arXiv**: [2501.05数**ar
?*arXiv**: [2501.05?*arXiv**: mathbf
?*arXiv**: [2501.05790](https3]$?*arXiv**: [2501.05?*arXiv**: ??*ar??. **RLHF ???偏??*arX?**：构建 Concise 验证? 
首次将影响函数系统性应用于 RLHF 奖励模型的人类反馈方法
首次将?ixt{tr}}; \theta)^{-1} \nabl首次???
首次将影响函数系统性应??首次将?ixt{tr}}; \theta)^{-1} \nabl首次将?is、KNN
## 研究腍置：4 e首次将?ixt{tr}}; \theta)^{-1} \nabl首次???
首次将影响函数系统???次将影响函数系统性应??首次将??# 研究背景丄??# 研究背景丄?首次将??测方法 | 长度偏差 AUC | 谄媚偏差 A#??# 研究背景丄 ?# 研究背景丄?首次将影响??首次将?is、KNm-?e1. **RLH??1. **RLHF 的数据依

?*arXiv**: [2501.05790](https://arxi??-s
?*arXiv**: [2501.05790](https://arxi??**会议**: ark 4.1 的核心?ro (few-shot) | 0.544* | 0.592* | 表现不稳??**arXiv**: [2501.05?*arXiv**: mathbf
?*arXiv**: [2501.05790](https3]$?*arXiv**: [2501.05?S?f-confidence | ~0.6 | ~0.6 | 接近随???次将影响函数系统性应用于 RLHF 奖励模型的人类反馈方法
首次将?ixt{tr}}; \theta)^{-1} \nabl首次???
? 首次将?ixt{tr}}; \theta)^{-1} \nabl首次???
首次将影响函数系统????次将影响函数系统性应??首次将?ro## 研究腍置：4 e首次将?ixt{tr}}; \theta)^{-1} \nabl首次???
首次将影响函数?-首次将影响函数系统???次将影响函数系统性应??首38
?*arXiv**: [2501.05790](https://arxi??-s
?*arXiv**: [2501.05790](https://arxi??**会议**: ark 4.1 的核心?ro (few-shot) | 0.544* | 0.592* | 表现不稳??**arXiv**: [2501.05?*arXiv**: mathbf
?*arXiv**: [2501.05790](https3]$?*arXiv**: [2501.05?S?f-confidence | ~0.6 | ~0.6 |---?*arXiv**: [2501.05790](https://arxi??*??**arXiv**: [2501.05790](https3]$?*arXiv**: [2501.05?S?f-confidence | ~0.6 | ~0.6 | 接近随???次将影响函数系统性应用于 RLHF 奖励模型?|首次将?ixt{tr}}; \theta)^{-1} \nabl首次???
? 首次将?ixt{tr}}; \theta)^{-1} \nabl首次???
首次将影响函数系统????次将影响函数系统性应??首次??R 首次将?ixt{tr}}; \theta)^{-1} \nabl首??首次将影响函数系统????次将影响函数糏?首次将影响函数?-首次将影响函数系统???次将影响函数系统性应??首38
?*arXiv**: [2501.05790](https://arxi??-s
?*arXiv1.?*arXiv**: [2501.05790](https://arxi??-s
?*arXiv**: [2501.05790](https://arxi??**会议**: ?**arXiv**: [2501.05790](https://arxi???上?*arXiv**: [2501.05790](https3]$?*arXiv**: [2501.05?S?f-confidence | ~0.6 | ~0.6 |---?*arXiv**: [2501.05790](http?数据构造问题。
2. **47% 错标??R 首次将?ixt{tr}}; \theta)^{-1} \nabl首次???
首次将影响函数系统????次将影响函数系统性应??首次??R 首次将?ixt{tr}}; \theta)^{-1} \nabl首??首次将影响函数系统????次将影响函数糏?首次将影响函数?-首次将影响函数系统???次将影响函数系统性应??首38
?*arXiv**: ??首次将影响函数系统????次将影响函数?6?*arXiv**: [2501.05790](https://arxi??-s
?*arXiv1.?*arXiv**: [2501.05790](https://arxi??-s
?*arXiv**: [2501.05790](https://arxi??**会议**: ?**arXiv**: [2501.05790](https://arxi???上?*arXiv**: [2501.05790](https3]$?*arXiv**: [2501.05?S?f-confidence |8%），?*arXiv1.?*arXiv**: [2501.05790](https:?**arXiv**: [2501.05790](https://arxi??**会议**:??. **47% 错标??R 首次将?ixt{tr}}; \theta)^{-1} \nabl首次???
首次将影响函数系统????次将影响函数系统性应??首次??R 首次将?ixt{tr}}; \theta)^{-1} \nabl首??首次将影响函数系统????次将影响 O首次将影响函数系统????次将影响函数系统性应?????*arXiv**: ??首次将影响函数系统????次将影响函数?6?*arXiv**: [2501.05790](https://arxi??-s
?*arXiv1.?*arXiv**: [2501.05790](https://arxi??-s
?*arXiv**: [2501.05790](https://arxi??**会议**: ?**arXiv**: [2501.05790](https://arxi???上?*arXiv**: ???*arXiv1.?*arXiv**: [2501.05790](https://arxi??-s
?*arXiv**: [2501.05790](https://arxi??**会议**??了自?*arXiv**: [2501.05790](https://arxi??**会议**:????次将影响函数系统????次将影响函数系统性应??首次??R 首次将?ixt{tr}}; \theta)^{-1} \nabl首??首次将影响函数系统????次将影响 O首次将影响函数系统????次将影响函数系统性应?????*arXiv**: ??首次将影响函数系统????次将影响函数?6?*arXiv**: [2501.05790](https://ar???*arXiv1.?*arXiv**: [2501.05790](https://arxi??-s
?*arXiv**: [2501.05790](https://arxi??**会议**: ?**arXiv**: [2501.05790](https://arxi???上?*arXiv**: ???*arXiv1.?*arXiv**: [2501.05790](https://arxi??-s
?*arXiv**: [2501.05790](https://arxi??**会议**??了自?*arXiv**: [2501.05790](https://arxi??**会议**:????次将影响函数LL?*arXiv**: [2501.05790](https://arxi??**会议**:???*arXiv**: [2501.05790](https://arxi??**会议**??了自?*arXiv**: [2501.05790](https://arxi??**会议**:????次将影响函数系统????次将影响函数???*arXiv**: [2501.05790](https://arxi??**会议**: ?**arXiv**: [2501.05790](https://arxi???上?*arXiv**: ???*arXiv1.?*arXiv**: [2501.05790](https://arxi??-s
?*arXiv**: [2501.05790](https://arxi??**会议**??了自?*arXiv**: [2501.05790](https://arxi??**会议**:????次将影响函数LL?*arXiv**: [2501.05790](https://arxi??**会议**:???*arXiv**: [2501.05790](https://arxi??**会议**??了自?*arXiv**: [2501.05790](https://arxi??**会议**:????次将影响函数系统????次将影响唨?*arXiv**: [2501.05790](https://arxi??**会议**??了自?*arXiv**: [2501.05790](https://arxi??**会议**:????次将影响函数LL?*arXiv**: [2501.05790](htt??**arXiv**: [2501.05790](https://arxi??**会议**??了自?*arXiv**: [2501.05790](https://arxi??**会议**:????次将影响函数LL?*arXiv**: [2501.05790](https://arxi??**会议**:???*arXiv**: [2501.05790](https://arxi??**会议**??了自?*arXiv**: [2501.05790](https://arxi??**会议**:????次将影响函数系统????次将影响唨?*arXiv**: [2501.05790](https://arxi??**会议**??了自?*arXiv**: [2501.05790](https://arxi??**会议**:????次将影响函数LL?*arXiv**: [2501.05790](htt??**a?洞察表述精准 |
| 实用价值 | ⭐⭐⭐⭐ | 47% 错标发现有实际意义，开源代码可直接使用；但验证集构造的人工成本限制了即插即用性 |
