---
title: >-
  [论文解读] Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models
description: >-
  [ICML 2026][LLM安全][LLM unlearning] 本文指出现有 LLM unlearning 方法在"把知识从参数里抹掉"的同时，会把"用户在 prompt 里重新提供该知识时模型能正确利用"的能力（contextual utility）一起抹掉，作者提出在已有 unlearning loss 上加一项 KL 正则——让 unlearn 后的模型在"问题+上下文"输入上的分布对齐原始模型——即可在几乎不损失遗忘效果和保留集效用的前提下，把 Contextual QA 的 LLM-Judge 分数从 0.00–0.84 拉回到 0.95+。
tags:
  - "ICML 2026"
  - "LLM安全"
  - "LLM unlearning"
  - "contextual utility"
  - "KL 正则"
  - "TOFU"
  - "RAG 友好遗忘"
---

# Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models

**会议**: ICML 2026  
**arXiv**: [2510.17620](https://arxiv.org/abs/2510.17620)  
**代码**: 未公开  
**领域**: LLM 安全 / 机器遗忘  
**关键词**: LLM unlearning, contextual utility, KL 正则, TOFU, RAG 友好遗忘

## 一句话总结
本文指出现有 LLM unlearning 方法在"把知识从参数里抹掉"的同时，会把"用户在 prompt 里重新提供该知识时模型能正确利用"的能力（contextual utility）一起抹掉，作者提出在已有 unlearning loss 上加一项 KL 正则——让 unlearn 后的模型在"问题+上下文"输入上的分布对齐原始模型——即可在几乎不损失遗忘效果和保留集效用的前提下，把 Contextual QA 的 LLM-Judge 分数从 0.00–0.84 拉回到 0.95+。

## 研究背景与动机

**领域现状**：LLM 训练在 web 规模语料上，难免吃进版权内容、个人隐私等需要被"删除"的信息。直接重训成本不可接受，于是出现了一批 unlearning 方法：梯度上升族（GradAscent / GradDiff）、偏好优化族（NPO、DPO 变体）、re-labeling 族（UNDIAL）、表示扰动族（RMU）等。评测标准基本是两条腿：forget set 上要忘干净（Direct QA 分数低），retain set 上要保住能力（model utility 不掉）。

**现有痛点**：作者观察到一个被整个社区忽视的第三维度——RAG 和长 prompt 场景里，模型经常会拿到"理论上已被遗忘"的内容作为输入（用户自己上传的文档、检索到的版权章节等），合规上允许使用，因为信息是用户当场提供的、不是模型记住的。但现有 unlearning 方法在这种"题目+答案都摆在眼前"的 Contextual QA 上同样失败：在 Gemma-2B-IT 上 RMU / GradAscent / GradDiff 把 Contextual QA 几乎打到零，NPO/UNDIAL 也掉 15.5%+。case study 显示输出从"换了个国家的幻觉"一路退化到"denden den den..."这种纯乱码。

**核心矛盾**：所有现有 loss 都只在"forget vs retain"二元 trade-off 上做文章，本质是在对 $\mathcal{S}_f$ 的参数表征施加惩罚。这种惩罚不会"只惩罚记忆调用"——它会顺着表示空间外溢到 inference 时的 context conditioning：当同样的 token 作为上下文出现时，模型也丧失了 grounding 在这些 token 上生成正确答案的能力。RMU 这种通过扰动激活来抹除的方法尤其严重，因为它直接破坏了相关概念的表示通路。

**本文目标**：(1) 系统量化 6 种 SOTA unlearning 方法在 Contextual QA 上的副作用；(2) 设计一个"可即插即用、对原方法改动最小"的修补项，把 contextual utility 救回来而不破坏 forgetting 和 utility。

**切入角度**：RLHF 里早就证明 KL 正则可以稳住模型在某些行为维度上不偏离原模型。既然问题是"unlearn 后模型在 (q, c) 这种输入下行为变了"，那么直接拿原始模型做锚点，在 $(q, c)$ 这条数据流上加一个 KL 约束就好——它和 forget term 作用在不同的输入分布上（$\mathcal{S}_f$ vs $\mathcal{S}_f^{\text{ctx}}$），不会互相打架。

**核心 idea**：在标准两项 unlearning loss 上加第三项 $\lambda_c \cdot \mathrm{KL}(p_w(\cdot|q,c) \,\|\, p_{\text{orig}}(\cdot|q,c))$，显式地把"上下文条件分布"锚在原模型上，把"参数记忆"和"上下文使用"在 loss 层面解耦，实现"do not recall from memory, but do use when provided"。

## 方法详解

### 整体框架
全文是一个"先诊断、后修补"的闭环。诊断阶段在 TOFU benchmark（虚构作者档案，保证未被预训练见过）上用 5% forget ratio 跑 6 种 SOTA unlearning 方法（GradAscent、GradDiff、NPO、DPO、UNDIAL、RMU），但在标准的 Direct QA 之外加一条新评测线：同样的 forget set 问题，把 ground truth 当作 context 一起塞进 prompt，看模型"答案都摆在眼前时还能不能答对"——结果暴露出现有方法连这都做不到。修补阶段顺着诊断结论给所有 unlearning loss 打同一个补丁：在 forget set 之外另构一份 $\mathcal{S}_f^{\text{ctx}} = \{(q, a, c)\}$（$c$ 是含答案的 context），加一项 KL 正则 $\mathcal{C}(\mathcal{S}_f^{\text{ctx}}, w)$，逼 unlearn 中的模型在这条 contextual 数据流上的预测分布对齐冻结的原始模型 $p_{\text{orig}}$。目标函数从两项变三项：$\mathcal{J}(w) = -\lambda_f L_f(\mathcal{S}_f, w) + \lambda_r L_r(\mathcal{S}_r, w) + \lambda_c \mathcal{C}(\mathcal{S}_f^{\text{ctx}}, w)$，分别管 forget、retain、contextual。

### 关键设计

**1. Contextual QA 评测协议：把 unlearning 漏掉的第三维度量化出来**

以前的评测只有两条腿——forget set 上要忘干净（Direct QA 低）、retain set 上要保住能力（utility 高）——但 LLM 越来越多跑在 RAG / 长 prompt 场景里，"用户当场重新提供的合法信息能不能被用上"是部署里真正关心却没人测的事。这个协议补上这一维：对 forget set 里每个 $(q, a)$ 配一份 context $c$（包含 ground truth 描述），构造"问题 + 提供的 context"prompt 喂给 unlearn 后的模型，用 ROUGE-L 衡量字面重合、用 LLM-Judge（Appendix 有人类一致性验证）衡量语义正确，两个指标都落在 $[0, 1]$。理想的 unlearn 模型应当三项齐活：Direct QA 低、Contextual QA 高、utility 高。为防止模型只是死记 context 表层，协议还配了 paraphrase 和 reasoning 两种 context 变体。一上手这个协议就抓出了问题：Table 1 里 6 种方法有 5 种在"答案就在 context 里"时仍产出错误甚至乱码，副作用一直藏在旧评测的盲区里。

**2. Context-aware KL 正则项：用 RLHF 的旧药锚住悬空的那一维**

核心洞察是原来两项 loss 只约束了 $\mathcal{S}_f$ 和 $\mathcal{S}_r$ 两个数据分布，contextual 这第三个分布完全悬空，于是 forget term 的惩罚顺着表示空间外溢、把模型在 $(q, c)$ 输入下的 grounding 能力一起带塌了。补救办法是给这条悬空的数据流也钉一个锚：构造 $\mathcal{S}_f^{\text{ctx}} = \{(q, a, c)\}$（context $c$ 直接从 TOFU 的 ground truth 派生、等价于"答案陈述句"，无需额外标注），定义

$$\mathcal{C}(\mathcal{S}_f^{\text{ctx}}, w) = \frac{1}{|\mathcal{S}_f^{\text{ctx}}|} \sum_{(q,a,c)} \mathrm{KL}\big(p_w(\cdot|q,c) \,\|\, p_{\text{orig}}(\cdot|q,c)\big),$$

让当前模型在"问题+context"下的 token 分布逼近冻结的原模型。这一招的精妙全在锚点和作用域的选法上：RLHF 里 KL-to-reference 是稳住模型不偏离指定行为的教科书工具，这里把"reference"设成 pre-unlearning 的自己（所以不需要外部 teacher、不需要额外标注），把"被约束的行为"严格限定在 $(q, c)$ 输入流（所以不碰 $\mathcal{S}_f$ 也不碰 $\mathcal{S}_r$，与原 unlearning loss 正交）。因为它根本没动 forget set 上的 loss，遗忘强度仍由原方法决定；又因为 KL 是分布级而非单点 distillation 的约束，比硬推 token-level 目标更柔和，不会和 forget term 死斗。实践上也很省心：Appendix A.6 验证对 $\lambda_c$ 极不敏感，NPO/RMU/UNDIAL 在两个模型上各自取 0.01–2.0 之间都稳，无需精细调参。

**3. 与现有方法的即插即用集成：一个 context term 通吃三种范式**

unlearning 社区方法极度碎片化、每年新方法满天飞，如果补丁只能配自己那套 loss 就毫无普及价值。好在现有方法不管是 preference optimization（NPO）、re-labeling（UNDIAL）还是激活扰动（RMU），都遵循 "forget term + 可选 retain term" 的二项结构，新方法只是再缀一项 $+\lambda_c \mathcal{C}$——每个 step 多一次原模型前向（算 $p_{\text{orig}}(\cdot|q,c)$）和一次 KL 计算，唯一新增的超参 $\lambda_c$ 每个方法每个模型给一个值即可。同一个 context term 对这三类完全不同范式都能拉出大幅 Contextual QA 提升（RMU 从 $0.00 \to 0.99$ 最戏剧），既说明 contextual suppression 是跨方法的共病，也说明 KL anchor 是跨方法的通解。

### 损失函数 / 训练策略
最终 loss 即 $\mathcal{J}(w) = -\lambda_f L_f + \lambda_r L_r + \lambda_c \mathcal{C}$。训练沿用 TOFU 的标准设置（AdamW，原文 5 epoch 扩到 20 epoch 以保证充分收敛），新增的只有 $\lambda_c$：Gemma-2B-IT 上 NPO/RMU/UNDIAL 分别取 2.0 / 0.01 / 0.5，Qwen3-8B 上分别取 1.0 / 0.5 / 1.0。收敛准则是 Direct QA LLM-Judge、Contextual QA LLM-Judge、model utility 三者同时逼近该方法整条曲线的 global best。

## 实验关键数据

### 主实验
TOFU 5% forget ratio，三种 unlearning 方法的 vanilla vs context-aware 对比：

| 模型 | 方法 | 变体 | Direct ROUGE-L ↓ | Contextual ROUGE-L ↑ | Direct LLM-Judge ↓ | Contextual LLM-Judge ↑ | Utility ↑ |
|------|------|------|---|---|---|---|---|
| Gemma-2B-IT | NPO | Vanilla | 0.31 | 0.55 | 0.19 | 0.81 | 0.57 |
| Gemma-2B-IT | NPO | Context-aware | 0.36 | **0.87** (+0.32) | 0.25 | **0.98** (+0.17) | 0.57 |
| Gemma-2B-IT | RMU | Vanilla | 0.04 | 0.01 | 0.00 | 0.00 | 0.60 |
| Gemma-2B-IT | RMU | Context-aware | 0.13 | **0.91** (+0.90) | 0.01 | **0.99** (+0.99) | 0.57 |
| Gemma-2B-IT | UNDIAL | Vanilla | 0.33 | 0.53 | 0.39 | 0.82 | 0.54 |
| Gemma-2B-IT | UNDIAL | Context-aware | 0.34 | **0.87** (+0.34) | 0.38 | **0.98** (+0.16) | 0.55 |
| Qwen3-8B | NPO | Vanilla | 0.27 | 0.46 | 0.14 | 0.84 | 0.60 |
| Qwen3-8B | NPO | Context-aware | 0.29 | 0.63 (+0.17) | 0.20 | **0.95** (+0.11) | 0.61 |
| Qwen3-8B | RMU | Vanilla | 0.10 | 0.18 | 0.00 | 0.05 | 0.59 |
| Qwen3-8B | RMU | Context-aware | 0.13 | 0.67 (+0.49) | 0.01 | **0.97** (+0.92) | 0.57 |
| Qwen3-8B | UNDIAL | Vanilla | 0.32 | 0.59 | 0.38 | 0.97 | 0.60 |
| Qwen3-8B | UNDIAL | Context-aware | 0.33 | 0.68 (+0.09) | 0.39 | 0.98 (+0.01) | 0.61 |

最戏剧性的是 RMU：vanilla 在 Contextual QA 上几乎全军覆没（LLM-Judge $\leq 0.05$），context-aware 版本两个模型上都拉到 $\geq 0.97$。

### 消融实验

| 维度 | 配置 | 关键发现 |
|------|------|------|
| Forget ratio | 1% / 5% / 10% | 三种比例下 vanilla 都显著降 Contextual QA、context-aware 都能稳定救回，遗忘难度变化不影响 KL anchor 的有效性 |
| Context 变体（RMU） | 原文 / paraphrase / reasoning | vanilla RMU 三种变体全乱码；context-aware RMU 三种变体全部生成正确答案——证明不是死记 context 表层，而是真的恢复了利用语义信息的能力 |
| Direct QA 副作用 | 加 context term 后 | 平均 ROUGE-L 变 $\sim 4$pp，LLM-Judge 变 $\sim 2$pp（Gemma），Qwen 上分别 $\sim 2$pp 和 $\sim 3$pp，量级远小于 Contextual QA 收益 |
| Model utility | 加 context term 后 | Gemma 平均 -0.01，Qwen 平均 0.00，几乎零成本 |
| $\lambda_c$ 敏感性（Appendix A.6） | 各方法各模型 | 三种方法各自的 $\lambda_c$ 跨 0.01–2.0 数量级都稳定，调参极容易 |
| 数据集泛化 | TOFU + PISTOL（结构性纠缠实体） | 两个数据集趋势完全一致 |
| Noisy context（Appendix） | GPT 生成的长段落 / 含冲突 spans / 多个冲突 distractor | Contextual QA 随 context 噪声变难而下降，但 context-aware 相对 vanilla 的绝对增益依然显著 |

### 关键发现
- **Contextual suppression 是 unlearning 方法的共性副作用**：6 种 SOTA 方法里 5 种在 Contextual QA 上严重失败（Gemma 上 RMU/GradAscent/GradDiff 直接打到零），且这种失败是 paradigm-agnostic 的——梯度上升、偏好优化、激活扰动都中招。
- **RMU 在标准评测里表现最好，在新评测里反而最差**：因为它通过激活扰动来抹除概念表示，这种"重击表示空间"的手法对 contextual conditioning 的破坏也最彻底；UNDIAL 因为是 re-labeling 而非惩罚式，副作用最小但 Direct QA 遗忘也较弱。这揭示了 unlearning 范式选择上的一个新 trade-off。
- **KL anchor 是稳定的救命稻草**：对 $\lambda_c$ 不敏感意味着实际部署里不需要精细调参；对 paraphrase 和 reasoning context 的鲁棒性意味着学到的不是浅层 pattern matching；与原 unlearning loss 几乎无干扰意味着可以直接拼接到任何新出的 unlearning 方法上。

## 亮点与洞察
- **问题发现本身就是论文的一半价值**：Contextual QA 这个评测维度提出来之前，整个社区都在用 Direct QA + utility 两条腿评测，本文用一张图（Figure 1）把"unlearning 同时杀掉了一个被忽视的关键能力"这个隐藏成本翻出来，并通过 case study 给出了从"换国家幻觉"到"乱码"的视觉冲击力，是典型的"开辟新评测维度倒推方法改进"的范式。
- **解法的优雅在于"用 RLHF 的旧药治新病"**：KL-to-reference 在对齐里是教科书级工具，但搬到 unlearning 场景里时锚点选择（pre-unlearning 的自己 vs 某个外部模型）和锚定的输入流（contextual stream vs forget stream vs retain stream）都是非平凡选择。作者把它精确锚在 $(q, c)$ 输入流上，等于在 loss 拓扑上找到了一条之前没人占领的"安全缝隙"，既不和 forget term 死斗也不需要新数据标注。
- **"两段式"框架可迁移到其他遗忘类任务**：模型编辑（model editing）、概念抹除（concept erasure）等也存在类似的"参数级删除 vs 上下文级保留"的张力，本文这套"诊断—锚定—解耦"的三步走可以套用过去。

## 局限与展望
- 作者承认的局限：在"加强 contextual utility"和"加强 Direct QA forgetting"之间仍存在轻微 trade-off，权衡由部署场景决定（严格删除场景 vs RAG 场景）；评测主要在 TOFU 和 PISTOL 上，模型规模上限是 Qwen3-8B；缺少理论分析。
- 自己发现的局限：Contextual QA 的 context 来源于 TOFU ground truth，本质上是"答案陈述句"，离真实 RAG 场景的"长文档 + 多段落 + 噪声"距离较远（Appendix 虽有 noisy context 实验，但仍是合成的）；新增的 $\mathcal{S}_f^{\text{ctx}}$ 数据流需要每个 forget 样本配 context，实际部署里如果 forget 请求来自用户而 context 还没生成，如何流式构造 $\mathcal{S}_f^{\text{ctx}}$ 不显然；KL anchor 是在 token 分布层面，对长生成可能漂移，是否在更长输出上仍稳定值得追加实验。
- 改进思路：（1）把 context term 从 KL 推广到序列级一致性（如 sequence-level KD 或 self-consistency reward）；（2）研究 contextual utility 和 jailbreak 攻击的关系——如果模型对"用户提供的 forget 信息"敏感，恶意用户能否通过精心构造的 context 触发 leakage？（3）把这套框架扩展到 multi-document RAG 和 multi-turn dialogue 场景，验证锚定 $(q, c)$ 是否依然足够，还是需要锚定 $(q, c_1, \ldots, c_k)$。

## 相关工作与启发
- **vs TOFU (Maini et al., 2024) / MUSE (Shi et al., 2025) 等评测**：他们建立了 forget+retain 双轴评测协议，本文在同一 benchmark 上加了第三轴 Contextual QA，不是另起炉灶，而是补全维度，复用现有论文的可比性。
- **vs NPO / RMU / UNDIAL 等 unlearning 方法**：本文不和它们竞争，而是给它们打补丁——三类完全不同范式都能受益，说明 contextual suppression 是 unlearning 的范式级共病。
- **vs in-context unlearning（Pawelczyk et al., 2024; Muresanu et al., 2025）**：那些工作把 context 当作模拟 unlearning 的工具（"用 prompt 假装忘了"），本文相反——研究"参数级 unlearning 如何破坏 in-context 利用能力"，方向正交、互补。
- **vs unlearning reversal（Shumailov et al., 2024; Cooper et al., 2025）**：那条线把 context 当作攻击向量（对抗者用 context 唤回被遗忘的知识），本文把 context 当作合法用户输入，研究"如何保住合法 contextual 利用而不打开攻击面"——两者其实暗含张力，是未来重要的研究方向。
- **vs KL-regularized RLHF (Ouyang et al., 2022)**：本质上是把 RLHF 的 KL anchor 思想搬到 unlearning，但锚点选择和作用域设计是 unlearning 独有的非平凡问题。

## 评分
- 新颖性: ⭐⭐⭐⭐ 开辟 Contextual QA 评测维度是真实的概念贡献；解法上的 KL anchor 思路并不新但用对地方
- 实验充分度: ⭐⭐⭐⭐ 6 个 baseline × 2 模型 × 3 forget ratio × 4 context 变体 + 2 数据集，覆盖很全；模型规模止于 8B 略遗憾
- 写作质量: ⭐⭐⭐⭐ Figure 1 一图把问题讲清，case study 给得很有冲击力，loss 公式拆解清晰
- 价值: ⭐⭐⭐⭐⭐ 在 RAG 和合规需求并存的部署场景里几乎是必需补丁，且即插即用门槛极低

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models](dualoptim_bridging_shared_and_decoupled_optimizer_states_for_better_machine_unle.md)
- [\[ICML 2026\] COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models](coft_counterfactual-conformal_decoding_for_fair_chain-of-thought_reasoning_in_la.md)
- [\[ICML 2026\] BYORn: Bootstrap Your Own Responses to Defend Large Vision-Language Models Against Backdoor Attacks](byorn_bootstrap_your_own_responses_to_defend_large_vision-language_models_agains.md)
- [\[ICML 2026\] The Unlearnability Phenomenon in RLVR for Language Models](the_unlearnability_phenomenon_in_rlvr_for_language_models.md)
- [\[ICML 2026\] Differentially Private Preference Data Synthesis for Large Language Model Alignment](differentially_private_preference_data_synthesis_for_large_language_model_alignm.md)

</div>

<!-- RELATED:END -->
