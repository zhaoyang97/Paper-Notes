---
title: >-
  [论文解读] Position: Adversarial ML for LLMs Is Not Making Any Progress
description: >-
  [ICML 2026][LLM 其他][对抗ML] 这是一篇立场论文，作者认为对抗机器学习在 LLM 时代研究的问题相比传统分类器场景"更难定义、更难求解、更难评测"，过去十年在 $\ell_p$ 鲁棒等"玩具问题"上就进展缓慢，如今全面转向 LLM 后很可能再耗一个十年仍无法产出可度量、可复现的安全保证。
tags:
  - "ICML 2026"
  - "LLM 其他"
  - "对抗ML"
  - "LLM安全"
  - "越狱"
  - "提示注入"
  - "评测可复现性"
---

# Position: Adversarial ML for LLMs Is Not Making Any Progress

**会议**: ICML 2026  
**arXiv**: [2502.02260](https://arxiv.org/abs/2502.02260)  
**代码**: 无（立场论文）  
**领域**: LLM 安全 / 对抗机器学习 / 评测方法论  
**关键词**: 对抗ML, LLM安全, 越狱, 提示注入, 评测可复现性  

## 一句话总结
这是一篇立场论文，作者认为对抗机器学习在 LLM 时代研究的问题相比传统分类器场景"更难定义、更难求解、更难评测"，过去十年在 $\ell_p$ 鲁棒等"玩具问题"上就进展缓慢，如今全面转向 LLM 后很可能再耗一个十年仍无法产出可度量、可复现的安全保证。

## 研究背景与动机

**领域现状**：对抗机器学习起家于"小问题、大方法"——攻防对象是垃圾邮件分类器、CIFAR/ImageNet 上的 CNN 等窄任务模型；威胁模型典型形式为"对输入加一个 $\ell_p \le \epsilon$ 的扰动让模型误分类"，攻击目标用 cross-entropy loss 一阶梯度即可优化，防御效果用 test accuracy 一项指标即可比较。即便在如此理想的设定下，社区也用了十年没把 $\ell_p$-bounded robustness 真正解决，大量发表的 empirical defense 在后续 adaptive attack 下被打穿（Carlini & Wagner 2017、Tramer et al. 2020）。

**现有痛点**：研究焦点转向 LLM 后，"安全"不再是一个有形式化定义的任务。开发者关心的是 helpfulness/honesty/harmlessness（HHH）这类抽象属性；攻击者目标是让模型输出"有害"内容；威胁模型从"小幅扰动"放大到"任意 prompt + fine-tune + pruning"。jailbreak、prompt injection、unlearning、membership inference 等子问题都同时遭遇三个困境：(a) 攻击成功难以判定，普遍依赖 LLM-as-judge 这种自指评测；(b) 攻击搜索空间离散、无界、不可微，自动化攻击普遍干不过人工 red team；(c) 主流被测系统是闭源、持续更新的 API，结果根本无法复现。

**核心矛盾**：传统对抗 ML 之所以勉强算得上"科学"，是因为 $\ell_p$ 球 + 分类正确率构成了一个虽然简化、但可被精确定义、可被对抗优化、可被复现的"必要条件"。LLM 安全研究为追求"贴近真实威胁"，主动放弃了这套形式化骨架，但既没有给出新的可度量替代，也没有得到任何带证书的防御——结果是社区在"看似进步"（新模型 jailbreak 难度上升）和"实际无进步"（worst-case 失败率仍接近 100%）之间产生了系统性错觉。

**本文目标**：从"定义—求解—评测"三个维度系统梳理 LLM 时代对抗 ML 比传统设定额外多出的困难，并通过 6 个具体子领域案例（jailbreak / un-finetunable / poisoning / prompt injection / membership inference / unlearning）实证这些困难如何阻碍可累积的科学进展。

**切入角度**：作者不否认 LLM 安全是真问题，但坚持区分"研究真实世界安全漏洞"与"推进对抗 ML 的科学理解"两类工作——后者必须建立在形式化、可复现的 toy problem 之上；如果连缩小版的子问题都解决不了，整个 fuzzy 大问题的"进展"就是不可证伪的。

**核心 idea**：用一句话概括即"先把可定义的子问题解决了再谈安全"——呼吁社区为每个 LLM 安全方向定义类似 $\ell_p$-bounded perturbation 的最小可形式化版本，否则十年后回望仍将无法回答"我们到底进步了多少"。

## 方法详解

本文是立场论文，没有算法与训练，它的"方法"是一套用来给对抗 ML 研究做体检的分析框架。

### 整体框架

作者主张：LLM 安全研究在"定义—求解—评测"三个环节上都比分类器时代显著恶化，因而过去十年都没解决的对抗 ML 很可能再耗一个十年仍无可累积的进展。论证方式是把研究流程拆成"定义问题 → 求解问题 → 评测结果"的闭环，逐环节列出从分类器到 LLM 退化的子挑战，再用一张挑战矩阵（论文 Table 1）把 jailbreak、un-finetunable、poisoning、prompt injection、membership inference、unlearning 这 6 个子领域逐一对照——看哪些方向"全维度沦陷"、哪些方向"还剩一两个抓手"。

### 关键设计

**1. "定义"维度的三处塌方：连"什么算攻击成功"都说不清**

第一处塌方是攻击成功的判定。分类问题里只要比对预测标签，但 LLM 里"harmful"根本无法形式化，社区被迫退回 LLM-as-judge 这种带循环依赖的代理。第二处是攻击空间的边界：分类器有 $\|x' - x\|_p \le \epsilon$ 这样的几何约束，而 LLM 任意输入都可能触发不安全输出，几乎所有 jailbreak / prompt injection 论文都默认威胁模型"无界"，甚至把 fine-tune、prune 等修改模型本身的能力也算给攻击者。第三处是训练数据边界：传统 IID 的 train/test 划分在万亿 token 语料上彻底失效，membership inference 与 unlearning 因此从"是否包含某个样本点"退化成"是否包含某个概念"，连样本的同一性都没了。作者强调，定义是科学的最低门槛——如果连"什么算攻击成功"、"什么算合法攻击"、"什么算训练成员"都说不清，后续所有论文里的"提升 X%"就都失去了对比基准。

**2. "求解"维度的两处塌方：最强攻击靠人工、防御没有原理**

第一处塌方是攻击搜索。分类器场景下 PGD/CW 这类基于 input-gradient 的白盒攻击稳定胜过人工，攻击者只要朝 $\nabla_x \mathcal{L}$ 走即可；可 LLM 的离散 token 空间让梯度方法大打折扣，GCG 等代表方法生成的是 gibberish 串，效果跟随机搜索差不了多少，而真正强的攻击——persona modulation、多轮对话、社工话术——全靠人工 red team，结果是"worst-case 性能"在工程上根本无法被自动逼近。第二处是防御原理：分类器场景里既有 randomized smoothing 这类带证书的防御，也有 adversarial training 这类目标明确的经验防御；LLM 侧的防御却几乎清一色是针对已知攻击的对抗微调、latent space 虚拟对抗训练、外接 LLM 分类器或 Llama Guard、输入随机预处理，全都说不清自己"到底防住了什么"，又一次次被新攻击打穿（Łucki et al. 2024 等）。判断对抗 ML 是否在真解决问题，标尺本应是"最强攻击下系统是否守住"；可当最强攻击靠人手、防御又不可形式化时，攻防双方就各拍各的脑袋，永远收敛不到一起。

**3. "评测"维度的两处塌方：判别器循环依赖 + 闭源靶子无法复现**

第一处塌方是危害与效用的度量。传统任务一个 misclassification rate / clean accuracy 就能同时衡量攻击和效用，LLM 却必须靠 LLM-as-judge 来判断输出是否有害——而 judge 自己会被 prompt 攻击（Mangaokar et al. 2024）、会把"任何非拒绝回答"都误判成攻击成功（Souly et al. 2024）、还会和"以同类判别器为基础的防御"产生互相抬分的偏差；与此同时"模型还有没有用"也没有标准答案，一个对什么都拒答的 trivial defense 在安全分上堪称完美，却把全部 utility 都毁了。第二处是可复现性：主流被测对象 GPT-4 / Claude 等闭源模型频繁静默更新，攻击 prompt 可能一周后就失效，许多论文报告的成功率根本无法被独立验证，等于在移动靶上射击，社区因此无法在时间轴上累积可对比的结果。说到底，科学共同体只能在可复现的基准上攒进展，当评测既不客观也不稳定，"今年的 SOTA"就只是个论文标题，并不能证明系统真的更安全了。

**4. 规范性主张：把每篇论文显式分进"漏洞 demo"或"科学研究"两条 lane**

作者最后给出一条带处方性质的建议，本质上是一种 meta-loss：要求每篇论文明确归类为"研究真实世界漏洞"还是"推进对抗 ML 的科学理解"。前者可以接受 fuzzy 评价，但必须把危害讲具体；后者则必须自我约束到形式化的 toy 子问题上——比如固定长度后缀越狱、bounded sentence modification 攻击——并接受 adaptive 评测，否则就不该被当作科学贡献。这条主张相当于把"必要条件"的逻辑从 $\ell_p$ 球迁回 LLM 安全：连缩小版的子问题都解决不了，整个 fuzzy 大问题的"进展"就根本不可证伪。

## 实验关键数据

本文是立场论文，没有量化对照实验；作者用 6 个子领域案例 + 1 张挑战矩阵替代"主表"，下表是对论文 Table 1 与第 3 节叙述的整理。

### 主"实验"：6 大子领域 × 7 子挑战 的塌方矩阵

| 子领域 | 定义攻击成功 | 攻击空间无界 | 数据边界模糊 | 攻击搜索难 | 防御无原理 | 危害/效用难测 | 复现性差 |
|---|---|---|---|---|---|---|---|
| Jailbreaks | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| Un-finetunable Models | ✓ | ✓ | — | ✓ | ✓ | ✓ | — |
| Poisoning & Backdoors | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Prompt Injections | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| Membership Inference | ✓ | — | ✓ | — | — | ✓ | — |
| Unlearning | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |

读法：✓ 表示该子领域在该维度上已被 LLM 化显著恶化、相比分类器时代失去了原有抓手。所有 6 个方向都至少在 4 个维度上沦陷，jailbreak / poisoning / prompt injection / unlearning 更是接近全维度沦陷。

### "消融"：LLM 时代 vs 分类器时代的关键能力对比

| 设定 | 攻击目标 | 攻击空间 | 最强攻击来源 | 评测器 | 可复现性 |
|---|---|---|---|---|---|
| 经典 $\ell_p$ 对抗样本 | 误分类（明确） | $\|x'-x\|_p \le \epsilon$ | 白盒 PGD/CW（自动） | test accuracy | 数据集 + 权重公开 |
| LLM 越狱 | 输出"有害"内容（主观） | 任意 token 序列 + 可微调/可剪枝 | 人类 red team（手工） | LLM-as-judge | 闭源 API 持续更新 |
| LLM unlearning | 抹除"某概念" | 任意 prompt + 白盒干预 | 自适应 fine-tune | 难以分离 utility 影响 | retrain 不可行 |

### 关键发现

- 最关键的恶化不是"问题更难"，而是"成功标准本身消失"——LLM-as-judge 的循环依赖让攻防双方都可以通过攻击 judge 间接刷分。
- 攻击侧最反直觉的事实是：在 LLM 上手工攻击长期强于自动优化（GCG 与简单 random search 接近），这与图像对抗样本时代"白盒自动攻击碾压人工"的格局完全相反，意味着"worst-case"已无可计算上界。
- 模型越新越难被越狱并不等价于"安全在进步"——评测能力本身在劣化，更可能是大家失去了发现失败案例的工具。
- 评测器与防御共用同源 LLM 时会产生人工高分（Liu et al. 2024 的输出过滤型防御例子），构成结构性 benchmark 污染。

## 亮点与洞察

- 用"定义—求解—评测"三段式 + 子挑战矩阵把散落在 jailbreak / unlearning / MI 等子社区的痛点统一成一张体检表，给出了少见的横向诊断视角，可直接拿来扫描自己手里的安全论文是否在某个维度上"裸奔"。
- 把"研究真实世界漏洞"和"推进科学理解"做两条 lane 的区分非常实用——它把"我做 demo 攻击 GPT-4"和"我证明某防御的 robust radius"分开评判，避免两类工作互相用错的标尺批判对方。
- 关于 LLM-as-judge 的循环依赖与防御-评测同源偏差，是当前安全 benchmark 几乎人人在用却很少形式化讨论的盲点，这种洞察可平移到 RLHF 评测、red team benchmark 设计上。
- 把"必要条件"思路从 $\ell_p$ 球迁移到 LLM 安全——例如"固定长度后缀的越狱可检测性"、"有界编辑距离 prompt injection 的可防御性"——是相对落地的研究纲领，能直接孵化形式化子任务。

## 局限与展望

- 作者承认存在反方观点：复杂度上升可能是"终于在解真问题"的代价，jailbreak 的"无论 context 都不许输出"反而比"正确分类 guacamole 同时不能把猫误判为 guacamole"更简单；本文的回应（举出 representation engineering 也被打穿等反例）偏定性，没有给出量化判据来区分"暂时未解决"和"原则上无法解决"。
- 全文以"批判 + 呼吁"为主，并未提出任何具体的 LLM 安全 toy benchmark；读者会问"那应该用什么替代 HarmBench / JailbreakBench"，作者只给方向（"类似 $\ell_p$ 的形式化子任务"），没有候选设计。
- 案例覆盖偏向 jailbreak/unlearning，对 LLM agent 安全（工具调用安全、多智能体协议攻击等）只在 prompt injection 节带过，对智能体世界的恶化分析不够。
- 文章默认"可形式化 = 可累积科学"，但形式化并不自动等于"贴近真实危害"——比如 $\ell_p$ robustness 十年研究对真实人脸识别系统的安全提升是否成正比，本身就是开放问题，把它当作 LLM 安全的范本本身有可商榷之处。
- 可延伸方向：为每个子领域 propose 一个最小 formal benchmark（如"长度 ≤ k 的固定 suffix 越狱率"），并配套 adaptive evaluation 协议，让"按本文呼吁去做"具有可操作的第一步。

## 相关工作与启发

- **vs Carlini & Wagner 2017 / Tramer et al. 2020 的"防御被打穿"系列**：这两篇是图像对抗时代的同款警告，本文在结构上延续了"empirical defense 必须接受 adaptive attack 才能算数"的传统，扩展到 LLM 场景；区别在于图像时代攻击方法（PGD）尚可形式化，LLM 时代连 adaptive attack 该长什么样都没有共识。
- **vs HarmBench / JailbreakBench 等评测工作**：那些工作试图把"危害评分"标准化，本文则指出标准化本身的循环依赖，并不否认这些 benchmark 短期内仍是必要妥协，但呼吁额外引入形式化 toy 子任务，作为可累积的科学坐标。
- **vs Representation Engineering / Circuit Breakers**：本文把这类"在隐空间识别危险方向"的防御与图像时代的 detection-based defense 类比，提示研究者警惕"detection 方法在新攻击下集体失效"的历史重演；启发是任何依赖隐式表征的防御都需要 adaptive evaluation。
- **vs Cooper et al. 2024（unlearning 立场论文）**：两者都对 LLM unlearning 表示悲观，本文从对抗 ML 三段式角度补充了"概念级 unlearning 与 sample 级 unlearning 不可混为一谈"的论证，把它放进与 jailbreak、prompt injection 等同结构的失败模式中。
- 启发：未来论文若想避免被作者"打 fuzzy"，至少应在开篇明确自己属于"漏洞 demo"还是"科学研究"，后者需要附带一个形式化定义 + 一个 adaptive attack 协议。

## 评分
- 新颖性: ⭐⭐⭐⭐ 框架本身（三段式 + 子挑战矩阵）整合度高，但具体观点在子社区里早已分散存在。
- 实验充分度: ⭐⭐⭐ 立场论文无实验，6 个案例覆盖典型方向但分析深度不均，jailbreak/unlearning 深入，poisoning/MI 偏概述。
- 写作质量: ⭐⭐⭐⭐⭐ 论证层次清晰，正反观点都给到位（第 4 节专门列对立视角），是少见的"建设性悲观"立场论文。
- 价值: ⭐⭐⭐⭐⭐ 给整个 LLM 安全社区提供了一面镜子，对评审标准和后续工作选题都有实质牵引。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Position: The ML Community Must Build an AI-Augmented Peer-Review Ecosystem](position_the_ml_community_must_build_an_ai-augmented_peer-review_ecosystem.md)
- [\[ACL 2025\] Biased LLMs Can Influence Political Decision-Making](../../ACL2025/llm_nlp/biased_llms_can_influence_political_decision-making.md)
- [\[ICML 2026\] Position: The Turing-Completeness of Autoregressive Transformers Relies Heavily on Context Management](position_the_turing-completeness_of_autoregressive_transformers_relies_heavily_o.md)
- [\[ICLR 2026\] When Stability Fails: Hidden Failure Modes of LLMs in Data-Constrained Scientific Decision-Making](../../ICLR2026/llm_nlp/when_stability_fails_hidden_failure_modes_of_llms_in_data-constrained_scientific.md)
- [\[ICML 2026\] Position: Hippocampal Explicit Memory Is the Cornerstone for AGI](position_hippocampal_explicit_memory_is_the_cornerstone_for_agi.md)

</div>

<!-- RELATED:END -->
