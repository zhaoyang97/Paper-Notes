---
title: >-
  [论文解读] AdaIAT: Adaptively Increasing Attention to Generated Text to Alleviate Hallucinations in LVLM
description: >-
  [CVPR 2026][幻觉检测][LVLM幻觉] 针对"放大图像注意力虽能压幻觉、却让模型重复啰嗦"的痛点，本文发现真实物体 token 比幻觉 token 对**已生成文本** $T_p$ 的注意力更高，于是改为增大对 $T_p$ 的注意力（IAT），并进一步用逐层阈值控制"何时干预"、用逐头放大矩阵控制"放大多少"（AdaIAT），在 LLaVA-1.5/Janus-Pro/Qwen2.5-VL 上把幻觉率（CS/CI）显著降低的同时几乎不损失文本多样性。
tags:
  - "CVPR 2026"
  - "幻觉检测"
  - "LVLM幻觉"
  - "注意力干预"
  - "自适应解码"
  - "生成文本注意力"
  - "免训练"
---

# AdaIAT: Adaptively Increasing Attention to Generated Text to Alleviate Hallucinations in LVLM

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhong_AdaIAT_Adaptively_Increasing_Attention_to_Generated_Text_to_Alleviate_Hallucinations_CVPR_2026_paper.html)  
**代码**: https://github.com/XianguiKang/AdaIAT.git  
**领域**: 多模态VLM / 幻觉缓解  
**关键词**: LVLM幻觉, 注意力干预, 自适应解码, 生成文本注意力, 免训练

## 一句话总结
针对"放大图像注意力虽能压幻觉、却让模型重复啰嗦"的痛点，本文发现真实物体 token 比幻觉 token 对**已生成文本** $T_p$ 的注意力更高，于是改为增大对 $T_p$ 的注意力（IAT），并进一步用逐层阈值控制"何时干预"、用逐头放大矩阵控制"放大多少"（AdaIAT），在 LLaVA-1.5/Janus-Pro/Qwen2.5-VL 上把幻觉率（CS/CI）显著降低的同时几乎不损失文本多样性。

## 研究背景与动机
**领域现状**：大视觉语言模型（LVLM）的幻觉——描述里出现图中根本不存在的物体——是落地的主要障碍。近期一类很受欢迎的免训练方法是**注意力干预**：研究者发现幻觉与"模型忽视图像、对图像 token $V$ 的注意力不足"高度相关，于是在推理时直接放大对 $V$ 的注意力权重，代表作有 PAI、HGAI。这类方法不需训练、推理开销低，是目前性价比最高的幻觉缓解路线。

**现有痛点**：放大图像注意力是把双刃剑。作者观察到，PAI/HGAI 在压低幻觉率的同时，会明显损害语言能力——具体表现为**重复啰嗦**：模型反复描述图中最显眼的物体（论文图 1 里反复说"clock tower""the motorcycle is parked on the street"），文本多样性指标 Distinct-1 在 LLaVA-1.5-7B 上掉了约 15%。

**核心矛盾**：为什么放大图像注意力会导致重复？因为注意力经过 softmax 是"零和"的——把权重大量分给图像 token $V$，相对就**压低了对已生成文本 $T_p$ 的注意力**。而 $T_p$ 承载着"我刚才说过什么"的上下文记忆。注意力一旦偏离 $T_p$，模型就"忘了"前文，只能对着图里最突出的物体一遍遍重说。于是幻觉率与语言连贯性之间形成了 trade-off。

**切入角度与核心 idea**：作者没有继续在 $V$ 上做文章，而是反问——能不能不动 $V$、改去放大对 $T_p$ 的注意力？支撑这个想法的是一个关键观察：统计 22,015 个真实物体 token 与 9,473 个幻觉物体 token 的注意力分布后发现，**真实物体 token 对 $T_p$ 的注意力显著高于幻觉物体 token（约 1.5×–2.5×），这个差距甚至比它们在图像 $V$ 上的差距（约 1.2×–1.5×）更明显**。原因在于：$V$ 是视觉编码器产出的、与文本异构、且含大量"与指令无关"的视觉信息；而 $T_p$ 是 LLM 在已看过图像和指令后亲自组织出来的输出，天然就是"指令相关、被压缩提纯"的视觉信息，且本就处于文本特征空间 $I_s$ 内，弥合了模态鸿沟。因此核心 idea 是：**用增大对 $T_p$ 的注意力来同时压幻觉、保连贯**——前者靠 $T_p$ 里浓缩的指令相关视觉信息支撑准确描述，后者靠 $T_p$ 里的上下文知识维持语言多样性。

## 方法详解

### 整体框架
方法建立在标准自回归解码之上，是一个**免训练、仅在前向推理时修改注意力权重**的干预策略，分两层递进。设第 $n$ 步要预测 token $t_{n+1}$，LLM 输入 $I$ 由系统提示 $S$、图像 token $V$、用户指令 $U$、已生成文本 $T_p=\{t_1,\dots,t_n\}$ 拼成；对第 $l$ 层第 $h$ 头，对 token $t_n$ 的自注意力为 $\boldsymbol{A}^{(l,h)}=\mathrm{softmax}(\tilde{\boldsymbol{A}}^{(l,h)})$，其中 $\tilde{\boldsymbol{A}}^{(l,h)}=\boldsymbol{Q}^{(l,h)}_{t_n}(\boldsymbol{K}^{(l,h)})^\top/\sqrt{d_k}$。

第一层是 **IAT（Increase Attention to $T_p$）**：在 LLM 的中间层（5–18 层）对所有指向 $T_p$ 的注意力做朴素放大，把 PAI 那套"放大图像注意力"的目标对象从 $V$ 换成 $T_p$。第二层是 **AdaIAT**：朴素放大有两个粗糙之处——不管当前有没有幻觉倾向都一直放大（破坏正常预测），以及对所有注意力头一视同仁地用固定放大系数（忽略了不同头的差异）。AdaIAT 用**逐层阈值**判断"这一层对 $T_p$ 的注意力是否真的不足、需不需要触发干预"，再用**逐头放大矩阵 $\mathcal{M}$** 给每个头分配定制化的放大幅度，让干预既精准又最小化对原生预测的扰动。整套方法的关键参数（阈值 $\mathcal{T}$、矩阵 $\mathcal{M}$）从 COCO 统计一次得到后固定，迁移到其他数据集时不再重算。

### 关键设计

**1. IAT：把放大对象从图像换成"已生成文本" $T_p$，一举压幻觉又防重复**

这一步直接回应"放大图像注意力会导致重复啰嗦"的痛点。作者不再去加大对 $V$ 的注意力，而是在中间层（$l\in(5,18)$）对所有 $\mathcal{I}(i)\in T_p$ 的注意力做朴素放大：

$$\tilde{\boldsymbol{A}}^{(l,h)}(i)=\tilde{\boldsymbol{A}}^{(l,h)}(i)+\alpha\cdot|\tilde{\boldsymbol{A}}^{(l,h)}(i)|$$

其中 $\alpha$ 是放大系数。与 PAI 唯一也是最本质的区别是：PAI 的 $\mathcal{I}(i)\in V$（放大图像），IAT 的 $\mathcal{I}(i)\in T_p$（放大已生成文本）。之所以有效，是因为 $T_p$ 本身就是模型在看过图像和指令后提纯出的"指令相关视觉信息"，放大它等于让模型在每个自回归步都更倚重这份浓缩的视觉先验，从而压低幻觉；同时 $T_p$ 含有上下文记忆，注意力留在它身上，模型就不会"忘了前文"去重复描述显眼物体。实验里 IAT 的 Distinct-1 分布与原始 Greedy 几乎重合、甚至在高分段（0.65–0.8）占比更高，而 PAI/HGAI 的分布明显左移（更重复），印证了这一点。只在 5–18 中间层干预也很关键：消融显示在 0–18 或 5–31 这种跨段干预会让模型直接崩溃（F1 跌到 30–48、D1 跌到 0.03–0.16）。

**2. 逐层阈值：只在"对 $T_p$ 注意力确实不足"时才触发干预**

朴素 IAT 不管模型当前是否要犯幻觉都一直放大——可幻觉是偶发的，多数时候模型预测是正常的，盲目放大会让正常预测时 $T_p$ 注意力异常偏高，反而损害准确率。为此 AdaIAT 引入逐层阈值 $\mathcal{T}\in\mathbb{R}^L$：

$$\mathcal{T}=\bar{\mathbf{A}}_{T_p}^{h}+\beta\,(\bar{\mathbf{A}}_{T_p}^{r}-\bar{\mathbf{A}}_{T_p}^{h})$$

其中 $\bar{\mathbf{A}}_{T_p}^{r}$、$\bar{\mathbf{A}}_{T_p}^{h}$ 分别是真实/幻觉物体生成时（在 COCO 上统计的）逐层平均 $T_p$ 注意力，$\beta$ 是平衡系数，相当于把触发线放在"幻觉态"与"真实态"之间按 $\beta$ 插值的位置。推理时若某层实际注意力 $\bar{\mathbf{A}}^{(l)}_{T_p}<\mathcal{T}(l)$，说明这一层对 $T_p$ 关注不够、有幻觉风险，就触发 IAT；若 $\bar{\mathbf{A}}^{(l)}_{T_p}\ge\mathcal{T}(l)$，说明注意力充足，保持正常解码、不做任何干预。$\beta$ 太小则干预太弱、压不住幻觉，太大则触发过于频繁、又破坏原生预测——消融里 $\beta$ 从 0.1 升到 0.5 幻觉持续下降，$>0.5$ 后 CS/CI 反弹且 F1/D1 下滑，故取 $\beta=0.5$。

**3. 逐头放大矩阵 $\mathcal{M}$：按每个注意力头的"幻觉敏感度"定制放大幅度**

固定 $\alpha$ 对所有头一刀切，但不同头在生成真实 vs 幻觉物体时的注意力差异很不一样——有的头差异巨大（生成幻觉时对 $T_p$ 关注严重不足），有的头差异很小。AdaIAT 据此构造逐头放大比例矩阵：

$$\mathcal{M}=\frac{\mathbf{A}^{r}_{T_p}}{\mathbf{A}^{h}_{T_p}},\qquad \mathcal{M}\in\mathbb{R}^{L\times H}$$

$\mathcal{M}^{(l,h)}$ 表示第 $l$ 层第 $h$ 头上"真实态/幻觉态"的平均 $T_p$ 注意力之比，它天然指向"把幻觉注意力模式拉回真实注意力模式"的方向。于是把 IAT 的放大改写为按头加权、并重归一化以保持概率和为 1：

$$\boldsymbol{A}^{(l,h)}(i)=\boldsymbol{A}^{(l,h)}(i)+\alpha\cdot\mathcal{M}^{(l,h)}\cdot\boldsymbol{A}^{(l,h)}(i),\quad \mathcal{I}(i)\in T_p$$

$$\boldsymbol{A}^{(l,h)}(k)=\frac{\boldsymbol{A}^{(l,h)}(k)}{\sum_k\mathbf{A}^{(l,h)}(k)},\quad k\in(1,len)$$

这样 $\mathcal{M}$ 大的头（幻觉时注意力亏空大）获得更强放大、被使劲拉回真实模式，$\mathcal{M}$ 小的头则维持弱放大、避免扰乱本就正常的注意力。注意这里干预的是 softmax **之后**的 $\boldsymbol{A}$（IAT 朴素版干预的是 softmax 之前的 $\tilde{\boldsymbol{A}}$），因为 $\mathcal{M}$ 是 $\boldsymbol{A}$ 上真实/幻觉之比，需在同一空间放大后再归一化。$\alpha$ 在此是整体强度，消融显示 $\alpha=6$ 时 F1 最高（79.4），$\alpha\ge 8$ 后 D1 开始恶化，故取 6。正是这套"逐头自适应"让 AdaIAT 在相近幻觉率与多样性下，F1 比 IAT 高 2.6，体现更强的预测能力。

## 实验关键数据

### 主实验
在三类代表性 LVLM（LLaVA-1.5-7B/13B、Janus-Pro-7B、Qwen2.5-VL-7B）上用 CHAIR 评测，CS（句级幻觉率）/CI（实例级幻觉率）越低越好，F1（物体描述准确丰富度）/D1（Distinct-1 文本多样性）越高越好。下表节选 LLaVA-1.5-7B 与 Janus-Pro-7B（基于 Greedy 解码）：

| 模型 | 方法 | CS ↓ | CI ↓ | F1 ↑ | D1 ↑ |
|------|------|------|------|------|------|
| LLaVA-1.5-7B | Greedy | 49.0 | 13.3 | 77.9 | 0.60 |
| LLaVA-1.5-7B | PAI | 31.8 | 7.8 | 77.7 | 0.50 |
| LLaVA-1.5-7B | HGAI | 31.4 | 6.9 | 78.3 | 0.50 |
| LLaVA-1.5-7B | IAT | 29.8 | 9.0 | 76.8 | 0.61 |
| LLaVA-1.5-7B | **AdaIAT** | 31.4 | 8.3 | **79.4** | 0.60 |
| Janus-Pro-7B | Greedy | 25.8 | 6.7 | 76.8 | 0.62 |
| Janus-Pro-7B | PAI | 20.4 | 5.6 | 76.1 | 0.61 |
| Janus-Pro-7B | HGAI | 21.0 | 5.3 | 75.9 | 0.62 |
| Janus-Pro-7B | **AdaIAT** | **19.0** | **4.9** | 76.5 | 0.64 |

关键对照：PAI/HGAI 在 LLaVA-1.5-7B 上把 D1 从 0.60 砸到 0.50（多样性掉约 15%），而 IAT/AdaIAT 维持在 0.60–0.61，幻觉率却相当甚至更低。相对 LLaVA-1.5-7B 的 Greedy，AdaIAT 把 CS 降 35.8%、CI 降 37.1%（论文摘要口径）。AdaIAT 在 Janus-Pro/Qwen2.5-VL 上同时拿到最低 CS/CI 与较高 F1，并能叠加到 Sample 解码上（AdaIAT† 在 Qwen2.5-VL 上 CS 42→34.6、F1 72.5→75.1）。

补充评测进一步佐证"保多样性"这一卖点：

| 评测 | 指标 | Greedy | PAI | HGAI | IAT | AdaIAT |
|------|------|--------|-----|------|-----|--------|
| OpenCHAIR | Co ↓ | 0.292 | 0.266 | 0.261 | 0.254 | **0.252** |
| OpenCHAIR | D1 ↑ | 0.61 | 0.51 | 0.53 | 0.61 | 0.61 |
| IIW-400 文本质量 | Bself ↓ | 0.058 | 0.242 | 0.247 | 0.090 | 0.071 |
| IIW-400 文本质量 | Bd ↑ | 59.48 | 56.80 | 56.61 | 58.99 | 58.29 |

### 消融实验

| 配置 | 关键现象 | 说明 |
|------|---------|------|
| IAT vs AdaIAT | 相近 CS/D1 下 F1 +2.6 | 逐头自适应放大带来预测能力增益 |
| $\alpha$（IAT）| $\alpha\ge0.8$ 后 F1/D1 骤降 | IAT 取 $\alpha=0.8$ |
| $\alpha$（AdaIAT）| $\alpha=6$ 时 F1 最高 79.4，$\ge8$ 后 D1 退化 | AdaIAT 取 $\alpha=6$ |
| $\beta$ | 0.1→0.5 幻觉持续降，>0.5 后 CS/CI 反弹 | 触发阈值，取 $\beta=0.5$ |
| 干预层（5–18 中间层）| 跨段 0–18/5–31 让模型崩溃（F1 30–48, D1 0.03–0.16）| 中间层最均衡 |

### 关键发现
- **逐头自适应是 AdaIAT 相对 IAT 的核心增量**：去掉它退化为 IAT，相近条件下 F1 掉 2.6，说明"不同头差异化放大"对维持原生预测确实有用。
- **干预层位极其敏感**：只在中间 5–18 层干预最稳；一旦把浅层与中/深层组合（0–18、5–31），IAT 会直接崩盘（D1 跌到 0.036），AdaIAT 也明显掉点。这提示放大对 $T_p$ 的注意力是把双刃剑，必须限制在"语义已成形、又未到输出锁定"的中间层。
- **$\beta$ 存在最优值而非越大越好**：触发越频繁 ≠ 效果越好，过度触发会让 $T_p$ 注意力异常偏高，反而升幻觉、掉 F1/D1，印证了"自适应触发时机"的必要性。

## 亮点与洞察
- **把"放大什么"从 $V$ 换成 $T_p$ 是四两拨千斤的视角转换**：同样是注意力干预、同样免训练，仅改变干预对象就同时解决了幻觉与重复，且有清晰的统计观察（真实 token 对 $T_p$ 的注意力 1.5–2.5× 高于幻觉 token）支撑，不是拍脑袋。
- **用"真实态/幻觉态"统计量同时驱动阈值与放大幅度**：$\mathcal{T}$（何时干预）和 $\mathcal{M}$（放大多少）都来自 $\mathbf{A}^r_{T_p}$ 与 $\mathbf{A}^h_{T_p}$ 这对离线统计量，思路统一——"把幻觉时的注意力分布拉回真实时的分布"。这种"用真实-幻觉差异图谱当导航"的做法可迁移到其他需要细粒度、逐头干预的解码任务。
- **解释了 PAI/HGAI 重复的成因（softmax 零和挤压 $T_p$）**，把一个经验现象讲成了机制，对后续注意力干预方法是有用的告诫。

## 局限性 / 可改进方向
- **阈值 $\mathcal{T}$ 与放大矩阵 $\mathcal{M}$ 依赖在 COCO 上的离线统计**：虽然论文显示固定后可跨数据集迁移，但其在与 COCO 分布差异很大的领域（医学、文档、长尾物体）上的鲁棒性未充分验证 ⚠️，统计样本的偏置可能被带进 $\mathcal{M}$。
- **干预层窗口（5–18）像是按 LLaVA-1.5-7B 经验定的超参**：换模型/换层数时这个窗口是否仍最优、需不需要重新搜索，论文未给自动化方案；消融也表明层位选错会导致崩溃，鲁棒性边界偏窄。
- **评测集中在图像详述（captioning）类任务与 CHAIR 系指标**：对 VQA、多轮对话、推理类任务的幻觉是否同样有效尚不清楚；GPT-4 辅助评测（HalluBench）只在 200 张 VG 图上做，规模有限。

## 相关工作与启发
- **vs PAI**：PAI 全局放大对图像 token $V$ 的注意力，本文放大对已生成文本 $T_p$ 的注意力——唯一本质差别就是干预对象 $\mathcal{I}(i)$ 从 $V$ 换成 $T_p$。优势是避免了 PAI 因挤压 $T_p$ 注意力导致的重复啰嗦（D1 不掉），代价是需要离线统计 $\mathcal{T}/\mathcal{M}$。
- **vs HGAI**：HGAI 也是放大图像注意力、并融合多头信息，同样有 D1 退化问题；AdaIAT 在 HSR/HWR 上略逊 HGAI 但 D1 远高，整体 trade-off 更均衡。
- **vs 解码对比类（VCD/AGLA）**：VCD 靠对比原始与扰动视觉输入的输出分布、AGLA 靠全局+局部注意力集成解码；它们不直接改注意力权重，开销与本文路线不同，AdaIAT 走的是更轻量的纯注意力放大路线。
- **vs 后处理类（LURE/Woodpecker）**：后者需训练改写器或调用专家模型做事后纠错，开销大；本文免训练、仅前向干预，推理成本低。

## 评分
- 新颖性: ⭐⭐⭐⭐ "放大对象从图像换成生成文本"是清晰且有观察支撑的新视角，但仍在注意力干预这一既有框架内。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 4 个 LVLM、4 类幻觉评测（CHAIR/OpenCHAIR/HalluBench/IIW-400）+ 充分消融，但任务类型偏 captioning。
- 写作质量: ⭐⭐⭐⭐ 动机—观察—方法递进清楚，公式与图示到位。
- 价值: ⭐⭐⭐⭐ 免训练、低开销、可叠加到不同解码，工程实用性强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Cross-Modal Attention Calibration for LVLM Hallucination Mitigation](cross-modal_attention_calibration_for_lvlm_hallucination_mitigation.md)
- [\[CVPR 2026\] Same Attention, Different Truths: Put Logit-Lens over Visual Attention to Detect and Mitigate LVLM Object Hallucination](same_attention_different_truths_put_logit-lens_over_visual_attention_to_detect_a.md)
- [\[CVPR 2026\] PAS: Prelim Attention Score for Detecting Object Hallucinations in Large Vision-Language Models](pas_prelim_attention_score_for_detecting_object_hallucinations_in_large_vision-l.md)
- [\[ICLR 2026\] SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense](../../ICLR2026/hallucination/shield_suppressing_hallucinations_in_lvlm_encoders_via_bias_and_vulnerability_de.md)
- [\[CVPR 2026\] Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations](beyond_global_scores_fine_grained_token_grounding_as_robust_detector_of_lvlm_hallucinations.md)

</div>

<!-- RELATED:END -->
