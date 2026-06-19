---
title: >-
  [论文解读] Towards Robust Multimodal Large Language Models Against Jailbreak Attacks
description: >-
  [CVPR 2026][AI安全][越狱防御] SAFEMLLM 是第一个直接对多模态大模型（MLLM）做对抗训练的越狱防御框架：它在 token 嵌入层注入一对可学习扰动矩阵来高效模拟跨模态攻击（CoE-Attack），再交替更新模型参数去抵消这些扰动，从而在白盒场景下把六种越狱攻击的成功率压到接近 0，同时几乎不损失正常多模态问答能力。
tags:
  - "CVPR 2026"
  - "AI安全"
  - "越狱防御"
  - "对抗训练"
  - "多模态大模型"
  - "token 嵌入扰动"
  - "对比损失"
---

# Towards Robust Multimodal Large Language Models Against Jailbreak Attacks

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Yin_Towards_Robust_Multimodal_Large_Language_Models_Against_Jailbreak_Attacks_CVPR_2026_paper.html)  
**代码**: https://github.com/ericyinyzy/SafeMLLM （论文称将开源）  
**领域**: AI 安全 / 多模态大模型越狱防御  
**关键词**: 越狱防御, 对抗训练, 多模态大模型, token 嵌入扰动, 对比损失

## 一句话总结
SAFEMLLM 是第一个直接对多模态大模型（MLLM）做对抗训练的越狱防御框架：它在 token 嵌入层注入一对可学习扰动矩阵来高效模拟跨模态攻击（CoE-Attack），再交替更新模型参数去抵消这些扰动，从而在白盒场景下把六种越狱攻击的成功率压到接近 0，同时几乎不损失正常多模态问答能力。

## 研究背景与动机

**领域现状**：MLLM 在视觉问答、图文理解等任务上很强，但也继承并放大了 LLM 的安全风险——越狱攻击（jailbreak）可以绕过安全护栏，诱导模型输出制造炸弹、违法建议等有害内容。现有防御主要走两条路：一是**推理期外挂模块**（用额外 LLM 当检测器、用奖励模型把解码分布往安全方向引导、给中间隐状态挂分类器判别有害性）；二是**安全对齐微调**（在「有害问题→拒答」的指令数据上微调，或做 RLHF）。

**现有痛点**：外挂模块有个致命前提——它必须对用户**保密**，否则攻击者拿到检测机制就能绕过；而且它只是在输出端拦截，并没有真正提升模型本身的安全性。安全微调方法（如 VLGuard）则在**白盒场景**下很脆：论文实测 VLGuard 在 LLaVA-1.5 上能挡住黑盒的 FigStep，却完全挡不住 ImgJP、GCG 这类拿到了梯度信息的白盒攻击（ASR 高达 79–88%）。

**核心矛盾**：要在白盒下（攻击者掌握参数和梯度）真正提升模型**内在**安全性，对抗训练（AT）是自然选择，但现成 AT 没法直接搬到 MLLM 上。closed-set 分类的 AT 不适用于开放式生成；把 LLM 上的潜在对抗训练（LAT）扩到 MLLM 又有两个坑：(1) 只在文本上加扰动，挡不住更强的连续值图像噪声；(2) 直接给每个 token 嵌入加扰动，计算开销爆炸——LLaVA-1.5-13B 一张图就占 576 个 image token，逐个扰动训练参数太多、又拖慢攻击优化。

**本文目标**：设计一个能在白盒下同时抵御图像、文本、图文三类越狱攻击，且不牺牲正常效用的对抗训练框架，还要解决 MLLM 上「图像 token 太多导致攻击优化又慢又弱」的效率难题。

**核心 idea**：不在像素或每个 token 上加扰动，而是在文本嵌入层放**两个紧凑的可学习扰动矩阵**（一个放在 query 前模拟对抗图像、一个放在 query 后模拟对抗文本后缀），用对比目标把它们优化成「最坏情况攻击」，再交替更新模型去中和它们——用 8 个扰动 token 替代 576 个图像 token，攻防都更快更强。

## 方法详解

### 整体框架
SAFEMLLM 是一个**两步交替**的对抗训练框架。给定参数为 $\theta$ 的良性 MLLM，目标是学到鲁棒参数 $\theta^*$，且把训练好的 $\theta^*$ 和梯度都公开给攻击者（真·白盒）。可训练参数 $\Delta\theta^*$ 只来自跨模态 adapter 和 LLM decoder（用 LoRA 优化），视觉编码器冻结。

每一轮迭代 $i$ 里：**Step I（攻击）** 固定模型参数，用 CoE-Attack 在 token 嵌入层优化出最强的对抗扰动 $\{P^h_M, P^t_M\}$，模拟跨模态越狱；**Step II（防御）** 固定这组扰动，更新模型参数去抵消它们的攻击效果，同时用一项效用损失保住正常图文问答的能力。更新后的模型再进入下一轮 Step I，如此迭代 $T$ 轮得到 $\theta^* = \theta_T$。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["良性 MLLM θ（视觉编码器冻结，<br/>adapter+decoder 用 LoRA 训）"] --> B["Step I：CoE-Attack 双 token 扰动注入<br/>Ph 放 query 前模拟对抗图像<br/>Pt 放 query 后模拟对抗后缀"]
    B --> C["对比式攻击目标<br/>L_adv = L_target + λ·L_contra<br/>M 步梯度上升求最坏扰动"]
    C -->|固定扰动 Ph_M, Pt_M| D["Step II：防御更新<br/>L_def 抵消扰动 + L_utility 保效用"]
    D -->|更新 θ_{i-1}→θ_i| B
    D --> E["鲁棒 MLLM θ*（公开参数+梯度）"]
```

### 关键设计

**1. CoE-Attack 的双 token 嵌入扰动：用 8 个 token 替代 576 个图像 token**

直接的最坏情况攻击是同时优化一张对抗图像 $I'$ 和一段文本后缀 $x'$，但这又慢又重——文本后缀要在整个词表上贪心搜索，图像扰动要过沉重的视觉编码器。CoE-Attack 的关键观察是：既然图像在所有 MLLM 里都被放在文本前面、后缀放在文本后面，那干脆**绕过模态本身，直接在文本嵌入层注入两个扰动矩阵** $P^h_0 \in \mathbb{R}^{K\times C}$ 和 $P^t_0 \in \mathbb{R}^{K\times C}$（$K$ 是 token 数，$C$ 是嵌入维度），$P^h_0$ 放在 query 前充当对抗图像、$P^t_0$ 放在 query 后充当对抗后缀。这样输入里就**省掉了真实的 $I'$ 和 $x'$**，直接对这两个矩阵做梯度优化。

每轮迭代会随新采样的恶意 query 集 $X_i$ 重新初始化这两个矩阵（从词嵌入里随机取）。这一设计的收益在效率实验里非常直接：LLaVA-1.5-13B 一张图本来要 576 个 token，SAFEMLLM 只用 8 个扰动 token，单轮攻击运行时间从「直接优化对抗图像」的 263.56 秒、LAT 的 192.39 秒压到 **38.70 秒**，显存也最低，且 ASR 不退化。

**2. 对比式攻击目标：既要诱导肯定回答，又要相对压低拒答概率**

强越狱攻击要同时满足两点：放大模型生成「攻击者想要的肯定回答 $c_n$」（如 "Sure, here are steps..."）的概率，压低生成「安全/拒答回答 $r_n$」的概率。第一点用 target 损失即可：

$$L_{\rm adv}^{\rm target} = -\sum_{n=1}^{N}\log\big[p(c_n \mid P^h_0, x_n, P^t_0)\big]$$

但第二点如果天真地直接最小化 $\log p(r_n\mid\cdot)$，惩罚太狠会让模型攻击后**输出乱码**。SAFEMLLM 改用**对比损失**，只是「相对地」让肯定语气 $c_n$ 压过拒答语气 $r_n$：

$$L_{\rm adv}^{\rm contra} = -\sum_{n=1}^{N}\log\sigma\Big[\log p(c_n\mid P^h_0,x_n,P^t_0) - \log p(r_n\mid P^h_0,x_n,P^t_0)\Big]$$

其中 $\sigma$ 是 Sigmoid。最终攻击目标 $L_{\rm adv} = L_{\rm adv}^{\rm target} + \lambda\cdot L_{\rm adv}^{\rm contra}$，对 $\{P^h, P^t\}$ 做 $M$ 步梯度下降得到最强扰动。这里还有个细节：$c_n$ 和 $r_n$ 都由 gpt-4-turbo 生成，并被显式要求「语义风格和结构多样」，避免模型只学会防御 "Sure, here it is..." 这一种固定前缀。消融显示去掉对比损失后 ASR 平均回升 13.67%，说明它对攻击强度贡献很大。

**3. Step II 防御更新：在抵消扰动的同时用效用损失防过度拒答**

拿到固定的最强扰动 $\{P^h_M, P^t_M\}$ 后，Step II 更新模型参数，目标是「中和扰动 + 保住正常能力」。防御损失对称地复用了攻击端的两项，只是把目标翻过来——让模型在带扰动的恶意输入下输出安全回答 $r_n$，并用对比损失让 $r_n$ 压过 $c_n$：

$$L_{\rm def} = L_{\rm def}^{\rm target} + \lambda\cdot L_{\rm def}^{\rm contra}$$

光这样会带来「过度拒答」副作用：模型对正常问题也一律拒答，效用崩盘。所以再加一项**效用损失**，在 $H$ 对良性图文样本上做标准的生成监督：

$$L_{\rm utility} = -\sum_{j=1}^{H}\log\big[p(y_j\mid I_j, q_j)\big]$$

最终用 $L_{\rm def} + L_{\rm utility}$ 一起更新 LoRA 参数。消融里去掉 $L_{\rm utility}$ 后 MM-Vet 效用分从 37.8 暴跌到 21.6（LLaVA-1.5），正是过度拒答的后果——这一项是「安全」和「可用」之间的关键平衡器。

### 损失函数 / 训练策略
- 攻击端：$L_{\rm adv} = L_{\rm adv}^{\rm target} + \lambda L_{\rm adv}^{\rm contra}$，对扰动矩阵做 $M$ 步梯度上升。
- 防御端：$L_{\rm def} + L_{\rm utility}$，更新 LoRA 参数（adapter + decoder），视觉编码器冻结。
- 外层交替 $T$ 轮；$c_n/r_n$ 由 gpt-4-turbo 生成多样化的肯定/拒答回答。

## 实验关键数据

### 主实验
六种攻击（图像类 ImgJP/VAA、文本类 GCG/AutoDAN、图文类 FigStep/MM-SafetyBench）× 六个 MLLM，主指标是攻击成功率 **ASR（%，越低越好）**，由 gpt-4-turbo 判定输出是否有害。下表摘录各攻击在六模型上的**平均 ASR**：

| 攻击（模态） | Original | R2D2 | CAT | SAFEMLLM |
|--------------|----------|------|-----|----------|
| ImgJP（图像） | 51.33 | 27.33 | 11.33 | **5.17** |
| VAA（图像） | 32.92 | 8.75 | 4.75 | **1.25** |
| GCG（文本） | 33.83 | 15.83 | 4.83 | **0.00** |
| AutoDAN（文本） | 66.67 | 32.33 | 22.33 | **1.33** |
| FigStep（图文） | 38.00 | 20.33 | 19.33 | **1.00** |
| MM-SafetyBench（图文） | 26.62 | 13.37 | 10.39 | **2.27** |

SAFEMLLM 在所有攻击上都把平均 ASR 压到个位数甚至 0，且在更大模型（13B vs 7B）上往往更鲁棒——作者归因于可训练参数更多、对抗训练更充分。值得注意的是基线 R2D2/CAT 是 LLM 端的对抗训练，对图文类攻击（FigStep/MM-Safety）几乎无效，因为它们只往文本注入有害内容；SAFEMLLM 的跨模态扰动设计因此展现出更强的泛化。

对比安全微调代表 VLGuard（LLaVA-1.5）：

| 攻击 | 模型 | VLGuard | SAFEMLLM |
|------|------|---------|----------|
| ImgJP | 7B / 13B | 88.00 / 36.00 | **6.00 / 0.00** |
| GCG | 7B / 13B | 79.00 / 26.00 | **0.00 / 0.00** |
| AutoDAN | 7B / 13B | 81.00 / 61.00 | **1.00 / 0.00** |
| FigStep | 7B / 13B | 2.00 / 0.00 | 0.00 / 0.00 |

VLGuard 对黑盒图文攻击尚可，但白盒攻击全面失守，印证了「安全微调挡不住白盒」的核心论点。

### 消融实验
在 13B 模型上用 ImgJP / AdvBench 攻击，逐个移除模块（"×" 表示去掉）：

| 移除的模块 | MiniGPT-4 ASR↓ | LLaVA-1.5 ASR↓ | 说明 |
|------------|----------------|----------------|------|
| 去 $P^h_0$ | 5.00 | 1.00 | 前置扰动比后置更关键（ImgJP 把噪声加在 query 前） |
| 去 $P^t_0$ | 2.00 | 0.00 | 后置后缀扰动 |
| 去 $L^{\rm contra}_{\rm adv}$（攻击端对比损失） | 8.00 | 0.00 | 攻击变弱 |
| 去 $L^{\rm contra}_{\rm def}$（防御端对比损失） | 23.00 | 0.00 | 影响最大，平均 ASR 回升 13.67% |
| **SAFEMLLM（完整）** | **0.00** | **0.00** | — |
| 去 $L_{\rm utility}$（效用，MM-Vet 分↑） | 7.2 | 21.6 | 过度拒答致效用崩 |
| SAFEMLLM（效用，MM-Vet 分↑） | 22.8 | 37.8 | 效用保住 |

### 关键发现
- **对比损失（尤其防御端的）是鲁棒性的主要来源**：去掉后 MiniGPT-4 的 ASR 从 0 飙到 23，平均回升 13.67%。
- **前置扰动 $P^h_0$ 比后置 $P^t_0$ 更重要**，因为 ImgJP 这类图像攻击总把噪声加在 query 之前，前置扰动正好对位模拟。
- **效用损失不可省**：去掉后过度拒答，MM-Vet 效用分几乎腰斩，说明安全和可用必须联合优化。
- **效率优势显著**：8 个扰动 token vs 576 图像 token，13B 上单轮 38.7 秒（直接优化对抗图像要 263.6 秒），且 ASR 不掉。
- LLaVA-1.5 对去模块不太敏感，作者归因于其 decoder 基于已安全对齐的 Vicuna-1.5。

## 亮点与洞察
- **「模态错位」的巧解**：与其去优化真实的对抗图像（要过重的视觉编码器、576 个 token），不如在文本嵌入层放两个扰动矩阵分别顶替「图像位」和「后缀位」。既统一了跨模态攻击，又把参数量和算力压到极低——这是全文最漂亮的一步。
- **攻防对称复用对比损失**：同一个对比目标，Step I 让肯定语气压过拒答（造最强攻击），Step II 反过来让拒答压过肯定（练防御），结构优雅且实证有效。
- **对比损失而非硬压拒答概率**：直接最小化拒答概率会让模型输出乱码，用「相对偏好」的对比形式是个可迁移的小技巧，适用于其他需要「软抑制某类输出」的对抗/对齐场景。
- **真白盒设定**：训练后参数和梯度都公开给攻击者，比那些「必须对用户保密」的外挂检测器更可信地证明了内在鲁棒性。

## 局限与展望
- 防御只在六种**已知**攻击上验证；对训练时未见过的新型越狱（尤其更复杂的图文组合攻击）能否泛化，论文未充分回答。
- 仍依赖 gpt-4-turbo 生成 $c_n/r_n$ 并做有害性判定，评测和数据构造都有第三方闭源模型依赖，可复现性和判定偏差值得警惕。
- 对抗训练需要可训练参数和梯度，属于「需重训」的防御，对只能 API 调用的闭源 MLLM 不适用。
- 扰动 token 数 $K$、对比系数 $\lambda$、迭代步数 $M/T$ 等超参的敏感性分析较少；不同模型上「去模块不敏感」可能恰恰说明部分模块的贡献依赖底座是否已安全对齐。
- 改进方向：把扰动空间从「query 前后两段」扩展到自适应位置/长度；引入对未知攻击的鲁棒性正则；以及探索更轻量的判定器替代 gpt-4-turbo。

## 相关工作与启发
- **vs VLGuard（安全微调）**：VLGuard 在「有害图文→安全回答」数据上微调，本质是模仿学习，白盒下被梯度攻击轻易绕过（ASR 79–88%）；SAFEMLLM 用对抗训练主动构造最坏扰动再抵消，白盒下 ASR≈0。区别在「被动对齐」vs「主动对抗」。
- **vs R2D2 / CAT（LLM 端对抗训练）**：它们只在 LLM decoder 的文本/潜在表示上加扰动，挡不住图像和图文攻击；SAFEMLLM 在嵌入层统一模拟跨模态扰动，泛化到全部六种攻击。
- **vs LAT（潜在对抗训练扩展）**：把 LAT 扩到 MLLM 要在多层、大量图像 token 上加扰动，又慢又因参数过多反而削弱攻击；SAFEMLLM 用 8 个嵌入 token 实现更强更快的攻击，效率实验直接对比胜出。
- **vs 推理期外挂检测器**：那类方法必须对用户保密、只在输出端拦截；SAFEMLLM 提升的是模型内在安全性，参数公开仍鲁棒，思路上更彻底。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一个直接对 MLLM 做对抗训练的越狱防御，双 token 嵌入扰动的设计巧妙且切中 MLLM 的效率痛点。
- 实验充分度: ⭐⭐⭐⭐ 六攻击×六模型、对比安全微调/LLM 对抗训练、效率/人评/消融齐全；但缺未知攻击泛化与超参敏感性。
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰、攻防对称结构讲得明白；公式排版在缓存里略乱但逻辑完整。
- 价值: ⭐⭐⭐⭐ 给开源 MLLM 提供了实用、可重训的白盒鲁棒化方案，对 AI 安全落地有直接意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)
- [\[CVPR 2026\] FVBench: Benchmarking Deepfake Video Detection Capability of Large Multimodal Models](fvbench_benchmarking_deepfake_video_detection_capability_of_large_multimodal_mod.md)
- [\[CVPR 2026\] Hierarchically Robust Zero-shot Vision-language Models](hierarchically_robust_zero-shot_vision-language_models.md)
- [\[CVPR 2026\] SIF: Semantically In-Distribution Fingerprints for Large Vision-Language Models](sif_semantically_in-distribution_fingerprints_for_large_vision-language_models.md)
- [\[CVPR 2026\] VCP-Attack: Visual-Contrastive Projection for Transferable Black-Box Targeted Attacks on Large Vision-Language Models](vcp-attack_visual-contrastive_projection_for_transferable_black-box_targeted_att.md)

</div>

<!-- RELATED:END -->
