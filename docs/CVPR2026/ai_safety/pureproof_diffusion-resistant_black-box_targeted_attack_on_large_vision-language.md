---
title: >-
  [论文解读] PureProof: Diffusion-Resistant Black-box Targeted Attack on Large Vision-Language Models
description: >-
  [CVPR 2026][AI安全][VLM 安全] PureProof 是首个能扛住"扩散净化（DBP）"防御的黑盒定向攻击：它用一个扩散代理只跑单步反向预测来对齐目标语义（SRA）、用时间步自适应再加噪稳定梯度（ARA）、再用自一致正则约束局部连贯（SCR），让对抗图像在被 DiffPure 等净化后仍能诱导 VLM 输出攻击者指定的目标文本。
tags:
  - "CVPR 2026"
  - "AI安全"
  - "VLM 安全"
  - "定向对抗攻击"
  - "黑盒攻击"
  - "扩散净化"
  - "对抗鲁棒性"
---

# PureProof: Diffusion-Resistant Black-box Targeted Attack on Large Vision-Language Models

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Cao_PureProof_Diffusion-Resistant_Black-box_Targeted_Attack_on_Large_Vision-Language_Models_CVPR_2026_paper.html)  
**代码**: 待确认  
**领域**: AI 安全 / 对抗攻击  
**关键词**: VLM 安全, 定向对抗攻击, 黑盒攻击, 扩散净化, 对抗鲁棒性

## 一句话总结
PureProof 是首个能扛住"扩散净化（DBP）"防御的黑盒定向攻击：它用一个扩散代理只跑单步反向预测来对齐目标语义（SRA）、用时间步自适应再加噪稳定梯度（ARA）、再用自一致正则约束局部连贯（SCR），让对抗图像在被 DiffPure 等净化后仍能诱导 VLM 输出攻击者指定的目标文本。

## 研究背景与动机

**领域现状**：大型视觉-语言模型（VLM）正被大量部署到 AI agent 等场景，但对"定向对抗攻击"（用人眼不可察的扰动诱导模型输出指定文本）很脆弱。然而绝大多数攻击研究都在**无防御**设定下评测，实用相关性存疑。在图像模态上，**扩散净化（Diffusion-Based Purification, DBP）** 是最主流、最有效的黑盒防御——用扩散模型的"前向加噪 + 反向去噪"把对抗扰动洗掉，已被 BlueSuffix 等 VLM 防御框架集成。

**现有痛点**：现有定向攻击（MF-it/MF-ii、CoA、AnyAttack、FOA 等）一旦遇到 DBP，扰动几乎被完全净化，模型恢复良性输出——实测这些攻击对 DBP 的 ASR（Target）几乎全为 0%。

**核心矛盾**：能绕过 DBP 的现有 evasion 攻击（DiffAttack、DiffHammer）有两个根本问题。① 它们要**反向传播穿过整条扩散去噪轨迹**，计算图极深、还容易梯度消失/爆炸；攻击 VLM 时还要叠上 CLIP 这类大代理编码器，成本爆炸。② 它们没能妥善处理扩散过程**固有的随机性**，导致梯度噪声大、优化不稳定。而且这些方法只面向白盒图像分类器，并不适配"既看不到 VLM 也看不到净化器"的黑盒 VLM 定向攻击。

**本文目标**：在 VLM 和 DBP 净化器**都不可见**的黑盒威胁模型下，造出经 DBP 净化后仍能诱导目标输出的对抗图像，同时避开全轨迹反传的高成本与梯度不稳。

**切入角度**：与其辛苦反传整条去噪链，不如借一个**扩散代理**在随机时间步只跑**单步反向预测**，闭式算出"干净图预览" $\hat x_0$，直接拿它对齐目标语义。

**核心 idea**：用单步反向的干净图预览对齐目标（SRA）替代全链反传，再用自适应再加噪（ARA）和自一致正则（SCR）专门压住扩散随机性带来的梯度抖动。

## 方法详解

### 整体框架
威胁模型：攻击者拿不到受害 VLM $M$、也拿不到具体 DBP 净化器 $P$（用户上传图先被自动净化再喂给模型）。目标是在 $\ell_\infty$ 预算 $\|\epsilon\|_\infty\le\varepsilon$ 内构造 $x_{adv}=x_{clean}+\epsilon$，使 $M(P(x_{adv}),t_{in})=t_{tar}$——即净化后仍输出攻击者指定的目标文本 $t_{tar}$。目标图 $x_{tar}$ 由公开文生图模型（Stable Diffusion）从 $t_{tar}$ 生成。

PureProof 的优化在每次迭代里走同一条"单步管线"：把当前对抗图 $x_{adv}$ **前向加噪到随机采样的时间步** $t\sim\mathrm{Unif}\{1,\dots,T_p\}$ 得 $x_t$，用扩散代理做**一步反向去噪**闭式预测干净图 $\hat x_0(x_t,t)$，然后在 $\hat x_0$ 上施加三项损失：SRA 把 $\hat x_0$ 对齐目标图、ARA 对 $\hat x_0$ 再加噪取均值平滑、SCR 约束两次干净图估计一致；三项合成总损失后用 PGD 更新 $x_{adv}$。整条管线不穿越完整去噪轨迹，因此既省算又稳。由于核心贡献是三项损失项（作用于同一单步预测），按惯例不再画 pipeline 框图，下面用公式逐项讲清。

### 关键设计

**1. Stochastic Reverse Alignment（SRA）：用单步反向的"干净图预览"对齐目标，替代全链反传**

针对"反传穿整条扩散链导致计算图极深、梯度消失/爆炸"的痛点，SRA 不再模拟完整去噪，而是用代理扩散模型只跑**一步**反向。每次迭代把 $x_{adv}$ 前向加噪到随机时间步 $t$ 得 $x_t$，再用 DDPM 闭式公式直接预测干净图：$\hat x_0(x_t,t)=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}$（⚠️ 缓存中扩散系数符号存在 OCR 噪声，记号以原文为准）。然后把这个预览与目标图 $x_{tar}$ 在预训练编码器（如 CLIP）嵌入空间的余弦相似度上对齐：$L_{SRA}=-\mathbb{E}_t[\mathrm{sim}(\hat x_0(x_t,t),x_{tar})]$，外层期望靠每步均匀采样时间步近似。因为 $\hat x_0$ 直接落在模型的去噪方向上、且只过一步，SRA 既绕开了全链反传的算力黑洞，又规避了深层计算图的梯度消失/爆炸，给出更稳的梯度估计。实测单步比全链反传的 DA-cos 快约 25 倍（1.761s/step vs 44.717s/step）。

**2. Adaptive Re-noising Augmentation（ARA）：按时间步自适应再加噪，把损失面变成曲率感知的平滑正则**

扩散的随机性随时间步 $t$ 增大而增强（注入的高斯噪声逐渐压过信号），使单步反向预测高方差、有偏，梯度信号不可靠。ARA 的做法是：对每个采样时间步 $t$，把预测的 $\hat x_0$ 按**前向同等噪声水平**再注入 $K$ 个独立高斯噪声生成增广变体 $\tilde x_t^{(k)}=\sqrt{\bar\alpha_t}\,\hat x_0+\sqrt{1-\bar\alpha_t}\,\epsilon^{(k)}$，再对这些变体与目标的相似度取均值：$L_{ARA}=-\mathbb{E}_t\big[\frac1K\sum_{k=1}^K\mathrm{sim}(\tilde x_t^{(k)},x_{tar})\big]$。时间步越大、加噪越强，正则也成比例更强，正好对应"$\hat x_0$ 越不可靠时越要平滑"。作者还给了 Theorem 1：对 $\phi(x)=-\mathrm{sim}(x,x_{tar})$ 做二阶 Taylor 展开取期望，一阶项因 $\mathbb{E}[\epsilon]=0$ 消失，留下 $\mathbb{E}[\phi(\tilde x_t)]=\phi(\mathbb{E}[\tilde x_t])+\frac12\sigma_t^2\,\mathrm{tr}(H_\phi)+R_t$，即 ARA 等价于一个**曲率感知正则**——惩罚大曲率 $\mathrm{tr}(H_\phi)$ 区域、把梯度方向引向损失面稳定区，且 $\sigma_t^2=1-\bar\alpha_t$ 随 $t$ 增大自动加强正则、无需额外超参。消融显示 ARA 是三项里贡献最大的组件。

**3. Self-Consistency Regularization（SCR）：约束两次干净图估计一致，提升局部时序连贯**

为让对抗更新停留在扩散流形的局部一致区域，SCR 对 $\hat x_0(x_t,t)$ 按前向同等噪声水平再加噪得 $\tilde x_t'=\sqrt{\bar\alpha_t}\,\hat x_0+\sqrt{1-\bar\alpha_t}\,\epsilon'$，再做一次单步反向去噪得新的干净图估计 $\hat x_0'=\hat x_0(\tilde x_t',t)$，然后惩罚两次估计的差异：$L_{SCR}=\mathbb{E}_t[\gamma_t\cdot\|\hat x_0'-\hat x_0(x_t,t)\|_2^2]$，其中 $\gamma_t=1-t/T_p$ 在后期时间步下调权重（此时扩散噪声占主导、一致性约束意义减弱）。SCR 让相邻的干净图估计彼此连贯，进一步增强对抗样本对随机净化轨迹的鲁棒性。

### 损失函数 / 训练策略
总目标把三项组合：$L_{PureProof}=\beta\cdot L_{SRA}+(1-\beta)\cdot L_{ARA}+L_{SCR}$（⚠️ 缓存中权重记号被 OCR 成 ε，按上下文应为平衡系数 $\beta$，取值 0.3，以原文为准）。优化用 PGD 跑 100 步、步长 1/255、$\ell_\infty$ 预算 $\varepsilon=16$。代理设置：用三个 CLIP 编码器（ViT-B/16、ViT-B/32、ViT-g-14-laion2B）做集成以增强迁移性；要绕 DBP 的攻击额外用 Guided Diffusion 当扩散代理、EOT=10 近似其去噪行为；时间步上界 $T_p=150$、再加噪变体数 $K=3$。

## 实验关键数据

> 评测指标：**Ensemble CLIP Score**（用 RN50/RN101/ViT-B/32/ViT-B/16/ViT-L/14 五个 CLIP 文本编码器算输出与目标文本相似度后取均值，越高越好）；**ASR（Target）** = 完全成功攻击的比例，**ASR（Fool）** = 完全成功 + 仅被误导两类之和（由 GPT-4 判定）。因 DBP 有随机性，按 N=10 评测取均值/多数投票。

### 主实验

开源 VLM 对抗三种 DBP 防御（节选 DiffPure 与 LM 列，CLIP Score / ASR-Target / ASR-Fool %）：

| VLM | 方法 | DiffPure CLIP | DiffPure Tgt | DiffPure Fool | LM CLIP | LM Tgt | LM Fool |
|-----|------|--------------|--------------|---------------|---------|--------|---------|
| LLaVA-1.5 | DH-cos | 0.5047 | 0.0 | 41.4 | 0.4932 | 0.0 | 23.8 |
| LLaVA-1.5 | **PureProof** | **0.5983** | **12.3** | **76.8** | **0.6231** | **18.6** | **77.3** |
| LLaVA-1.6 | DH-cos | 0.4734 | 0.0 | 41.8 | 0.4670 | 1.6 | 25.8 |
| LLaVA-1.6 | **PureProof** | **0.5647** | **17.8** | **78.1** | **0.5830** | **22.5** | **84.4** |
| Gemma 3 | DH-cos | 0.4487 | 0.0 | 36.7 | 0.4412 | 0.0 | 36.5 |
| Gemma 3 | **PureProof** | **0.5231** | **8.6** | **85.4** | **0.5410** | **20.7** | **87.7** |
| Qwen3-VL | DH-cos | 0.3699 | 1.4 | 38.3 | 0.3452 | 0.0 | 22.3 |
| Qwen3-VL | **PureProof** | **0.4493** | **13.3** | **77.5** | **0.4613** | **25.6** | **81.4** |

所有无"扩散感知"的基线对 DBP 的 ASR（Target）全为 0%，凸显 DBP 防御力之强与旧攻击之脆；PureProof 在 Qwen3-VL/LM 上把 ASR（Target）拉到 25.6%，ASR（Fool）相比无扩散感知基线普遍提升 50% 以上。

商业 VLM 对抗 DiffPure（CLIP / ASR-Target / ASR-Fool %）：

| VLM | 方法 | CLIP | Target | Fool |
|-----|------|------|--------|------|
| GPT-5 | DH-cos | 0.4633 | 0.0 | 29.0 |
| GPT-5 | **PureProof** | **0.5457** | **11.0** | **77.0** |
| Gemini-2.5 | DH-cos | 0.4610 | 0.0 | 37.0 |
| Gemini-2.5 | **PureProof** | **0.5287** | **11.0** | **73.0** |

### 消融实验

| 配置 / 分析 | 关键结果 | 说明 |
|------|---------|------|
| 完整 $L_{PureProof}$ | 三项全开最优 | SRA+ARA+SCR |
| 去掉各损失项 | 各项均掉点，**ARA 贡献最大** | ARA 平滑损失面、稳梯度 |
| ARA 变体数 $K$ | $K{:}0{\to}1$ 大涨，$K{\ge}2$ 趋稳 | 少量增广即够，最终取 $K=3$ |
| 单步耗时 | PureProof 1.761s vs DA-cos 44.717s | 单步反向 ≈25× 提速 |
| 抗高斯噪声 (σ=16/255) | LLaVA-1.5 0.6757 / Qwen3-VL 0.5264，均最高 | 隐式考虑噪声变体，抗后处理 |

### 关键发现
- **旧攻击对 DBP 几乎全军覆没**：无扩散感知的 MF/CoA/AnyAttack/FOA 在所有 DBP 下 ASR（Target）= 0%，证明 DBP 是真正有效的黑盒防御，也说明此前"无防御"评测严重高估了攻击实用性。
- **ARA 是稳梯度的主力**：消融里去掉 ARA 掉点最多，印证其"曲率感知平滑"对扩散随机性下的优化最关键。
- **强迁移 + 强抗噪**：PureProof 在高斯噪声扰动下 CLIP Score 几乎不掉（MiniGPT-4 上 σ 从 8/255→16/255 仅 0.6342→0.6315），而 CoA 从 0.6095→0.4969，说明其对抗样本对一般后处理也内在鲁棒。
- **无防御下也有竞争力**：纯无防御设定下 PureProof 在 GPT-5 上 CLIP 最高；接入 CoA 目标（PureProof+CoA）后全面登顶，说明框架可灵活换损失目标。

## 亮点与洞察
- **"单步反向预览"是核心提效点**：用 DDPM 闭式 $\hat x_0$ 把"对齐净化输出"近似成"对齐单步去噪预览"，一举绕开全链反传的深计算图与梯度爆炸，约 25× 提速且更稳——这个"用单步预览代替全轨迹"的思路可迁移到其他需要穿越扩散链的优化问题。
- **再加噪当曲率正则有理论支撑**：ARA 不是经验 trick，Theorem 1 证明它二阶展开后等价于按 $\sigma_t^2$ 自适应的曲率惩罚，把"扩散越乱越要平滑"写进了梯度里且零额外超参。
- **首个真正扛 DBP 的黑盒 VLM 定向攻击**：揭示了"集成 DBP 的 VLM 防御框架"并非铁壁，对评估真实部署（含 agent 场景）的安全性很有警示价值。
- **抗高斯噪声是免费副产品**：因优化隐式遍历再加噪变体，对抗样本天然抗一般后处理扰动，这点对现实部署威胁更大。

## 局限与展望
- **依赖代理质量**：⚠️ 攻击效果依赖扩散代理（Guided Diffusion）与 CLIP 集成代理对真实净化器/受害模型的近似程度；面对与代理差异很大的未知 DBP，迁移性仍是开放问题（论文用 EOT=10 缓解但未穷尽）。
- **ASR（Target）绝对值仍不高**：即便 SOTA，完全成功率多在 8–25% 区间，多数收益体现在 ASR（Fool）上，说明"精确诱导指定文本"在 DBP 下依然很难。
- **作为攻击的双刃性**：该工作本质暴露 VLM 安全漏洞，需配套防御研究；作者将其定位为"促进更严谨防御评估"。
- **改进方向**：可探索更通用的扩散代理蒸馏、或把 SRA/ARA 推广到多步少量反向以在效率与保真间取更优折中。

## 相关工作与启发
- **vs DiffAttack / DiffHammer**：二者是面向白盒分类器的 DBP evasion，靠穿整条扩散链反传（DiffHammer 用选择性 EM 缓解随机性）；PureProof 只跑单步反向、面向黑盒 VLM 定向攻击，既省算又稳，实测大幅超过把它们改写成迁移攻击的 DA-cos/DH-cos。
- **vs AttackVLM (MF-it/MF-ii) / CoA / FOA**：这些是无扩散感知的 VLM 定向攻击，在无防御下尚可、遇 DBP 即 0% ASR（Target）；PureProof 专门把"扩散感知"注入优化，是它们在防御场景的根本性升级。
- **vs BPDA / EOT 等稳梯度技巧**：BPDA 的恒等近似对 DBP 无效、EOT 仅经验稳梯度；ARA 用带理论保证的自适应曲率正则取而代之，对未见净化轨迹更稳健。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个扛 DBP 的黑盒 VLM 定向攻击，"单步反向 + 曲率感知再加噪"组合新颖且有理论支撑。
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 5 个开源 + 2 个商业 VLM、3 种 DBP、抗高斯噪声/无防御/耗时/超参全面消融，对比基线充分。
- 写作质量: ⭐⭐⭐⭐ 动机—方法—理论链条清晰，Theorem 1 提供了 ARA 的理论解释；公式符号偏密。
- 价值: ⭐⭐⭐⭐ 戳破"集成 DBP 即安全"的假象，对真实 VLM/agent 部署的安全评估有重要警示意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] VCP-Attack: Visual-Contrastive Projection for Transferable Black-Box Targeted Attacks on Large Vision-Language Models](vcp-attack_visual-contrastive_projection_for_transferable_black-box_targeted_att.md)
- [\[CVPR 2026\] SIF: Semantically In-Distribution Fingerprints for Large Vision-Language Models](sif_semantically_in-distribution_fingerprints_for_large_vision-language_models.md)
- [\[CVPR 2026\] What Your Features Reveal: Data-Efficient Black-Box Feature Inversion Attack for Split DNNs](what_your_features_reveal_data-efficient_black-box_feature_inversion_attack_for_.md)
- [\[CVPR 2026\] PROMPTMINER: Black-Box Prompt Stealing against Text-to-Image Generative Models via Reinforcement Learning and VLM-Guided Optimization](promptminer_black-box_prompt_stealing_against_text-to-image_generative_models_vi.md)
- [\[CVPR 2026\] Unlearning without Forgetting: Securely Removing Targeted Concepts from Large-Scale Vision-Language Open-Vocabulary Detectors](unlearning_without_forgetting_securely_removing_targeted_concepts_from_large-sca.md)

</div>

<!-- RELATED:END -->
