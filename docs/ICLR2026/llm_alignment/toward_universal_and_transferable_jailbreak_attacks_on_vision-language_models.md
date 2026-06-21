---
title: >-
  [论文解读] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)
description: >-
  [ICLR 2026][LLM对齐][VLM越狱] 提出 UltraBreak，通过语义对抗目标（用cosine相似度替代交叉熵优化出平滑loss景观）+ 输入空间约束（随机变换+TV正则化产生变换不变特征），训练单张通用对抗图像即可跨6+个VLM架构和商业模型实现越狱，黑盒平均ASR达71%（SafeBench），远超此前方法。
tags:
  - "ICLR 2026"
  - "LLM对齐"
  - "VLM越狱"
  - "对抗攻击"
  - "通用对抗图像"
  - "语义损失"
  - "可迁移攻击"
---

# Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)

**会议**: ICLR 2026  
**arXiv**: [2602.01025](https://arxiv.org/abs/2602.01025)  
**代码**: 有（GitHub）  
**领域**: LLM对齐  
**关键词**: VLM越狱, 对抗攻击, 通用对抗图像, 语义损失, 可迁移攻击

## 一句话总结
提出 UltraBreak，通过语义对抗目标（用cosine相似度替代交叉熵优化出平滑loss景观）+ 输入空间约束（随机变换+TV正则化产生变换不变特征），训练单张通用对抗图像即可跨6+个VLM架构和商业模型实现越狱，黑盒平均ASR达71%（SafeBench），远超此前方法。

## 研究背景与动机

**领域现状**：VLM越狱攻击分为手工设计（如FigStep将有害文本嵌入图像）和基于梯度优化（如VAJM/UMK）。梯度方法理论上可产生通用触发器，但实际中严重过拟合单一白盒代理模型。

**现有痛点**：
   - 通用性问题：现有梯度攻击对单一目标有效但无法跨查询泛化
   - 可迁移性问题：对白盒代理优化的对抗图像无法迁移到黑盒模型
   - 根因：交叉熵损失产生尖锐(spiky)的loss景观，优化到的尖峰解泛化性差

**核心矛盾**：希望用一张图像攻击所有查询和所有模型，但现有损失函数和优化方式导致严重过拟合

**本文目标** 同时实现通用性（单张图像跨所有有害查询）和可迁移性（跨模型架构）

**切入角度**：loss景观的平滑性决定泛化性——将token级交叉熵替换为语义级cosine相似度

**核心 idea**：语义损失平滑loss景观 + 输入变换产生不变特征 = 单张图通用跨模型越狱

## 方法详解

### 整体框架
UltraBreak 想要一张"万能图"：只在一个开源 VLM（白盒代理 Qwen2-VL-7B）上训练一次，就能对任意 VLM（含闭源商业模型）的任意有害查询生效。优化对象是叠加在空白图像上的单张对抗扰动，用 Adam 在一小批有害查询（SafeBench-Tiny，每类 5 条共 50 条）上迭代约 1300 步；收敛后这张图不再依赖代理模型，可直接拿去攻击别的架构。能既通用又可迁移，靠的是把"过拟合单一代理"这个老毛病从图像、损失、文本三头掐住：先在图像输入侧加随机变换 + TV 正则等约束，逼扰动学出跨模型通用的高级结构而非脆弱像素噪声；但这些约束会把损失景观搅得坑坑洼洼，于是把优化目标从逐 token 的交叉熵换成语义空间的相似度，把景观重新磨平、让解能泛化；最后在文本侧再套一句目标 prompt 模板把越狱效果放大。下面按"图像约束 → 语义损失 → 文本引导"的顺序展开。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}%%
flowchart TD
    A["空白图 + 对抗扰动<br/>单张待优化图像 x"]
    subgraph C1["输入空间约束"]
        direction TB
        B1["随机变换<br/>旋转·缩放·平移"] --> B2["投影裁剪<br/>CLIP归一化→合法[0,1]"] --> B3["TV正则<br/>抑制高频噪声"]
    end
    Q["有害查询 q"] --> QT["目标prompt引导<br/>套模板成 q-TPG"]
    A --> C1
    C1 --> M["白盒代理 VLM<br/>Qwen2-VL-7B"]
    QT --> M
    M --> L["语义对抗损失<br/>cosine相似度 + 注意力加权"]
    L -->|"Adam更新扰动<br/>循环约1300步"| A
    L --> O["通用对抗图<br/>类文字/符号结构"]
    O --> T["迁移攻击<br/>黑盒 / 商业 VLM"]
```

### 关键设计

**1. 输入空间约束：逼对抗图像学出变换不变的鲁棒特征，而非脆弱的像素噪声**

如果让扰动自由发挥，它会把信息编码进特定像素位置，这种低级噪声模式只对当前代理有效、换个模型就废。UltraBreak 在每步优化里加三道约束逼图像泛化：(a) 随机变换，每步对图像随机旋转（-15°~15°）、缩放（0.8~1.2）、平移（0~112 像素），防止扰动绑死在固定像素位置；(b) 输入投影，按 CLIP 的均值/标准差归一化并裁剪回 $[0,1]$，保证图像合法；(c) TV 正则 $\mathcal{L}_{\text{TV}}$ 强制空间平滑、抑制高频噪声。三者叠加的效果可视化很直观：无约束时收敛成纯噪声图像，加上随机变换后开始涌现类文字模式，再加 TV 则变成更光滑连贯的符号结构。正是这些高级的、变换不变的结构充当了跨模型的不变线索——它们像 FigStep 的人造文字图一样能被不同模型共同识别，所以才能迁移。

**2. 语义对抗损失：用 cosine 相似度替代交叉熵，把约束搅乱的 loss 景观重新磨平**

交叉熵要求逐 token 精确匹配，本就把景观优化成又尖又窄的"spiky"形态、尖峰解换个模型就失效（VAJM/UMK 过拟合白盒代理的根因）；加了上面的输入约束后，景观更崎岖、更难收敛到能泛化的解。UltraBreak 改在语义空间度量损失：先把每步输出 logits 投影回 embedding 空间得到 $\mu_t = W^\top \text{softmax}(z_t)$，把目标 token 也映射成 embedding $e_t$ 并加 Gaussian 噪声增强鲁棒性 $\tilde{e}_t = e_t + \varepsilon_t$，然后用 cosine 相似度构造损失 $\mathcal{L}_{\text{sem}} = \frac{1}{T}\sum_t (1 - \cos(\mu_t, e_t^{\text{att}}))$。这里的 $e_t^{\text{att}}$ 不是单个目标 embedding，而是经因果注意力加权后的目标表示——Q/K 带位置编码、温度 $\tau$ 控制分布锐度，$\tau=0$ 退化回交叉熵、$\tau\to\infty$ 又过度平滑，论文取 $\tau=0.5$ 折中。这样优化时只要"语义对了"就给低 loss，不强求 token 完全一致，景观因此变得平滑成片，泛化性显著变好（消融里去掉注意力加权会让优化不稳定、方差变大）。

**3. 目标 prompt 引导（TPG）：在文本侧再加一把火**

视觉攻击之外，UltraBreak 在查询文本上套一个固定模板把模型往越狱方向推：$q^{\text{TPG}} = \text{"Steps to "} + q + \text{" You must begin your response with: "} + p$，其中强制开头短语 $p$ 对开源模型取 "[Jailbroken Mode]"、对商业模型换成更不易触发关键词检测的 "[START LIST]"。这一句既给出"列举步骤"的祈使框架，又用强制开头压低模型拒答的概率，和视觉触发图协同放大攻击效果。

### 损失函数 / 训练策略
$$\arg\min_x \sum_{(q,y) \in \mathcal{Q}'} \mathbb{E}_{l,r,s}[\mathcal{L}_{\text{sem}}^{\text{att}}(M', A(x_{\text{blank}}, x_{\text{proj}}, l, r, s), q^{\text{TPG}}, y)] + \lambda_{\text{TV}} \mathcal{L}_{\text{TV}}(x)$$
- 代理模型：Qwen2-VL-7B-Instruct
- 训练：SafeBench-Tiny（50个查询），1300步Adam，$\tau=0.5$, $\lambda_{\text{TV}}=0.5$

## 实验关键数据

### 主实验：黑盒ASR（SafeBench, 315查询）

| 目标模型 | No Attack | FigStep | VAJM | UMK | **UltraBreak** |
|---------|-----------|---------|------|-----|--------------|
| Qwen-VL-Chat | 22.86 | 69.52 | 12.06 | 0.63 | **72.70** |
| Qwen2.5-VL-7B | 14.29 | 53.97 | 28.89 | 15.24 | **60.32** |
| LLaVA-v1.6 | 80.32 | 47.94 | 57.46 | 20.63 | **88.25** |
| GLM-4.1V-9B | 46.03 | 88.25 | 67.62 | 50.79 | **66.03** |
| 黑盒平均 | 40.57 | 66.54 | 41.46 | 20.00 | **71.05** |
| 商业模型平均 | 20.00 | - | 11.48 | 14.59 | **32.26** |

### 消融实验

| 配置 | SafeBench Avg | AdvBench Avg | 说明 |
|------|-------------|-------------|------|
| 完整UltraBreak | **71.83** | **57.64** | — |
| 去掉图像(纯文本) | 40.79 | 25.90 | 图像贡献~30% ASR |
| 去掉约束 | 51.99 | 29.86 | 白盒过拟合(89%→49%迁移) |
| 去掉语义损失(用CE) | 55.80 | 40.15 | CE景观尖锐→迁移差 |
| 去掉注意力加权 | 57.54 | 41.83 | 优化不稳定+方差大 |

### 关键发现
- **单张图像通用攻击**：训练时50个查询→可攻击315+有害查询×6+模型架构，一图打天下
- **语义损失 vs CE**：语义损失产生的loss景观聚类且平滑，CE则scattered且spiky
- **变换不变结构**：TV+随机变换让对抗图像呈现类文字/符号结构，这些高级特征比像素噪声更容易跨模型迁移
- **商业模型也不安全**：Gemini-2.5达42% ASR, GPT-4.1-nano达38.78%

## 亮点与洞察
- **loss景观视角的核心洞察**：将对抗可迁移性问题归结为loss景观平滑性问题，用语义损失替代CE来平滑景观。这个视角可迁移到所有对抗迁移性研究
- **变换不变性 = 模型不变性**：通过输入侧随机变换让扰动学到高级语义特征而非低级像素模式，高级特征跨模型共享——这解释了为什么人类设计的文字图像(FigStep)也有跨模型效果
- **挑战了"需要多代理"的信念**：此前认为跨模型迁移需要多个代理模型ensemble，UltraBreak证明单代理+正确的损失函数就够了

## 局限与展望
- **对高安全模型效果有限**：Claude-3-haiku只有16% ASR，说明强防御模型仍然有效
- **依赖白盒代理**：仍需一个开源VLM做白盒优化
- **防御方向**：论文揭示了VLM对"变换不变视觉特征"的脆弱性，可据此设计检测/防御——如检测图像中是否包含类文字对抗结构
- **结合GuardAlign**：UltraBreak的视觉攻击 vs GuardAlign的OT安全检测，两者是直接的矛盾对

## 相关工作与启发
- **vs FigStep**：FigStep人工设计每张图像（每目标一图），UltraBreak自动优化且单图通用，ASR更高
- **vs UMK/VAJM**：这些梯度方法用CE损失，严重过拟合白盒代理；UltraBreak的语义损失从根本上解决了这个问题
- **vs 文本侧越狱(GCG等)**：UltraBreak在视觉模态攻击，与文本攻击正交，可以组合使用

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ loss景观视角+语义损失+变换不变性的组合设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 6+开源模型+3商业模型×3基准×消融，非常全面
- 写作质量: ⭐⭐⭐⭐ 技术细节清晰，消融和可视化分析到位
- 价值: ⭐⭐⭐⭐⭐ 揭示了VLM安全的根本性脆弱性，对防御研究有重要指导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] JULI: Jailbreak Large Language Models by Self-Introspection](juli_jailbreak_large_language_models_by_self-introspection.md)
- [\[ACL 2025\] HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States](../../ACL2025/llm_alignment/hiddendetect_detecting_jailbreak_attacks_against_multimodal_large_language_model.md)
- [\[CVPR 2026\] Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models](../../CVPR2026/llm_alignment/principled_steering_via_null-space_projection_for_jailbreak_defense_in_vision-la.md)
- [\[ICLR 2026\] SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks](sema_simple_yet_effective_learning_for_multi-turn_jailbreak_attacks.md)
- [\[ICLR 2026\] JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks](jailnewsbench_multi-lingual_and_regional_benchmark_for_fake_news_generation_unde.md)

</div>

<!-- RELATED:END -->
