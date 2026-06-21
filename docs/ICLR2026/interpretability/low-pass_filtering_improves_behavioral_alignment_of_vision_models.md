---
title: >-
  [论文解读] Low-Pass Filtering Improves Behavioral Alignment of Vision Models
description: >-
  [ICLR 2026][可解释性][行为对齐] 作者发现，此前被归功于"生成式目标"的 Imagen 类模型的高度人类化视觉行为，其实主要来自一个不起眼的降采样操作（等效低通滤波）；只要在**测试时**对输入图像做高斯模糊，普通判别式 CLIP 就能在 model-vs-human 基准上刷新 SOTA，把人机行为对齐的差距砍掉一半。
tags:
  - "ICLR 2026"
  - "可解释性"
  - "行为对齐"
  - "低通滤波"
  - "错误一致性"
  - "形状偏好"
  - "对比敏感度函数(CSF)"
  - "model-vs-human"
---

# Low-Pass Filtering Improves Behavioral Alignment of Vision Models

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=YhgBy6jTR8](https://openreview.net/forum?id=YhgBy6jTR8)  
**代码**: 基于 [model-vs-human](https://github.com/bethgelab/model-vs-human) 与 OpenCLIP  
**领域**: 视觉模型可解释性 / 人机行为对齐 / 认知科学  
**关键词**: 行为对齐, 低通滤波, 错误一致性, 形状偏好, 对比敏感度函数(CSF), model-vs-human  

## 一句话总结
作者发现，此前被归功于"生成式目标"的 Imagen 类模型的高度人类化视觉行为，其实主要来自一个不起眼的降采样操作（等效低通滤波）；只要在**测试时**对输入图像做高斯模糊，普通判别式 CLIP 就能在 model-vs-human 基准上刷新 SOTA，把人机行为对齐的差距砍掉一半。

## 研究背景与动机
**领域现状**：深度神经网络（DNN）虽在视觉基准上表现强劲，但作为"人类视觉系统的计算模型"仍不合格，典型差距体现在两点——缺乏**形状偏好**（人看冲突线索图按形状分类，DNN 按纹理分类）和**错误一致性**低（DNN 觉得难的图和人觉得难的图不是同一批）。社区用 model-vs-human（MvH）基准来量化这种行为对齐。

**现有痛点**：Jaini et al. (2023) 报告生成式模型 Imagen 达到了迄今最像人的行为（错误一致性 0.31、形状偏好 0.99），并据此推测是**生成式目标本身**驱动了这种人类化倾向。这一解释被卷入一个更大的争论：人类视觉到底是判别式自下而上（Helmholtz 的逆向推理），还是带 top-down 先验的生成式系统（预测编码）？如果 Imagen 像人是因为生成式目标，那似乎暗示人脑皮层也依赖生成原理。

**核心矛盾**：但 Imagen 有一个被忽视的处理步骤——它把输入降采样到 $64\times64$ 而非 $224\times224$，**这等效于一个低通滤波器**。而众所周知，人眼光学系统与早期神经处理本身就是低通/带通滤波器，尤其在 MvH 采用的 200ms 短呈现时间下更接近低通。于是"是生成式目标还是单纯低通滤波"成了一个可证伪的对立假设。

**本文目标**：检验"测试时低通滤波"对行为对齐的影响（区别于以往"训练在模糊图上"的工作），并据此给出 Imagen 对齐性的替代解释。**核心 idea**：行为对齐的本质不是模型架构或训练目标，而是**让模型看到的频谱与人类视觉皮层接收到的频谱相匹配**——而这只需在推理时加一个低通滤波器即可。

## 方法详解

### 整体框架
方法极简：在冻结的判别式视觉模型（OpenCLIP ViT-H-14，零样本）前面"前置"一个低通滤波器，再在 MvH 基准上测错误一致性（EC）、形状偏好（SB）、OOD 准确率。围绕这个核心操作，作者做了四层递进论证：① 用高斯模糊/降采样两种方式实现低通，扫描强度看对齐变化；② 直接在傅里叶空间**学习**一个最大化 EC 的最优滤波器，验证它是否真长成低通形状；③ 把最优高斯与人类对比敏感度函数（CSF）对齐，给出物理解释；④ 计算 MvH 基准所有 pareto 最优解的"天花板曲线"，揭示准确率—一致性的内在权衡。

```mermaid
flowchart LR
    A[输入图像 224x224] --> B[前置低通滤波器<br/>高斯模糊σ / 降采样到R1再放大]
    B --> C[冻结判别式模型<br/>OpenCLIP ViT-H-14 零样本]
    C --> D[MvH 基准评测]
    D --> E1[错误一致性 EC↑]
    D --> E2[形状偏好 SB↑]
    D --> E3[OOD 准确率 略降]
    F[傅里叶空间学习最优滤波器] -.验证.-> B
    G[人类CSF@200ms] -.物理解释.-> B
```

### 关键设计

**1. 测试时低通滤波：用一行预处理复刻 Imagen 的"人类化"**
作者用两种等效方式实现低通——torchvision 的 `GaussianBlur`（强度 $\sigma$），以及模仿 Imagen 的 `Resize`（先把 $R_0\times R_0$ 降到 $R_1\times R_1$ 再 bicubic 放回，resize 强度即比值 $R_0/R_1$）。关键转折在于**只在测试时滤波、完全不重训**：随着模糊强度增大，高频纹理信息被抹掉，模型被迫依赖低频形状线索，于是形状偏好单调上升、错误一致性从 $\kappa=0.28$ 升到 $\sigma=2.5$ 时的 $\kappa=0.37$，一举超过 Imagen 的 0.31。这直接证明 Imagen 的高对齐性**无需生成式组件即可复现**——一个标准 ResNet-101 前置 $\sigma=3.0$ 的低通滤波就能拿到 0.8 形状偏好，而 Jaini 等人为此重训了带扩散噪声增强的 ResNet-50 才得到 0.78。

**2. 傅里叶空间学最优滤波器：验证"最优解确实是低通"**
仅证明"加低通有用"还不够，作者进一步问"**最优**滤波器长什么样"。他们冻结 CLIP，在傅里叶域学一个滤波器 $G_\theta$ 直接最大化与人类的错误一致性。为尊重 Hermitian 对称（只调幅度不调相位），把 $224\times224$ 的 DFT 矩阵参数化为一个 $112\times112$ 实矩阵作为左上象限，其余象限取复共轭得到。单张图的相似度与损失为
$$s_i = \mathrm{sim}\big(f_I(\mathcal{F}^{-1}(\mathcal{F}(x)*b_\gamma(G_\theta))),\, f_T(y_i)\big),\quad L(x)=H\big(\sigma(s/\tau),\hat{y}\big)+\lambda\|\theta\|_1.$$
由于只有不到 1.2 万张 MvH 图却要学 12544 个参数，极易过拟合成"对抗噪声"，作者用 L1 稀疏正则（$\lambda=5\times10^{-5}$）加上对滤波器本身做模糊（$\gamma=6.0$）强制平滑，并引入可学温度 $\tau$ 放大 CLIP 过近的相似度信号。学出来的滤波器频谱明显集中在低频——**最优解确实是一个低通滤波器**，反向坐实了核心假设。

**3. 用 CSF 给出物理解释：低通有效因为它"模拟人眼"**
为什么低通正好有用？作者把最优高斯的频谱与人类**对比敏感度函数（CSF）**对齐。CSF 描述人对正弦光栅的对比阈值倒数随空间/时间频率的变化；在 200ms 短呈现下，CSF 从带通退化为近似低通。用按自然图像功率谱 $P(f)=f^{-\beta}$（$\beta\approx2$）加权的 WRMSE 度量拟合优度：
$$L_{\mathrm{WRMSE}}=\sqrt{\frac{\int_{f_{\min}}^{f_{\max}} f^{-\beta}(\mathrm{CSF}(f)-G(f))^2\,df}{\int_{f_{\min}}^{f_{\max}} f^{-\beta}\,df}}.$$
结果：拟合最好的高斯 $\sigma\approx2.5\text{px}$，恰好与经验上最大化 EC 的 $\sigma$ 重合；而**直接拿 CSF 当滤波器**用，EC 达 0.365，与最优高斯统计上不可区分。这说明低通之所以奏效，是因为它把模型输入处理成了人类视觉皮层实际接收到的样子（被眼球光学衍射、被视网膜/LGN 前几级神经变换过滤）。

**4. 准确率—一致性权衡与 pareto 前沿：标定基准天花板**
作者还揭示 MvH 内部存在结构性矛盾：OOD 准确率与错误一致性不能同时拉满。要与人类 EC 最大（0.674），必须把 OOD 准确率压到 0.67；当 OOD 准确率趋近 100%，EC 趋近 0。中间是一条光滑的 pareto 最优曲线（前沿），由于这些值是最优且固定的、无不确定性。这首次给出了 MvH 的真实天花板——证明基准**远未饱和**。其哲学含义是：一台达到超人准确率的机器必然用到人类不用的特征，从而行为上必然偏离人类；通过低通滤波主动给模型"设限"反而能换来对齐。

## 实验关键数据

### 主实验表格（MvH 基准，OpenCLIP ViT-H-14 为参考模型）

| 模型 | 错误一致性 ↑ | 形状偏好 ↑ | OOD 准确率 ↑ |
|------|------------|-----------|------------|
| 人类（平均） | 0.43 | 0.96 | 0.72 |
| ViT-22B-384 | 0.26 | 0.87 | 0.80 |
| OpenCLIP ViT-H-14（基线） | 0.28 | 0.60 | 0.78 |
| Imagen（前 SOTA，生成式） | 0.31 | 0.99 | 0.71 |
| ViT-H-14 + Resize(64×64) [本文] | 0.35 | 0.91 | 0.75 |
| ViT-H-14 + Blur(σ=2.5) [本文] | **0.37** | **0.96** | 0.72 |
| ViT-H-14 + 傅里叶学习滤波 [本文] | **0.38** | 0.95 | 0.73 |

### 消融 / 分析实验

| 维度 | 设置 | 关键结果 |
|------|------|---------|
| 模糊强度扫描 | σ 从 0 增大 | SB 单调↑；EC 升到"临界点"后回落 |
| 实现方式 | 高斯模糊 vs 降采样 | 两者趋势一致，模糊效果更强（EC 0.37 vs 0.35）|
| OOD 准确率代价 | σ=2.5 / resize 64 | 仅降 6pp / 3pp，非灾难性 |
| 跨模型泛化 | ResNet/BiT-M/ViT/Noisy Student/多个 OpenCLIP | 几乎所有模型 SB、EC 均随低通提升，小 patch(14) 的 CLIP 增益最大 |
| 直接用 CSF 当滤波器 | — | EC=0.365，超过 Imagen 且与最优高斯不可区分 |
| 无需训练 | ResNet-101 + σ=3.0 低通 | SB=0.8，超过重训 ResNet-50（0.78）|

### 关键发现
- 测试时模糊（σ=2.5）让 EC 从 0.28→0.37，**超过 Imagen 0.31**，把人机—人人对齐差距砍半。
- 形状偏好随低通**单调上升**，错误一致性存在**临界点**（过模糊反而下降）。
- 学出来的最优滤波器、最优高斯、人类 CSF@200ms 三者频谱高度一致，σ 都落在 ~2.5px。
- MvH 最大可能 EC 为 0.674（对应 OOD acc 0.67），基准远未饱和。

## 亮点与洞察
- **一个降采样揭穿一个宏大叙事**：把"生成式 vs 判别式视觉"的哲学争论，还原成"低通滤波"这个工程细节，极具说服力且方法廉价（测试时一行预处理，零训练）。
- **三重交叉验证**：经验最优 σ、梯度学习的最优滤波器、人类 CSF 拟合最优 σ，三条独立路径都指向 ~2.5px 低通，论证链非常扎实。
- **首次标定 MvH 天花板**：计算 pareto 前沿，把"基准是否饱和"从猜测变成确定结论，对后续行为对齐研究有方法论价值。
- **反直觉的对齐观**：提出对齐的代价是放弃部分准确率——超人准确率必然意味着用了非人类特征，从而行为偏离人类。

## 局限与展望
- 低通滤波会牺牲 OOD 准确率（虽非灾难性），如何在不掉准确率前提下提升对齐仍是开放问题（pareto 前沿上仍有大量改进空间）。
- 学习傅里叶滤波器允许旋转不对称，导致出现疑似过拟合的伪影，最优滤波器形态的"上界"仍带噪声。
- 结论主要在 MvH 这一基准上得到，而 MvH 本身有已知缺陷（形状偏好可被平凡刷高、EC 噪声大）。
- CSF 解释建立在 200ms 短呈现 + 后向掩蔽的特定实验范式上，是否推广到长呈现、自然观看条件仍待验证。

## 相关工作与启发
- **直接对手**：Jaini et al. (2023) 将 Imagen 的对齐归因于生成式目标，本文给出低通滤波的替代解释并实证反驳。
- **训练时 vs 测试时模糊**：以往工作（Jinsi 2023；Jang & Tong 2024；Lu 2025）训练在低通图上，EC 多停留在 0.2 以下；本文证明**测试时**滤波效果更强。
- **视觉系统频率调谐**：承接 Campbell & Robson (1968) 的通道理论、Kelly (1979) 的 CSF，以及 DNN 中类人 CSF 的发现（Li 2022；Akbarinia 2023），补上了"短呈现需要额外低通适配"这一环。
- **启发**：把"人脑施加的物理约束"作为正则/先验注入模型，可能是行为对齐比堆数据/换架构更本质的路径；也提示评测人类化时要先排除掉预处理这类混淆变量。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 用一个被忽视的降采样操作推翻"生成式目标驱动对齐"的主流解释，洞察犀利且反直觉。
- 实验充分度: ⭐⭐⭐⭐ — 跨模型、跨实现方式、学习最优滤波器、CSF 拟合、pareto 前沿多角度互证；略受限于单一 MvH 基准。
- 写作质量: ⭐⭐⭐⭐⭐ — 假设—检验—解释的逻辑链清晰，物理动机与机器学习证据衔接自然。
- 价值: ⭐⭐⭐⭐⭐ — 既刷新 SOTA 又澄清了视觉科学的一个核心争论，并为基准标定天花板，对认知科学与可解释性社区都有分量。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Sequences of Logits Reveal the Low Rank Structure of Language Models](sequences_of_logits_reveal_the_low_rank_structure_of_language_models.md)
- [\[ICLR 2026\] Inducing Dyslexia in Vision Language Models](inducing_dyslexia_in_vision_language_models.md)
- [\[ICLR 2026\] MICLIP: Learning to Interpret Representation in Vision Models](miclip_learning_to_interpret_representation_in_vision_models.md)
- [\[ICLR 2026\] Towards Cognitively-Faithful Decision-Making Models to Improve AI Alignment](towards_cognitively-faithful_decision-making_models_to_improve_ai_alignment.md)
- [\[ICLR 2026\] Linear Mechanisms for Spatiotemporal Reasoning in Vision Language Models](linear_mechanisms_for_spatiotemporal_reasoning_in_vision_language_models.md)

</div>

<!-- RELATED:END -->
