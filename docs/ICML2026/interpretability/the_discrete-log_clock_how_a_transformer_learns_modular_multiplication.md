---
title: >-
  [论文解读] The Discrete-Log Clock: How a Transformer Learns Modular Multiplication
description: >-
  [ICML 2026 (Mechanistic Interpretability Workshop)][可解释性][模乘法] 之前的工作发现 Transformer 学会模乘法 $a\cdot b\bmod p$ 后，嵌入的 Fourier 谱是"稠密"的（要用上所有频率），看起来比模加法复杂得多、不可解释；本文指出**这只是分析基选错了**——模乘法的天然 Fourier 基不是加性 DFT 而是**乘性特征变换**（即乘法群 $(\mathbb{Z}/p\mathbb{Z})^*$ 上的 Fourier 变换），换到这个基下谱立刻变稀疏（Gini 从 0.07 升到 0.58、只剩 4 个关键频率…
tags:
  - "ICML 2026 (Mechanistic Interpretability Workshop)"
  - "可解释性"
  - "模乘法"
  - "离散对数"
  - "乘性特征变换"
  - "Fourier 基"
  - "Clock 算法"
---

# The Discrete-Log Clock: How a Transformer Learns Modular Multiplication

**会议**: ICML 2026 (Mechanistic Interpretability Workshop)  
**arXiv**: [2606.17399](https://arxiv.org/abs/2606.17399)  
**代码**: https://github.com/danny-cpp/Discrete-Log-Clock (有)  
**领域**: 可解释性 / 机制可解释性 / grokking  
**关键词**: 模乘法, 离散对数, 乘性特征变换, Fourier 基, Clock 算法

## 一句话总结
之前的工作发现 Transformer 学会模乘法 $a\cdot b\bmod p$ 后，嵌入的 Fourier 谱是"稠密"的（要用上所有频率），看起来比模加法复杂得多、不可解释；本文指出**这只是分析基选错了**——模乘法的天然 Fourier 基不是加性 DFT 而是**乘性特征变换**（即乘法群 $(\mathbb{Z}/p\mathbb{Z})^*$ 上的 Fourier 变换），换到这个基下谱立刻变稀疏（Gini 从 0.07 升到 0.58、只剩 4 个关键频率，96.9% 的 MLP 神经元干净地调谐到单一频率），从而证明 Transformer 是**先用离散对数把乘法化成加法**，再套用和模加法一模一样的"Clock"三角恒等式机制，作者称之为"离散对数时钟"。

## 研究背景与动机

**领域现状**：grokking（网络在记忆训练集很久之后突然泛化）已成机制可解释性的标准试验台。对**模加法** $a+b\bmod p$，内部算法已被完全逆向工程：模型学出稀疏 Fourier 表示，把整数嵌成圆上的旋转，再用三角恒等式合成旋转——这就是 Nanda et al. (2023) 的"Clock"算法（Zhong et al. 2023 还发现了变体"Pizza"）。

**现有痛点**：换到**模乘法** $a\cdot b\bmod p$，模型同样能 grok，但内部算法一直没说清。Doshi et al. (2024) 推出的解析 MLP 解需要**稠密 Fourier 分量（所有频率）**；Furuta et al. (2024) 也实验确认 grokked Transformer 在所有频率上都有余弦偏置分量。这些结果让人以为：乘法用的是一种和加法**根本不同、且更不可解释**的算法。

**核心矛盾**：所谓"稠密谱"到底是算法本身复杂，还是分析工具用错了？加性 Fourier 变换是对"在整数标签 $a$ 上周期"的函数自然的基，但乘法的群结构根本不是加法群——拿加法的尺子量乘法，稠密很可能是基不匹配的**假象**而非算法复杂度。

**本文目标**：找到与任务代数结构匹配的分析基，看看模乘法的内部算法在对的基下是否其实简单、可解释。

**切入角度**：由于 113 是素数，非零元 $\{1,\dots,112\}$ 在模乘下构成**循环群**，存在原根 $g$ 使离散对数 $\log_g$ 成为群同构 $(\mathbb{Z}/p\mathbb{Z})^*\xrightarrow{\cong}\mathbb{Z}/(p-1)\mathbb{Z}$。在这个重标号坐标下，**乘法变成加法**：$a\cdot b\equiv 3^{(\alpha+\beta)\bmod 112}$。既然如此，学会群结构的模型，其嵌入应当在**重标号坐标**里周期，正如加法模型在原坐标里周期。

**核心 idea**：把分析基从加性 DFT 换成乘性特征变换（先按离散对数重排嵌入行、再做标准 DFT），稠密谱就会坍缩成稀疏谱，露出"离散对数时钟"算法。

## 方法详解

### 整体框架
本文不训练新机制，而是**逆向工程**一个已 grok 的 Transformer。流程是：在 $a\cdot b\bmod 113$ 上训练一个 1 层 decoder-only Transformer 直到 grok（约第 9K–14K epoch 测试损失骤降到近零）；然后把训练好的嵌入矩阵 $W_E\in\mathbb{R}^{112\times 128}$ 同时投到**加性基**和**乘性特征基**上，用稀疏度指标（Gini、IPR）、神经元单频调谐比例、log-序 SVD 三套证据，论证模型实际实现的是"离散对数时钟"——即先经离散对数把乘法化加法，再用和加法相同的三角恒等式电路完成计算。

模型实际实现的电路可拆成四步，环环相扣：

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入 (a, b)"] --> B["乘性特征嵌入<br/>把 a 嵌成 sin/cos(ω_k·log_g a)"]
    B --> C["注意力路由<br/>把 a,b 的频率分量送到 '=' 位"]
    C --> D["MLP 三角恒等式<br/>算出 cos/sin(k(α+β))"]
    D --> E["与去嵌入点积打分<br/>logit(c) ≈ Σ A_k cos(ω_k(α+β−γ))"]
    E -->|仅正确答案 γ*=(α+β) 全频建设性干涉| F["输出 c = a·b mod p"]
```

### 关键设计

**1. 乘性特征变换：给乘法换上对的 Fourier 尺子**

痛点是加性 DFT 量乘法必然看到稠密谱。乘性特征基用的是 $\sin(2\pi k\log_g(a)/q)$、$\cos(2\pi k\log_g(a)/q)$（$q=112$），操作上就是**先按离散对数重排嵌入行、再取标准 DFT**——这正是乘法群 $(\mathbb{Z}/p\mathbb{Z})^*$ 上的 Fourier 变换，把函数分解成乘性特征（群的不可约表示）。动机直接来自 $a\cdot b\equiv 3^{(\alpha+\beta)\bmod 112}$：乘法重标号后变加法，所以学会群结构的嵌入应当在重标号坐标 $\alpha=\log_g a$ 里周期。一个嵌入行 $W_E[a]$ 单看毫无结构，因为它是 128 个波在 $\alpha=\log_g(a)$ 处的点采样加噪声；只有按 $\alpha$ 重排后，每一维近似 $W_E[g^\alpha,j]\approx A_j\sin(\omega_{k_j}\alpha+\phi_j)$ 的正弦结构才显现。这是全文的方法核心——**匹配分析基与任务代数结构**。

**2. 稀疏度量化：用 Gini / IPR / 关键频率把"稠密到稀疏"做成硬数字**

光说"变稀疏"不够，需要可比指标。作者对每个频率 $k$ 算合并范数 $\|f_k\|=\sqrt{\|s_k\|^2+\|c_k\|^2}$（$s_k,c_k$ 是 sin/cos 投影，各 128 维），再用 Gini 系数

$$G=\frac{\sum_{i=1}^n(2i-n-1)|x_i|}{n\sum_{i=1}^n|x_i|}$$

度量集中度（$x_i$ 是排序后的频率范数，$G=0$ 均匀稠密、$G=1$ 极稀疏），并把超过中位数 $5\times$ 的频率记为"关键频率"。结果：加性基 $G=0.071$、关键频率 0 个；乘性基 $G=0.579$（8.1× 提升）、能量集中在 $\{2,8,47,56\}$ 四个峰。Inverse Participation Ratio 也从 52.7 降到 4.1。**同一个嵌入、只换基，谱从稠密变 4 峰**，是"稠密是假象"最直接的证据。

**3. 神经元单频调谐 + log-序重排：从单元层面坐实"乘法化加法"**

嵌入稀疏还可能是表面现象，作者进一步看 512 个 MLP 神经元。对每个神经元算它在所有输入对上的 2D 激活 $h_n(a,b)$（重排成 $112\times112$ 网格），在两种基下做 2D DFT、测单一频率占的最大能量比；超过 85% 算"单频调谐"。乘性基下 **496/512（96.9%）神经元单频调谐**，加性基下是 0 个。更直观的是：把神经元激活热图**按离散对数重排**后，出现清晰的对角条纹，说明神经元计算的是 $\alpha+\beta$ 的周期函数——这正是 $\cos(k(\alpha+\beta))$ 三角恒等式在 log 空间运转的签名，和 Nanda 在加法里发现的机制一模一样，只是发生在离散对数变换**之后**。再加 log-序 SVD：$W_E$ 主成分在整数序下全是噪声、按离散对数重排后露出正弦结构，有效秩仅 10.3/112，确认嵌入活在低维子空间、编码的是乘性特征。

### 一个例子：电路怎么算出 $a\cdot b$
给一对 $(a,b)$，记 $\alpha=\log_3 a,\beta=\log_3 b$。嵌入把 $a$ 编成各关键频率上的 $(\cos\omega_k\alpha,\sin\omega_k\alpha)$；注意力把 $a,b$ 的分量路由到"="位；MLP 用三角和角公式算出 $\cos_k(\alpha+\beta)=c_k(\alpha)c_k(\beta)-s_k(\alpha)s_k(\beta)$、$\sin_k(\alpha+\beta)=s_k(\alpha)c_k(\beta)+c_k(\alpha)s_k(\beta)$；最后对每个候选 $c$（$\gamma=\log_3 c$）与去嵌入列做点积打分，利用 $\cos A\cos B+\sin A\sin B=\cos(A-B)$，每个频率的贡献化为 $\text{logit}(c)\approx\sum_{k\in\mathcal K}A_k\cos(\omega_k(\alpha+\beta-\gamma))$。**只有正确答案 $\gamma^*=(\alpha+\beta)\bmod q$ 让所有频率同时取 $\cos 0=1$（建设性干涉）**，其余候选各频率相位错开互相抵消——这就是"时钟"在 log 空间对齐指针的过程。

### 损失函数 / 训练策略
1 层 decoder-only Transformer，$d_{\text{model}}=128$、4 个注意力头（$d_{\text{head}}=32$）、MLP 隐层 512 + ReLU、无 LayerNorm；输入 token 序列 $[a,b,=]$，从 "=" 位 logits 读预测。30% 数据训练（12,544 对中的 3,763 对），AdamW，学习率 $10^{-3}$，**权重衰减 1.0**，全批量梯度下降训 40,000 epoch。超参完全照搬 Nanda et al. (2023)，**唯一改动是任务（乘法替加法）**以保证可比。模型约第 500 epoch 记忆训练集，第 9K–14K epoch 突然 grok，最终测试准确率 100%。权重衰减惩罚记忆所需的大而无结构权重，逐步逼出能泛化的紧凑 Fourier 表示。

## 实验关键数据

### 主实验：两种基的稀疏度对比（$a\cdot b\bmod 113$）

| 指标 | 加性基 | 乘性基 |
|------|--------|--------|
| Gini 系数 | 0.071 | 0.579 |
| Inverse Participation Ratio | 52.7 | 4.1 |
| 检出关键频率数 | 0 | 4（$\{2,8,47,56\}$）|
| 神经元 >85% 单频 | 0/512 | 496/512 |
| 平均最大频率占比 | 0.054 | 0.925 |

### 跨素数 / 跨种子 / 推广

| 设置 | 结果 | 说明 |
|------|------|------|
| 10 个素数（$p=59$–$113$） | 均稀疏乘性谱，4–7 个关键频率 | 离散对数时钟普遍成立 |
| $p=113$ 多随机种子 | ~80% 种子出现该机制，Gini 0.45–0.61 | 机制基本通用，关键频率因种子而异 |
| log-序 SVD | 有效秩 10.3/112，露正弦结构 | 嵌入低维、编码乘性特征 |
| 模幂 $a^b\bmod 41$（2 层） | 100% 准确率 | 已知首个 Transformer 完美学会模幂；机制待研究 |

### 关键发现
- **同一模型换基即稀疏**：加性基 Gini 0.071 → 乘性基 0.579（8.1×），证明"稠密谱"是分析基假象而非算法复杂度。
- **神经元几乎全单频**：乘性基下 96.9% 神经元锁定单一频率、加性基下 0%，单元层面坐实模型在 log 空间做加法式三角运算。
- **log-序重排是钥匙**：嵌入热图、SVD 主成分在整数序下是噪声、按离散对数重排后才显出周期/正弦结构——离散对数是把"乱"变"齐"的非线性置换。
- **为什么前人会漏**：$\log_g(a)$ 是整数的非线性置换，在 $\log_g(a)$ 上周期的函数在 $a$ 上看就非周期，于是加性 DFT 必然报告"用上所有频率"，把基不匹配误读成算法复杂。

## 亮点与洞察
- **"对的基让噪声变结构"是可迁移的方法论**：核心洞察——分析基必须匹配任务的代数结构。对任意代数任务，用相关群上的 Fourier 变换就该露出可解释结构。这把一条具体发现升格成一般原则。
- **把一桩"反例"翻案**：前人用稠密谱论证乘法 ≠ 加法、更难解释，本文证明那是工具错位，乘法其实就是"log 空间里的加法 Clock"，优雅地统一了两类任务。
- **离散对数置换的认识论教训**：同一份数据/权重，换个排序方式，可解释性天差地别——提醒机制可解释性研究里"看不到结构"未必是没结构，可能只是没换对坐标。

## 局限与展望
- 作者承认：**未证明这是唯一可能算法**。加法上 Zhong et al. (2023) 已表明不同架构会收敛到不同解（Clock vs Pizza），乘法或许也有变体；跨种子也只有 ~80% 出现该机制。
- 规模有限：核心分析在单个素数 $p=113$、1 层 Transformer 上，虽补了 10 个素数与多种子，但都属小模型 grokking 玩具任务，对真实大模型的启示是间接的。
- 模幂 $a^b\bmod p$ 只给了"能 100% 学会"的存在性结果，**内部机制尚未逆向**，多层架构如何组合特征仍开放。
- 关键频率随种子变化（$\{2,8,47,56\}$ 只是某次训练），说明具体频率不重要、重要的是"稀疏 + 单频调谐"的结构——但这也意味着无法给出唯一闭式电路。

## 相关工作与启发
- **vs Nanda et al. (2023) Clock（加法）**: 他们逆向出加法的稀疏 Fourier + 三角恒等式 Clock；本文证明乘法是同一机制在离散对数空间的版本，方法论一脉相承。
- **vs Doshi et al. (2024)**: 他们用二次激活推出含 $\log_g$ 的解析 MLP 解，但没把它和训练好的 ReLU Transformer 的经验 Fourier 图景连起来；本文补上了这条经验链。
- **vs Furuta et al. (2024)**: 他们报告乘法谱"用所有频率"、与加法对比缺稀疏结构；本文直接反驳——稠密是基假象，换乘性基即稀疏。
- **vs Chughtai et al. (2023) / Stander et al. (2024)**: 他们研究网络学群运算的不可约表示、置换群（$S_5,S_6$）的陪集电路；本文聚焦把乘性特征基具体应用到训练好的 Transformer 上的 $(\mathbb{Z}/p\mathbb{Z})^*$。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "稠密是基假象、换乘性特征基即稀疏"一针见血地翻案，方法论可推广
- 实验充分度: ⭐⭐⭐⭐ 多指标 + 神经元 + SVD 三重证据，并跨 10 素数/多种子验证；但仍限于小模型玩具任务
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑链清晰、定义与公式齐全，附录给出完整电路推导
- 价值: ⭐⭐⭐⭐ 解决了模乘法机制可解释性的悬案，并提出"匹配分析基与代数结构"的通用原则

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Cortically Inspired Architecture for Modular Perceptual AI](../../ICLR2026/interpretability/a_cortically_inspired_architecture_for_modular_perceptual_ai.md)
- [\[ICML 2026\] How Language Models Process Negation](how_language_models_process_negation.md)
- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ICLR 2026\] Noise Stability of Transformer Models](../../ICLR2026/interpretability/noise_stability_of_transformer_models.md)
- [\[ICML 2026\] Query Circuits: Explaining How Language Models Answer User Prompts](query_circuits_explaining_how_language_models_answer_user_prompts.md)

</div>

<!-- RELATED:END -->
