---
title: >-
  [论文解读] Quantized Residuals to Continuous Prompts for Few-Shot Class Incremental Learning in Vision-Language Models
description: >-
  [CVPR 2026][自监督学习][FSCIL] QR-Prompt 把 CLIP 视觉特征与文本特征之间被对比学习抹平的"残差"离散量化成一组冻结的判别码本（DSQ），再通过层次提示编码器（HPE）和提示组合器（PC）把这些离散码翻译成类自适应的连续提示，从而在少样本类增量学习中同时拿住稳定性与可塑性，在 CUB200/CIFAR100/miniImageNet 上全面超过现有 SOTA。
tags:
  - "CVPR 2026"
  - "自监督学习"
  - "FSCIL"
  - "CLIP 残差"
  - "乘积量化"
  - "连续提示"
  - "灾难性遗忘"
---

# Quantized Residuals to Continuous Prompts for Few-Shot Class Incremental Learning in Vision-Language Models

**会议**: CVPR 2026  
**论文**: [CVF OpenAccess](https://openaccess.thecvf.com/content/CVPR2026/html/Sinha_Quantized_Residuals_to_Continuous_Prompts_for_Few-Shot_Class_Incremental_Learning_CVPR_2026_paper.html)  
**代码**: 无（论文未提供）  
**领域**: 自监督/持续学习 · 少样本类增量学习（FSCIL）· 视觉-语言模型  
**关键词**: FSCIL、CLIP 残差、乘积量化、连续提示、灾难性遗忘

## 一句话总结
QR-Prompt 把 CLIP 视觉特征与文本特征之间被对比学习抹平的"残差"离散量化成一组冻结的判别码本（DSQ），再通过层次提示编码器（HPE）和提示组合器（PC）把这些离散码翻译成类自适应的连续提示，从而在少样本类增量学习中同时拿住稳定性与可塑性，在 CUB200/CIFAR100/miniImageNet 上全面超过现有 SOTA。

## 研究背景与动机

**领域现状**：少样本类增量学习（FSCIL）要求模型在每个新会话里只用极少样本（典型 5-way 5-shot）学会新类，同时不能忘掉旧类。早期方法多基于纯视觉骨干（ResNet-18 等）做原型对齐或动态扩网；近年转向用 CLIP 这类视觉-语言模型（VLM）做提示学习（prompt learning），因为 VLM 的多模态对齐表示迁移性更强。

**现有痛点**：作者点出两层矛盾。其一是提示的"刚柔"难两全——完全可优化的提示（L2P、DualPrompt、CODA-Prompt）有可塑性但容易语义漂移（semantic drift），随会话推进精度崩塌；而静态或量化提示（VQ-Prompt）稳是稳了，却把细粒度区分能力压没了，细微的类间差异会塌缩进同一个量化码。其二更根本：CLIP 的对比预训练靠"特征去相关 + 均匀化"实现全局对齐，这会**抹平视觉属性之间的自然相关性**（颜色、纹理、形状这些可复用的结构线索），让特征流形变得更平更均匀。这些被抹掉的细粒度相关恰恰是少样本泛化的关键——线索没了，模型学每个新类几乎都要从零开始。

**核心矛盾**：可塑性 ↔ 稳定性 的 trade-off 在 FSCIL 里格外脆弱（小小的表示漂移就触发灾难性遗忘），而 VLM 的对比目标又主动压制了细粒度判别信息，二者叠加让"既稳又细"几乎不可能。

**切入角度**：作者做了一个关键观察——CLIP 的视觉嵌入 $x^v$ 与文本嵌入 $x^t$ 之间的**残差** $r = x^v - x^t$，正好保留了被对比学习压制的局部流形结构。论文用两个证据支撑：(a) 残差的跨类互相关明显高于已被去相关的视觉特征（图1a）；(b) 残差的幅度与视觉特征二阶 Hessian 项 $H_{f_v}(X)[\delta_i,\delta_i] \propto f_v(X+\delta_i)+f_v(X-\delta_i)-2f_v(X)$ 的幅度正相关（图1b），说明残差大的方向恰是流形曲率高、细粒度变化丰富的方向。一句话：**残差是一个"曲率感知"的互补信号**，藏着文本嵌入欠拟合掉的细节。

**核心 idea**：把残差做"离散量化 → 连续提示"的转换——先把残差空间量化成冻结的判别码本充当稳定锚点（治稳定性），再把离散码重新编码组合成类自适应的连续提示（治可塑性），用离散与连续的分工破解 FSCIL 的刚柔两难。

## 方法详解

### 整体框架
QR-Prompt 的输入是一张图像和它的类别文本模板，输出是一个注入到 CLIP 文本编码器里的、随类自适应的连续提示，最终得到与视觉特征对齐的判别文本特征用于分类。整条管线分三步流转：**先用冻结的 CLIP 视觉/文本编码器算出残差 $r = x^v - x^t$ → 经判别子空间量化（DSQ）把残差离散成每个子空间的最近码索引 → 层次提示编码器（HPE）把离散码查表并做跨子空间注意力变回连续细粒度特征 → 提示组合器（PC）把这些子空间特征聚合成单个提示并映射进文本空间**。

关键的稳定性来源在于会话分工：**基会话**里端到端训练 DSQ 码本（含旋转矩阵）以及 HPE+PC；进入**增量会话**后，DSQ 码本被**冻结**当作不变锚点，只微调 HPE 和 PC 这两个轻量模块去适配新类。而且因为提示主要依赖冻结码本生成，模型**不需要在每个会话后存储类级统计量**，天然省内存、抗遗忘。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["图像 + 类别模板"] --> B["冻结 CLIP 双编码器<br/>残差 r = xv − xt"]
    B --> C["判别子空间量化 DSQ<br/>M 子空间 + 旋转 R + Fisher 正则<br/>(基会话训完即冻结)"]
    C -->|每子空间最近码索引| D["层次提示编码器 HPE<br/>查表 + 跨子空间注意力"]
    D --> E["提示组合器 PC<br/>残差调制 query + 交叉注意力"]
    E -->|拼到 a photo of a {class}| F["冻结 CLIP 文本编码器<br/>→ 自适应文本特征"]
```

### 关键设计

**1. 判别子空间量化 DSQ：把残差离散成"曲率感知 + 类可分"的冻结码本**

DSQ 针对的痛点是：直接优化连续提示会漂移，而朴素量化又会把细粒度差异塌缩掉。它的解法是**量化残差而非视觉嵌入本身**——因为前面已论证残差保留了被对比预训练衰减掉的曲率细节。具体做法：在基会话把残差空间 $\mathbb{R}^D$ 划成 $M$ 个子空间，每个维度 $D/M$，每个第 $m$ 个子空间配一个码本 $C_m = \{c_{m,1},\dots,c_{m,K}\}$；残差 $r_i$ 的量化表示为各子空间最近码的拼接 $\hat r_i = \{c_{1,q_{i1}},\dots,c_{M,q_{iM}}\}$，其中 $q_{im} = \arg\min_k \|r_i^m - c_{m,k}\|_2$。这是标准的乘积量化（PQ）骨架。

DSQ 与传统 PQ/OPQ 的本质区别在于它不只追求"重构失真最小"，而是显式建模残差流形的几何：学一个正交变换 $R \in \mathbb{R}^{D\times D}$ 把子空间对齐到残差场的**主曲率方向**，让量化格子更贴合流形的局部线性邻域；同时引入 Fisher 风格正则保判别性——用类间散度 $S_b = \sum_y (\mu_y-\mu)(\mu_y-\mu)^\top$ 和类内散度 $S_w = \sum_y \sum_{i:y_i=y}(r_i-\mu_y)(r_i-\mu_y)^\top$，把训练目标写成

$$\mathcal{L} = \sum_i \|R^\top r_i - \hat r_i\|^2 - \lambda\, \mathrm{trace}\big((S_w + \epsilon I)^{-1} S_b\big),$$

第一项压重构误差、第二项（Fisher 判别项，$\lambda$ 控权重）拉大类间间隔。训练在"更新码本指派"和"精化 $R$"之间交替。最关键的一笔是：**码本基会话训完后全程冻结**。这既保证跨会话的量化子空间一致（缓解遗忘），又因为新会话样本极少、冻结码本能防止在小数据上学新码导致的过拟合。

**2. 层次提示编码器 HPE：把离散码翻译回带跨属性交互的连续细粒度特征**

DSQ 给出的是一堆离散码索引，痛点是它们既不连续、也只编码了孤立的子空间属性，无法直接喂给语言模型，更缺少属性之间的上下文依赖（光有"红""黑""条纹"，组不出"长着黑脸的红鸟"）。HPE 用两级把离散码升维成上下文感知的提示向量。第一级是**层次嵌入查表**：维护 $M$ 个独立嵌入表 $\{E_1,\dots,E_M\}$，每个 $E_m \in \mathbb{R}^{K\times D_p}$ 把一个码索引映成 $D_p$ 维连续向量，对残差 $\{c_{1,i},\dots,c_{M,i}\}$ 查出 $V_i = \{E_1(c_{1,i}),\dots,E_M(c_{M,i})\} \in \mathbb{R}^{M\times D_p}$；独立表让每个子空间学到自己专属的"语义词汇"。

第二级是**跨子空间注意力**：把一个 batch 的 $L$ 个样本嵌入堆成 $F_L = [V_1\,V_2\,\dots\,V_L] \in \mathbb{R}^{(LM)\times D_p}$，做多头自注意力 $A_L = \mathrm{Softmax}\!\big(Q_L K_L^\top / \sqrt{D_p}\big) V_L$，建模子空间之间的成对依赖，把分散的属性级线索合成连贯的细粒度提示特征。整体形成三级层次：DSQ 码本（低层粗粒度+曲率）→ 嵌入查表（中层连续化）→ 跨子空间注意力（高层细粒度语义）。论文用 t-SNE（图4a/b）佐证：注意力前子空间特征因低秩粗特征主导而聚成紧簇，注意力后簇会展开互混、暴露更丰富的语义结构。

**3. 提示组合器 PC：用残差调制的 query 把碎片特征聚成单个类自适应提示**

HPE 输出的是一组跨子空间嵌入，痛点是它们仍是分布式、部分冗余的碎片，没有机制把它们组合成一个类级的整体提示。PC 用可学习 query 注意力来做聚合，且把类自适应直接编进 query。它从一个基础 query $Q_0 \in \mathbb{R}^{1\times D_p}$ 出发（充当挑选显著语义的隐式槽位），用该类的均值残差 $\mu_y$ 调制：

$$Q_p = Q_0 \odot \big(1 + \tanh(W_c \mu_y)\big),$$

其中 $W_c$ 把残差投到 query 空间、$\odot$ 是逐元素乘——这一步把 query 偏向于含该类判别线索的子空间。再用调制后的 query 对 HPE 输出 $A_L$ 做交叉注意力 $p = \mathrm{Softmax}\!\big((Q_p W_q)(A_L W_k)^\top/\sqrt{D_p}\big)(A_L W_v)$ 得到单个聚合提示向量，最后线性映射进 CLIP 文本空间 $p^* = W_t\, p$（$W_t \in \mathbb{R}^{D\times D_p}$）。这个 $p^*$ 被拼到文本模板 $T = [\text{"a photo of a"}, \text{class}]$ 后送进冻结文本编码器得 $z_t = f_T([T, p^*])$。冻结的量化子空间保稳定、PC 的自适应组合保可塑性，分工正好对上 FSCIL 的两难。论文还观察到（图4c）：相似物种的子空间注意力分布相近，且共享某一视觉属性的不同物种会依赖一组相关的子空间（如子空间 7）。

### 损失函数 / 训练策略
DSQ 在基会话用上面的"重构 + Fisher 判别"联合目标交替优化 15 轮（$\lambda=0.1$）。HPE 和 PC 端到端联合训练，目标是 InfoNCE 对比损失，拉近生成文本特征与视觉特征：

$$\mathcal{L} = -\log \frac{\exp(f(z_t, x^v)/\tau)}{\sum_{i=1}^{N}\exp(f(z_t^i, x^v)/\tau)},$$

其中 $f(\cdot)$ 为余弦相似度、$\tau$ 为温度。基会话训 50 轮、每个增量会话微调 20 轮，增量阶段只更新 HPE/PC 这些轻量组件，DSQ 码本冻结。CUB200 用 $\tau=0.001$，CIFAR100/miniImageNet 用 $\tau=0.07$，batch size 16。

### 理论分析
论文给了两条界来支撑设计选择。**定理1（泛化界）** 是 PAC-Bayes 风格：若任务 $i$ 引入至多 $s_i$ 个新码，泛化松弛项 $\Delta$ 随每会话新引入码数**对数增长**（含 $\log(1 + s_j/(Z_j-1))$ 项）。因此令 $s_j=0$（即码本冻结）能得到更紧的界；加上基会话样本多、经验风险偏差小，这就解释了"为什么 DSQ 码本只在基会话优化、之后冻结"。**定理2（间隔保持界）** 量化对分类间隔的影响：类原型 $p_y = t_y + \beta\mu_y$ 量化后 $\hat p_y = t_y + \beta(\mu_y + q_y)$，则间隔损失期望 $\mathbb{E}[\gamma_{y,c} - \hat\gamma_{y,c}] \le \beta\sqrt{2\,\mathrm{tr}(P_U \Sigma_q P_U)}$（$U$ 为判别子空间、$P_U$ 为其投影）；并给出充分条件 $M\log_2 K \ge \frac{D}{2}\log_2(2\beta^2\sigma_r^2/\delta^2)$ 来把间隔损失压到 $\delta$ 以下。结论是增大 $M$ 或 $K$ 能收紧界，但收益取决于量化误差能否对齐到流形的局部线性区——而 DSQ 的正交旋转 $R$ 正是干这件事的。

## 实验关键数据

### 主实验
三个标准 FSCIL benchmark：CUB200（100 基类 + 10 个 10-way 5-shot 增量会话）、CIFAR100 与 miniImageNet（各 60 基类 + 8 个 5-way 5-shot 增量会话）。所有方法统一用 CLIP 预训练、ImageNet 微调的 ViT-B/16 骨干。指标为各会话 top-1 准确率、平均准确率 Avg（越高越好）、性能跌幅 PD（首末会话差，越低越好）。

| 数据集 | 指标 | QR-Prompt | 最强基线 | 说明 |
|--------|------|-----------|----------|------|
| CUB200 | Avg(%)↑ | **82.12** | 80.49 (BiMC) | 全场最高，最后会话仍 80.68% |
| CUB200 | PD(%)↓ | **6.17** | 7.00 (BiMC) | 跌幅最小，抗遗忘最稳 |
| CIFAR100 | Avg(%)↑ | 79.32 | 82.29 (VQPrompt) | Avg 非最高，但 VQPrompt PD 高达 20.21 |
| CIFAR100 | PD(%)↓ | **7.88** | 8.37 (BiMC) | 跌幅最小 |
| miniImageNet | Avg(%)↑ | **97.43** | 96.61 (VQPrompt) | 最高，PD 1.64 接近最优 |

关键对照：VQ-Prompt 在前几个会话冲得很高（CUB200 首会话 84.5、CIFAR100 首会话 94.11），但后期严重崩塌（CUB200 PD 26.87、CIFAR100 PD 20.21），正是"静态量化牺牲表达力 + 语义漂移"的典型；QR-Prompt 全程稳住，体现冻结判别码本 + 自适应提示组合的价值。CIFAR100 上 Avg 略逊于 VQPrompt，作者归因于低分辨率图像细粒度细节少、残差方差低，可挖的曲率信息有限。

### 消融实验
在 CUB200 上逐一移除 DSQ 旋转、HPE 注意力、PC（✓启用 / ✗关闭），并对比用视觉特征替代残差：

| 特征 | DSQ | HPE | PC | 末会话 Acc(%) | 说明 |
|------|-----|-----|----|----|------|
| 残差 | ✓ | ✓ | ✓ | **80.68** | 完整模型（首会话 86.49） |
| 残差 | ✗ | ✓ | ✓ | 78.46 | 去 DSQ 旋转，后期掉点明显 |
| 残差 | ✓ | ✗ | ✓ | 78.42 | 去 HPE，跨子空间泛化变弱 |
| 残差 | ✓ | ✓ | ✗ | 80.07 | 去 PC，多级提示线索聚合受损 |

完整模型各会话均最优，三模块互补。去 DSQ 旋转对后期会话伤害最大（说明对齐残差曲率、降量化失真很关键）；去 HPE 削弱跨子空间泛化；去 PC 让多级提示无法有效聚合。论文还报告用视觉特征代替残差会进一步掉点，再次印证残差是更有信息、更可迁移的适配基底。

### 关键发现
- **子空间数 $M$ 比码字数 $K$ 重要**：$M$ 从 8 升到 32 精度上升（子空间维度变小、类内方差降低，DSQ 能更精细建模局部残差结构），但超过 32 后下降（子空间太小抓不到语义关系，提示组合退化成噪声平均）；而固定 $M$ 改 $K$ 只有微弱增益——与定理2吻合：高维 CLIP 特征（$D=512$）下，$M$ 与旋转 $R$ 带来的结构分解比单纯加码字重要，子空间对齐好后增大 $K$ 收益递减。
- **$\lambda$ 呈凹形最优区 $[0.1, 0.2]$**：太小偏重构、子空间缺判别力；太大判别项主导、扭曲子空间几何使量化向量不再代表真实残差统计，违背 DSQ 假设。
- **定理2经验验证**：实测 $\mathbb{E}[\gamma_{y,c}-\hat\gamma_{y,c}]$ 始终落在推导上界之下（图6a/b），说明把 $U$ 子空间里的量化误差能量压小就能减小量化对精度的影响。
- **可解释性**：显著图（图5d）显示 Zero-Shot CLIP 常关注无关背景，QR-Prompt 则聚焦判别属性（红头、黑喙）；t-SNE（图5a）显示适配后文本/视觉原型对齐更好。

## 亮点与洞察
- **"残差即被压制的曲率信号"这个观察很漂亮**：把"对比学习抹平流形"这件抽象的事，落到 $r = x^v - x^t$ 这个可计算量上，还用 Hessian 二阶项的相关性做了实证。这是全文的发动机——把残差当一等公民来量化，而不是去量化视觉特征本身，是与 VQ-Prompt 等的根本分水岭。
- **"离散管稳、连续管变"的分工干净利落**：冻结的判别码本是稳定锚点（理论上让泛化界更紧），轻量 HPE/PC 提供可塑性，正好把 FSCIL 的刚柔两难拆成两个独立可调的部件。这个"冻结离散记忆 + 可微连续解码"的范式可迁移到其他持续学习/少样本场景。
- **不存类级统计量**：提示主要由冻结码本生成，省掉了很多原型类方法每会话存统计量的开销，工程上对长会话序列友好。
- **理论与超参直接挂钩**：定理2 的 $M\log_2 K \ge \frac{D}{2}\log_2(\cdot)$ 给了选 $M,K$ 的实际指导，而消融恰好验证"$M$ 比 $K$ 重要"，理论不是摆设。

## 局限与展望
- **作者承认的局限**：当前子空间记忆假设基类与新类之间存在**部分属性重叠**——码本是基类残差学出来并冻结的，若新类的属性分布和基类差异极大（出现全新属性），冻结码本可能覆盖不到。作者把"捕捉未见属性分布"列为未来工作。
- **CIFAR100 上不占优**：低分辨率图像残差方差低、可挖曲率有限，QR-Prompt 的 Avg 反被 VQPrompt 反超，说明该方法吃"细粒度高分辨率"数据（CUB200 鸟类、miniImageNet）更香，对粗粒度低分辨率任务收益打折。
- **依赖 CLIP 的多模态结构**：整套方法建立在视觉-文本残差之上，换成纯视觉骨干或残差信号弱的 VLM 时是否成立，论文未验证。
- **理论假设较强**：定理2 里量化误差零均值、协方差 $\Sigma_q \preceq \kappa I$ 等假设在真实码本上是否严格成立值得推敲（⚠️ 详细证明在补充材料，正文只给结论）；公式细节以原文为准。
- **改进思路**：可以让码本在增量会话里以极保守的方式"增量扩容"（而非完全冻结）来覆盖新属性，同时用定理1 的 $\log(1+s_j/Z_{j-1})$ 项约束新增码数，在稳定性与新属性覆盖间找平衡。

## 相关工作与启发
- **vs VQ-Prompt（NeurIPS24）**：两者都用向量量化造离散提示码本，但 VQ-Prompt 直接量化连续嵌入、只为提升稳定性和降重构误差，细微类间差异会塌缩进共享码导致前高后崩（PD 高达 20+）；QR-Prompt 量化的是残差、显式对齐流形曲率并加 Fisher 判别正则，再用 HPE/PC 解码回细粒度连续提示，稳定与表达兼得。
- **vs CODA-Prompt / DualPrompt / L2P（CVPR22-23）**：这些靠 key-query 匹配选提示或分解提示组件，可塑性有了但缺显式的跨任务判别分离机制，易语义漂移、后期掉点；QR-Prompt 用冻结判别码本充当不变锚点系统性压制漂移。
- **vs BiMC（CVPR25）/ FDR（ICCV25）**：BiMC 无需微调、靠校准视觉/文本原型纠正模态偏差，FDR 用文本派生掩码分解重组 CLIP 特征（但关键词抽取有歧义噪声）；QR-Prompt 不依赖文本关键词分解，而是从模态残差的几何结构里挖判别信息，对语义重叠的细粒度类更鲁棒，CUB200 上 Avg/PD 双双胜出。
- **vs OPQ / MoPQ / DPQ（乘积量化系）**：传统 PQ 家族优化子空间旋转只为降重构失真；DSQ 额外强调对齐视觉流形曲率以保类可分性与提示判别力，把"压缩"目标换成了"少样本判别 + 增量稳定"目标。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "残差=被压制曲率"的洞察 + 残差量化→连续提示的离散/连续分工，角度新且自洽
- 实验充分度: ⭐⭐⭐⭐ 三 benchmark + 完整模块消融 + $M/K/\lambda$ 敏感性 + 理论界经验验证，但缺代码与更多骨干/跨域验证
- 写作质量: ⭐⭐⭐⭐ 动机层层递进、图表佐证到位；理论符号偏密集，部分证明压在补充材料
- 价值: ⭐⭐⭐⭐ "冻结离散记忆 + 可微连续解码"范式对 FSCIL/持续学习有可迁移性，但对低分辨率/属性外推场景收益有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Semantic-Guided Global-Local Collaborative Prompt Learning for Few-Shot Class Incremental Learning](semantic-guided_global-local_collaborative_prompt_learning_for_few-shot_class_in.md)
- [\[CVPR 2026\] HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning](hycal_training_free_prototype_calibration_for_cross_discipline_fscil.md)
- [\[CVPR 2026\] Exemplar-Free Class Incremental Learning via Preserving Class-Discriminative Structure](exemplar-free_class_incremental_learning_via_preserving_class-discriminative_str.md)
- [\[CVPR 2026\] Representation-Steered Incremental Adapter-Tuning for Class-Incremental Learning with Pre-Trained Models](representation-steered_incremental_adapter-tuning_for_class-incremental_learning.md)
- [\[ICLR 2026\] PonderLM: Pretraining Language Models to Ponder in Continuous Space](../../ICLR2026/self_supervised/ponderlm_pretraining_language_models_to_ponder_in_continuous_space.md)

</div>

<!-- RELATED:END -->
