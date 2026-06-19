---
title: >-
  [论文解读] RNED: Rotary Number Encoding and Decoding for Medical VLMs
description: >-
  [CVPR 2026][多模态VLM][医学VLM] 针对医学 VLM 在「数值预测」上的硬伤，本文提出 RNED：编码端仿照 RoPE 用一个「值相关的旋转矩阵」把标量旋进一个专用 `[NUM]` token（保范数、保序、宽量程），解码端用 score-matching 从隐状态里查回连续数值，在放射学测量估计和医学视觉定位两个任务上稳定超过现有 VLM 基线。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "医学VLM"
  - "数值编码"
  - "旋转编码"
  - "score-matching"
  - "定量预测"
---

# RNED: Rotary Number Encoding and Decoding for Medical VLMs

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_RNED_Rotary_Number_Encoding_and_Decoding_for_Medical_VLMs_CVPR_2026_paper.html)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 医学VLM, 数值编码, 旋转编码, score-matching, 定量预测

## 一句话总结
针对医学 VLM 在「数值预测」上的硬伤，本文提出 RNED：编码端仿照 RoPE 用一个「值相关的旋转矩阵」把标量旋进一个专用 `[NUM]` token（保范数、保序、宽量程），解码端用 score-matching 从隐状态里查回连续数值，在放射学测量估计和医学视觉定位两个任务上稳定超过现有 VLM 基线。

## 研究背景与动机
**领域现状**：医学 VLM（LLaVA-Med、BiomedGPT、CT-CHAT 等）正被用来读 CT/MRI/超声并回答临床问题，但它们大多是为分类和文本生成优化的。

**现有痛点**：临床本质上是**定量**的——病灶尺寸、主动脉直径、射血分数、bounding-box 坐标，这些都是数字，且要可靠到能支撑诊疗决策。可底层 LLM 在数值上很弱：标准 tokenizer 会把 `42.5` 切成 `[4][.][5]`（Llama3）或 `[4][2][.][5]`（Mistral），模型根本没法把它当成一个数值实体；而 next-token 的交叉熵对「目标是 4，预测成 3 还是 9」给的是同样的 loss，完全不惩罚数值大小偏差，于是模型靠记忆和语料统计「编」出一个貌似合理的数，这在医学里是危险的。

**核心矛盾**：要把连续数值塞进一个为离散 token 设计、且带 LayerNorm/RMSNorm 的预训练 LLM，必须同时满足三件互相牵制的事——编码要**宽量程且保序**、解码要**抗噪**、整套要**对归一化不敏感**才能不破坏原模型。已有的 xVal 用一个 number head 做回归，但它会改变 `[NUM]` token 的范数，和 LayerNorm 冲突、拖累文本生成质量。

**本文目标**：先把「医学 VLM 的数值表示」该满足的性质形式化为 C1（宽量程保序编码）、C2（解码抗噪）、C3（归一化不变），再设计一套同时满足三者的编解码方案。

**切入角度**：作者注意到 RoPE 用「旋转」注入位置、且旋转**保范数**——既然旋转能在不改变向量长度的前提下编码一个连续量，那能不能用「值相关的旋转」来编码数字本身？

**核心 idea**：用「数值专属的旋转矩阵」把标量编码进单个 `[NUM]` token（RNE），再把「从隐状态恢复数值」转成一个对噪声鲁棒的 score-matching 查表问题（RND）。

## 方法详解

### 整体框架
RNED 嫁接在标准 VLM（如 LLaVA / Qwen2.5-VL）上：图像经 vision encoder + projector 变成视觉 token，文本里每个数字被替换成特殊 token `[NUM]`，并由 **RNE** 用该数值对应的旋转矩阵把 `[NUM]` 的基础 embedding 旋转一下，再和视觉 token、普通文本 token 拼在一起喂进 LLM。LLM 照常做自回归 next-token 预测；当语言模型头预测出 `[NUM]` 时，它对应的输出 embedding $\hat{x}_i$ 不走词表，而是交给 **RND** 的 score-matching 解码器在候选集里查回一个连续值 $\hat{m}$，这个值再被重新编码成下一步的输入 token，继续自回归直到 `[EOS]`。整条链路里只有 LLM（LoRA）和 projector 训练，vision encoder 冻结，RNE/RND 本身不引入额外可训练参数。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["CT 图像 + 含数字的文本问题"] --> B["Vision Encoder + Projector<br/>视觉 token"]
    A --> C["Rotary Number Encoding (RNE)<br/>把每个数旋进单个 [NUM] token"]
    B --> D["LLM（LoRA 微调）<br/>视觉/文本/数值 token 统一自回归"]
    C --> D
    D -->|预测到普通 token| E["文本输出"]
    D -->|预测到 [NUM]| F["Rotary Number Decoding (RND)<br/>score-matching 查回连续值 m̂"]
    F -->|m̂ 重新编码回输入| D
    F --> G["数值输出（如 41.5 mm）"]
```

### 关键设计

**1. Rotary Number Encoding (RNE)：用保范数的旋转把一个标量编成单 token**

要解决的是「数字被切成多个子词、丢掉数值身份」这个痛点。RNE 把每个数值 $m$ 表示成一个 token：取 `[NUM]` 的基础 embedding $x_{[NUM]}$，用一个**值相关的旋转矩阵** $R_d(m\Omega)$ 旋转它，即 $G_{enc}(x_{[NUM]}, m) = R_d(m\Omega)\,x_{[NUM]}$。先看二维直觉：用 $R_2(m\omega_0)=\begin{bmatrix}\cos(m\omega_0) & -\sin(m\omega_0)\\ \sin(m\omega_0) & \cos(m\omega_0)\end{bmatrix}$ 同时旋转 query 和 key 后，注意力点积会化简成 $q^{m_1}\cdot k^{m_2}=f_q(x_{[NUM]})^{T} R_2((m_2-m_1)\omega_0) f_k(x_{[NUM]})$——也就是说，两个数的**差值** $m_2-m_1$ 被自然注入到 attention score 里（用到了 $R_2(\theta)^T=R_2(-\theta)$、$R_2(\theta_1)R_2(\theta_2)=R_2(\theta_1+\theta_2)$）；对纯文本 token 令 $m=0$ 即不旋转，所以文本表示完全不被扰动。这直接满足 C1 的保序性，而旋转 $\lVert R_2(m\omega_0)x\rVert=\lVert x\rVert$ 保范数，是「通过方向而非幅度」编码数值，不会扰乱预训练模型里点积的尺度，天然满足 C3、即插即用。

单一频率 $\omega_0$ 的问题是**周期性**：角度绕一圈后不同的 $m$ 会撞成同一表示，无法覆盖宽量程。于是推广到高维：把 $d$ 维空间切成 $d/2$ 个二维子空间，每个子空间配一个**自己频率**的旋转块，组成块对角矩阵，频率取 $\Omega=\{\omega_j = B^{-2j/d}\mid j=0,\dots,\tfrac{d}{2}-1\}$，$B=5\times10^5$。虽然每个二维块仍周期，但整体旋转只在所有块的周期 $2\pi/\omega_j$ **同时对齐**时才重复，这对不同的 $m$ 极少发生——多频率构造因此在很宽的数值范围内给出大量唯一表示，真正满足 C1 的「宽量程」。

**2. Rotary Number Decoding (RND)：把「反解数值」改写成抗噪的 score-matching 查表**

编码是闭式的，但从 LLM 的**输出** embedding $\hat{x}_i$ 反解 $m$ 却很棘手。理论上可对每个二维块用 `atan2` 估出相对旋转角 $\theta_j$，得到一组线性同余式 $m\omega_j = \theta_j + 2\pi k_j$（$k_j\in\mathbb{Z}$ 是未知周期整数）；但 $\hat{x}_i$ 受上下文、近似误差和文本生成噪声污染，未必严格服从正弦结构，角度一偏，整组同余式就互相矛盾，纯解析反解在实践中不可行（C2 要的就是抗噪）。

RND 放弃「每个子空间都精确对齐」，改用一个**全局对齐分数**：输出 embedding 与未旋转基向量的点积。在理想无噪情形下展开 $x_m\cdot x_{[NUM]}=\sum_{j=1}^{d/2}\lVert x_{[NUM],j}\rVert^2\cos(m\omega_j)$——关键洞察是：这个全局点积是各旋转角余弦项的加权和，权重就是各二维子分量的平方范数，于是**数 $m$ 和一个可由 embedding 算出的标量之间存在直接关系**，给解码提供了一个鲁棒目标。为了让这个目标信号随 $m$ 平滑变化（不被高频项主导而剧烈振荡），引入按 $\omega_j$ 加权的广义分数

$$S(m, p) = \sum_{j=1}^{d/2}\left(\frac{1}{\omega_j}\right)^{p}\lVert x_{[NUM],j}\rVert^{2}\cos(m\omega_j).$$

解码就是在预定义候选集 $M$ 上做最近邻查表：$\hat{m}=G_{dec}(\hat{x}_i)=\arg\min_{m'\in M}\big(S(\hat{x}_i,p)-S(m',p)\big)^2$。所有候选的目标分数 $S(m',p)$ 可一次性预计算并缓存（约 20 秒 CPU、≈1.2 MB），推理时每个 token 只查一次，几乎零开销，于是候选集能轻松覆盖很宽的数值范围。RND 不引入任何额外可训练参数，这也是它比「学一个线性头」更优雅的地方。

**3. 可分性—平滑性权衡：用单一标量 $p$ 调一条连续谱**

公式中的 $p$ 控制了一个具体的 trade-off。$p=0$ 时退化为无权重和，高频分量贡献强，$S(m,0)$ 随 $m$ 剧烈振荡——相邻 $m$ 区分度高（distinguishability 好），但大数值处容易混叠。$p>0$（如 $p=1$）则放大低频、压制高频振荡，曲线更平滑（smoothness 好），稳定性和可用量程都更好，但相邻 $m$ 在分数空间里靠得更近、可分性下降。增大 $p$ 等于把 $S(m,p)$ 沿「高可分但振荡 ↔ 平滑但难分」这条谱滑动，本文用一个标量把这条谱参数化，实测 $p=0.2\sim0.3$ 折中最好；$p$ 太大（如 0.5）会把 $[0, 30000]$ 压进 $[0.7, 1.0]$ 的窄分数区间，小变化分不开，反而掉点。

### 损失函数 / 训练策略
训练目标是 $L = L_{CE} + \lambda L_{MSE}$。$L_{CE}$ 是覆盖所有位置的标准 next-token 交叉熵；$L_{MSE}$ **只在** target token 为 `[NUM]` 的位置施加，惩罚模型输出分数与预计算目标的平方差 $L_{MSE}=\sum_i \mathbb{I}(w_i=[NUM])\,(S(\hat{x}_i,p)-S(m_i,p))^2$，既鼓励把 `[NUM]` 放在正确位置，又让其 embedding 的分数对上数值目标；$\lambda$ 用线性 ramp-up 调度。Opport-CT 上两阶段训练：先冻结 vision encoder 和 LLM 只训 projector，再做全指令微调（projector + LoRA，rank 128、$\alpha$ 256）。

## 实验关键数据

### 主实验
放射学测量估计（Opport-CT，in-house），MAE 单值单位 mm、双值单位 mm²，越低越好；R² 越高越好：

| 方法 | 单值 MAE↓ | 单值 R²↑ | 双值 MAE↓ | 双值 R²↑ | 成功率% |
|------|-----------|----------|-----------|----------|---------|
| xVal | 6.71 | 0.351 | 684.20 | 0.217 | 75.9 |
| Learnable | 10.28 | 0.150 | 925.62 | −0.06 | 77.4 |
| Abacus | 5.65 | 0.319 | 520.32 | 0.215 | 57.2 |
| Standard Token | 5.53 | 0.338 | 519.23 | 0.194 | 55.7 |
| **RNED** | **4.72** | **0.568** | **449.23** | **0.320** | **81.8** |

公开 CT-RATE 上，把基线 CT-CHAT 用 RNED 目标微调 1 个 epoch：

| 方法 | 单值 MAE↓ | 单值 R²↑ | 双值 R²↑ | 成功率% |
|------|-----------|----------|----------|---------|
| CT-CHAT | 5.88 | 0.370 | 0.164 | 85.0 |
| CT-CHAT + RNED | 4.80 | 0.592 | 0.608 | 90.0 |

医学视觉定位（MedSeq-Bench，8 任务）：相比 SOTA 的 MedSeq-Grounder，RNED 平均 IoU +1.66%、Acc@5 +2.62%，在 Multi-view、Object Tracking 这类更难的任务上提升明显（IoU 55.03→59.26、62.10→63.15）。

### 消融实验
$p$ 的敏感性（Opport-CT）：

| $p$ | 单值 MAE↓ | 单值 R²↑ | 双值 R²↑ | 成功率% |
|-----|-----------|----------|----------|---------|
| 0（无权重） | 5.99 | 0.547 | 0.269 | 80.8 |
| 0.2 | 4.76 | **0.622** | 0.276 | 81.3 |
| 0.3 | **4.72** | 0.568 | **0.320** | 81.8 |
| 0.4 | 5.10 | 0.541 | 0.055 | 82.5 |
| 0.5 | 6.01 | 0.510 | 0.210 | 82.4 |

编码 / 解码策略对比：

| 编码 | 解码 | 单值 MAE↓ | 单值 R²↑ | 成功率% | 说明 |
|------|------|-----------|----------|---------|------|
| Adding | Linear | 24.86 | −2.025 | 33.1 | 直接加原值，梯度不稳、成功率崩 |
| Individual | Linear | 13.68 | −0.664 | 67.9 | 每个数学独立 embedding，不保序、泛化差 |
| RNE | Linear | 6.04 | 0.486 | 73.4 | 好表示已大幅改善 |
| **RNE** | **RND** | **4.72** | **0.568** | **81.8** | 完整方法最优 |

### 关键发现
- **表示比解码头更关键，但两者叠加才最好**：RNE+Linear 已远超 Adding/Individual，说明保序保范数的编码是大头；换成 RND 解码后再涨一截，且 RND 不引入额外可训练参数、支持宽量程。
- **标准 tokenization 会退化成「猜均值」**：基线 R² 很低、成功率仅 55.7%，倾向于输出接近数据集均值的数；RNED 把数当作单一概念单元、预测长度短，双值场景下优势更明显，成功率 >80%。
- **不伤文本能力**：在 BLEU-1/ROUGE-L/METEOR/GREEN 等常规 VQA 指标上 RNED 与标准 tokenization 持平甚至略好，说明数值能力是「加上去」的、没有牺牲语义生成。

## 亮点与洞察
- **把 RoPE 的「旋转编位置」迁移到「旋转编数值」**：保范数这一性质恰好绕开了 xVal「改 token 范数和 LayerNorm 打架」的坑，是这篇最漂亮的一步类比。
- **解码从「解同余方程」退到「查表匹配分数」**：承认隐状态有噪、不强求精确反解，用全局点积分数 $S(m,p)$ 做鲁棒目标，工程上还能预计算缓存，几乎零推理开销。
- **用一个标量 $p$ 把抽象的「可分性 vs 平滑性」变成可调旋钮**：把高频/低频贡献的权衡显式参数化，可迁移到任何「连续量嵌入离散序列模型」的场景。

## 局限与展望
- 数值范围依赖预定义候选集 $M$（放射任务 $[0,3000]$、定位 $[0,400]$）和步长，超范围或需要更细粒度时要重设。
- 主力结果在 in-house 的 Opport-CT 上，公开基准多为「在已有模型上微调 1 epoch」的增益验证，跨机构/跨模态的稳健性还需更多公开数据支撑。⚠️ Opport-CT 为非公开数据集。
- 视觉定位上 8 任务里 3 个没涨（如 Image Difference Grounding），作者归因于 SOTA 已接近饱和，RNED 在那类任务上更多是「保持」而非「提升」。

## 相关工作与启发
- **vs xVal**：同样想给 LLM 一个连续数值表示，xVal 用线性缩放 + number head，但会改变 `[NUM]` 范数、与归一化冲突；RNED 用保范数旋转 + 无参 score-matching 解码，兼容预训练权重且文本能力不掉。
- **vs Abacus / p10·p100**：这些方法（反转数字顺序、特殊字符串格式）主要面向「数字占主导」的数学 benchmark，会拉长预测长度，在混合文本-数字的临床问答里反而比基线还差；RNED 把数压成单 token，预测长度短。
- **vs Individual embedding（消融基线）**：给每个候选数学一个独立 embedding 虽然范数稳定、成功率尚可，但不保序、对未见数泛化差，印证了「保序 + 宽量程」才是关键。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把 RoPE 的保范数旋转迁移去编码连续数值，解码改写成 score-matching 查表，角度清新且自洽。
- 实验充分度: ⭐⭐⭐⭐ 两任务多基准 + $p$ 与编解码策略消融齐全；公开侧多为微调增益验证、且依赖一个 in-house 数据集。
- 写作质量: ⭐⭐⭐⭐⭐ C1–C3 性质先行、从 2D 直觉推到高维、再到解码权衡，推导链条清晰。
- 价值: ⭐⭐⭐⭐ 直击医学 VLM 的定量短板，即插即用、不伤文本能力，对所有「要从 LLM 读出可靠数字」的场景都有借鉴。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Sparse Spectral LoRA: Routed Experts for Medical VLMs](sparse_spectral_lora_routed_experts_for_medical_vlms.md)
- [\[CVPR 2026\] Reliable Clustering Number Estimation for Contrastive Multi-View Clustering](reliable_clustering_number_estimation_for_contrastive_multi-view_clustering.md)
- [\[CVPR 2026\] Medic-AD: Towards Medical Vision-Language Model's Clinical Intelligence](medic-ad_towards_medical_vision-language_models_clinical_intelligence.md)
- [\[CVPR 2026\] Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning](similarity-as-evidence_calibrating_overconfident_vlms_for_interpretable_and_labe.md)
- [\[CVPR 2026\] ORION: ORthonormal Text Encoding for Universal VLM Adaptation](orion_orthonormal_text_encoding_for_universal_vlm_adaptation.md)

</div>

<!-- RELATED:END -->
