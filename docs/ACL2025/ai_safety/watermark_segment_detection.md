# Efficiently Identifying Watermarked Segments in Mixed-Source Texts

**会议**: ACL 2025  
**arXiv**: [2410.03600](https://arxiv.org/abs/2410.03600)  
**代码**: [https://github.com/XuandongZhao/llm-watermark-location](https://github.com/XuandongZhao/llm-watermark-location)  
**领域**: AI 安全 / 文本水印检测  
**关键词**: watermark detection, mixed-source text, geometric cover, online learning, watermark localization  

## 一句话总结
提出两种高效方法（Geometric Cover Detector 和 Adaptive Online Locator）用于在长文混合来源文本中检测和精确定位水印片段，时间复杂度从 O(n²) 降至 O(n log n)，在三种主流水印技术上均显著优于 baseline。

## 研究背景与动机
- LLM 生成文本的水印技术日益被用于检测合成文本，防止假新闻和学术不端等滥用
- 现有水印检测方法主要关注**整篇文档级别的分类**（是否含水印），但忽略了一个常见且重要的场景：在较长的**混合来源文档**中识别**哪些具体段落**是 LLM 生成的
- 实际应用场景：恶意行为者可能使用 LLM 修改新闻文章的某些段落来传播虚假信息，需要类似 Turnitin 抄袭检测系统的功能来定位这些片段
- 核心挑战：
  - 水印信号在长文中会被未加水印的部分"稀释"，整篇文档级检测方法失效
  - 暴力搜索所有可能的子序列区间需要 O(n²) 的时间复杂度，计算量过大
  - 单个 token 的水印分数噪声很大，直接阈值判断不可靠

## 方法详解

### 整体框架
论文提出两个互补的方法：
1. **Geometric Cover Detector (GCD)**：确定文档是否包含水印文本（分类任务）
2. **Adaptive Online Locator (AOL)**：精确定位水印文本的起止位置（定位任务）

两者均基于 Geometric Cover 技巧，时间复杂度为 O(n log n)。

### 关键设计

#### 三种目标水印方案
- **KGW-Watermark**：基于前缀 hash 将词表分为 green/red 两组，生成时提升 green token 的 logit
- **Unigram-Watermark**：固定 green/red 列表，不依赖前缀，鲁棒性更强
- **Gumbel-Watermark**：基于 Gumbel 技巧进行确定性采样，实现无失真水印

#### Geometric Cover Detector (GCD)
- 利用 Geometric Cover (GC) 技巧将长文本分割成多尺度的区间集合
- GC 定义：$\mathcal{I} = \bigcup_{k \in \mathbb{N} \cup 0} \mathcal{I}^{(k)}$，其中 $\mathcal{I}^{(k)}$ 是所有长度为 $2^k$ 的连续区间
- 每个 token 属于 $\lfloor \log n \rfloor + 1$ 个不同的区间，总共 O(n) 个区间
- 关键保证（Daniely et al., 2015 引理 5）：对于任意未知水印区间，GC 中总存在一个完全包含在其中且至少为其四分之一长度的区间
- 对每个区间独立进行水印检测，任一区间被检出则判定整篇文档含水印
- 通过 Bonferroni 校正控制 Family-Wise Error Rate (FWER)：段级 FPR τ → 文档级 FWER ≤ nτ
- 实际使用中从高阶区间开始（如 $\mathcal{I}^{(5)}$，32+ token 的区间），避免过短区间的不可靠判断

#### Adaptive Online Locator (AOL)
- 将水印定位问题转化为**在线序列去噪问题**（online denoising / nonparametric regression）
- 核心思想：每个 token 的水印检测分数是对期望分数的带噪观测，需要估计期望分数序列
  - Red-Green 水印：$s_t = \mathbf{1}(y_t \in \text{Green})$，期望值在水印段 > γ，非水印段 = γ
  - Gumbel 水印：$s_t = \log(1/(1-r_{y_t}))$，期望值在水印段 > 1，非水印段 = 1
- 采用 Aligator 算法（Baby et al., 2021）进行去噪
  - Aligator 内部也使用 Geometric Cover 结构
  - 提供最优的估计保证：在均方误差意义下与知道水印段位置的 oracle 竞争
  - 时间复杂度 O(n log n)
- **Circular Aligator**：为解决在线学习的边界效应
  - 将文本视为环形缓冲区，每次随机选择起始点遍历全序列
  - 进行 m 次不同起始点的迭代，最终对每个 token 取所有迭代预测的平均
  - 有效消除首尾边界处的预测不准确问题
- 最后对去噪后的平均分数应用阈值 ζ，分数超过阈值的 token 即判定为水印文本

### 损失函数 / 训练策略
本文是检测方法，不涉及模型训练。核心是：
- GCD 的阈值通过 FPR 校准函数 F 确定
- AOL 的阈值 ζ 根据水印方案设定（如 Gumbel 水印中 ζ=1.3）
- Circular Aligator 的迭代次数 m=10 个随机起始点

## 实验关键数据

### 实验设置
- 数据集：C4 和 Arxiv（真实文本作为非水印部分，LLM 生成的水印文本嵌入其中）
- 模型：LLaMA-7B 和 Mistral-7B
- 水印方案：KGW-Watermark, Unigram-Watermark, Gumbel-Watermark
- 评估指标：TPR（分类）, IoU（定位）

### 主实验 — 分类任务（C4 + LLaMA-7B）

| 方法 | KGW TPR | Unigram TPR | Gumbel TPR |
|------|---------|-------------|------------|
| Vanilla (整篇检测) | 0.602-0.692 | 0.006-0.058 | 0.650-0.918 |
| **GCD** | **0.912-0.934** | **0.874-0.958** | **1.000** |

- Vanilla 方法在 Unigram 水印上近乎完全失败（TPR 仅 0.006），因为水印信号被长文稀释
- GCD 在所有水印方案上都大幅优于 Vanilla 方法
- Gumbel 水印最容易检测，GCD 达到 100% TPR
- Unigram 水印的提升最为显著：从近乎随机（0.006）到 0.874+

### 主实验 — 定位任务
- AOL 在三种水印方案上的平均 IoU > 0.55，远超 baseline 方法
- Circular 初始化策略（m=10 个随机起始点）显著改善了边界处的定位精度
- 单次 Aligator 遍历存在明显的边界伪影，多次 circular 遍历有效消除

### 关键发现
1. **整篇文档级检测在混合来源场景下严重失效**：尤其对 Unigram 水印，所有信号被非水印文本淹没
2. **多尺度分析是关键**：GC 的多尺度区间划分确保了不同长度的水印片段都能被有效捕获
3. **在线去噪优于简单阈值**：单个 token 分数噪声大，必须通过适当的窗口平均来降噪
4. **Circular 策略对消除边界效应至关重要**：单次线性遍历预测的首尾位置不准确
5. **方法对水印方案的通用性强**：同一框架适用于三种完全不同机制的水印方案

## 亮点与洞察
1. **问题定义极具实用价值**：从"整篇文档是否有水印"进阶到"哪些段落有水印"，直接对接抄袭检测等真实需求
2. **算法设计优雅**：借用 Geometric Cover 和在线学习领域的成熟工具，将时间复杂度从 O(n²) 降到 O(n log n)
3. **Circular Aligator 设计巧妙**：简单但有效地解决了在线算法固有的边界效应问题
4. **理论保证扎实**：Aligator 的估计误差有明确的理论上界，且在段级别具有 strong adaptivity
5. **框架通用性强**：不依赖特定水印方案，可扩展到未来的新水印技术

## 局限性 / 可改进方向
1. **假设水印段是连续的**：实际中可能存在多个不连续的水印段，当前框架可能需要后处理来分割
2. **阈值 ζ 需要根据水印方案手动设定**：不同水印方案的最优阈值不同
3. **未考虑水印被攻击（如 paraphrase）后的鲁棒性**：实际场景中水印文本可能被改写
4. **仅在 7B 模型上评估**：未验证在更大模型（如 70B、GPT-4 级别）生成的水印文本上的效果
5. **FPR 控制使用保守的 union bound**：可能存在更紧的控制方法
6. **未讨论多段水印的定位**：当文档中有多个分散的水印段时，性能如何

## 相关工作与启发
- **水印方案**：KGW (Kirchenbauer et al., 2023), Unigram (Zhao et al., 2023), Gumbel (Aaronson, 2023, Kuditipudi et al., 2023)
- **部分水印检测**：Aaronson 的 plausibility score (O(n^{3/2})), Christ et al. 的 Substring Completeness (O(n²)), WinMax (O(n²))
- **在线学习**：Aligator (Baby et al., 2021) 的强自适应性保证
- **启发**：该框架可扩展到 (1) 代码水印的定位检测；(2) 与抄袭检测系统的集成；(3) AI 生成内容的细粒度标注；(4) 水印鲁棒性的对抗评估

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ — 首次在混合来源文本中实现高效水印段定位
- **技术深度**: ⭐⭐⭐⭐⭐ — 将 GC 和在线学习理论优雅结合，理论保证充分
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖三种水印方案和两个数据集，但模型规模有限
- **实用价值**: ⭐⭐⭐⭐⭐ — 直接对接抄袭检测和 AI 生成内容监管需求
- **写作质量**: ⭐⭐⭐⭐⭐ — 逻辑清晰，图示精美，理论与实验配合良好
- **综合评分**: 9.0/10
