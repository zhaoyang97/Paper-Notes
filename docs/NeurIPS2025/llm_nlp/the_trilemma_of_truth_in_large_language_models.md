# The Trilemma of Truth in Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2506.23921](https://arxiv.org/abs/2506.23921)  
**代码**: [GitHub](https://github.com/carlomarxdk/trilemma-of-truth)  
**领域**: LLM NLP / 可解释性 / 真实性探测  
**关键词**: veracity probing, multiple-instance learning, conformal prediction, truth direction, LLM internals

## 一句话总结
提出 sAwMIL（稀疏感知多实例学习）三类探测框架，结合 MIL 和保形预测，将 LLM 内部激活分类为 true/false/neither，揭示真假信号并非简单的双向对称编码，而是跨越多维子空间的分布式表征。

## 研究背景与动机
1. **领域现状**：现有 veracity probing 方法（如 mean-difference、CCS、TTPD）通过线性探针分离 LLM 内部激活中的"真"和"假"方向，基于最后一个 token 的表征做二分类
2. **现有痛点**：
   - 假设真假编码是双向对称的（$P(\phi|K_\mathcal{M}) = 1 - P(\neg\phi|K_\mathcal{M})$），但实际证据不支持
   - 假设 LLM 知道所有事实，忽略模型可能根本没有某些知识的情况
   - 仅用最后一个 token 的表征，丢失了句子中关键位置（如事实实现点）的信号
   - 输出分数未经校准，不能作为可靠的置信度估计
   - 只有二分类（true/false），无法处理"模型不知道"的情况
3. **核心矛盾**：二值逻辑无法准确描述 LLM 的内部知识状态——模型可能对某个陈述既不认为真也不认为假
4. **切入角度**：引入三值逻辑（true/false/neither），用 MIL 机制关注句子中最关键的 token，而非固定使用最后一个 token
5. **核心idea一句话**：用多实例学习让探针自动发现句子中承载真实性信号的关键位置，配合保形预测量化不确定性，实现三类分类

## 方法详解

### 整体框架
输入是一条陈述句经过 LLM 编码得到的中间层激活 $h_i(\boldsymbol{x}) \in \mathbb{R}^{L \times d}$（所有 token 的表征），输出是三类概率 $\{p_{true}, p_{false}, p_{neither}\}$。流程分三步：(1) 训练三个 one-vs-all 的 sbMIL 探针，(2) 通过 softmax 回归整合为多类概率，(3) 用保形预测校准输出。

### 关键设计

1. **稀疏感知多实例学习 (sAwMIL)**：
   - 做什么：将整个句子视为一个"bag"，每个 token 的激活是一个"instance"，自动识别哪些 token 承载了真实性信号
   - 核心思路：两阶段训练。第一阶段用 MIL-SVM 找到正例 bag 中得分最高的 instance，计算 $\eta$-分位数阈值。第二阶段只用阈值以上且属于"已实现部分"（actualized part）的 token 训练标准 SVM
   - 设计动机：句子中只有事实关键词（如"Latvia"在"The city of Riga is in Latvia"中）承载真实性信号，前缀部分（"The city of Riga is in"）不包含判定信息。MIL 自动定位这些关键 token，无需人工标注

2. **句内标签机制（Intra-bag Labels）**：
   - 做什么：将句子分为 pre-actualized 部分 $\boldsymbol{x}^p$（标签0）和 actualized 部分 $\boldsymbol{x}^a$（标签1）
   - 核心思路：$\eta$-分位数筛选后，再用 intra-bag 标签过滤，只保留 actualized 区域中得分高的 token
   - 设计动机：防止 MIL 错误地把非关键位置的噪声信号当作真实性信号

3. **One-vs-All → 多类整合**：
   - 训练三个独立的 sAwMIL 探针：is-true、is-false、is-neither
   - 通过 softmax 回归将三个探针输出整合：$p_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)}$，其中 $z_k = g_i^k(\boldsymbol{x}) \cdot \alpha_k + \beta_k$
   - 最终得到校准后的三类概率分布

4. **保形预测 (Conformal Prediction)**：
   - 做什么：将 SVM 原始距离分数转换为有统计保证的预测集
   - 核心思路：构造非一致性分数，确保预测集以 $1-\alpha$ 的概率覆盖真实标签
   - 设计动机：SVM 距离分数不是校准的概率，直接用 sigmoid 压缩也不可靠

## 实验关键数据

### 主实验：探针方法对比（16个LLM, 3个数据集平均）

| 方法 | 类型 | Correlation MCC | Generalization MCC |
|------|------|----------------|-------------------|
| Zero-shot prompting | 黑盒 | ~0.35 | ~0.25 |
| MD+CP (last token) | 二分类+CP | ~0.30 | ~0.15 |
| TTPD+CP (last token) | 二分类+CP | ~0.32 | ~0.18 |
| sPCA+CP (last token) | 二分类+CP | ~0.30 | ~0.17 |
| SVM+CP (last token) | 多分类 | ~0.55 | ~0.30 |
| **sAwMIL (full bag)** | **多分类+MIL** | **~0.60** | **~0.45** |

- 二分类探针在 full bag 设置下性能大幅下降，说明它们对噪声 token 敏感
- sAwMIL 在 full bag 下性能保持稳定甚至更好，说明 MIL 有效过滤了噪声

### 真假方向分析

| 指标 | SVM | sAwMIL |
|------|-----|--------|
| is-true vs is-false 余弦相似度 | ~-0.5 | ~-0.3 |
| is-true vs is-false Spearman相关 | ~-0.6 | ~-0.4 |
| 方向矩阵有效秩 | 1.73±0.012 | 1.93±0.004 |

- 如果真假是严格对称的，余弦相似度应接近 -1，有效秩应接近 1
- 实际有效秩接近 2，证明真假方向跨越二维子空间，不是单一轴上的对立

### 关键发现
- **二分类探针不可靠**：MD、TTPD 等方法在泛化评估中甚至不如 zero-shot prompting
- **neither 类至关重要**：二分类探针频繁将 neither 陈述错误分类为 true 或 false（高置信度误判）
- **真假编码不对称**：sAwMIL 的有效秩 1.93 接近 2，说明真和假在 LLM 内部确实是独立编码的
- **关键 token 位置很重要**：仅用最后 token 会丢失事实实现点的信号

## 亮点与洞察
- **三值逻辑的引入**：将 LLM 真实性探测从二分类扩展到三分类，"neither"类优雅地处理了模型无知的情况
- **MIL 自动定位关键 token**：不需要人工标注哪个 token 承载真实性信号，让数据自己说话，相比固定用最后 token 更鲁棒
- **五个假设的系统批判**：对现有文献的假设做了系统性审视，每个都给出了修正版本——这种研究范式可迁移到其他探测任务

## 局限性 / 可改进方向
- 只测试了二元关系陈述（城市-国家、药物-适应症），对多关系/多实体的复杂陈述未验证
- neither 类用合成实体生成（如 "Staakess" 这种不存在的城市），不确定是否真正反映了模型的"不知道"状态
- 模型规模限于 3-14B，更大模型（70B+）上的行为可能不同
- 探针仍是线性的，非线性探针可能捕获更复杂的真实性信号

## 相关工作与启发
- **vs Mean-Difference Probe (Marks et al.)**：只用最后 token + 二分类，无法处理 neither，泛化差；sAwMIL 用全句子 + 三分类，泛化好得多
- **vs CCS (Burns et al.)**：无监督对比学习找真假方向，但仍假设二值对称；sAwMIL 直接证明这个假设有问题
- **vs TTPD (Burger et al.)**：虽然也观察到多方向，但探针仍是二分类；sAwMIL 是第一个完整的三分类方案
- 对 LLM 安全和幻觉检测有启发：如果能实时检测模型内部是否有"neither"信号，可以在生成时主动拒绝回答

## 评分
- 新颖性: ⭐⭐⭐⭐ 三值逻辑+MIL 的组合很新颖，五个假设的批判有价值
- 实验充分度: ⭐⭐⭐⭐ 16个模型×3个数据集，多种基线对比，方向分析深入
- 写作质量: ⭐⭐⭐⭐⭐ 论述逻辑清晰，假设-批判-方案的结构很好
- 价值: ⭐⭐⭐⭐ 对理解LLM内部知识表征有重要贡献，但实际应用还需更多验证
