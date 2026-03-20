# Addressing Divergent Representations from Causal Interventions on Neural Networks

**会议**: ICLR 2026 Oral / **arXiv**: [2511.04638](https://arxiv.org/abs/2511.04638) / **代码**: [GitHub](https://github.com/grantsrb/rep_divergence) / **领域**: 机械可解释性 / **关键词**: causal intervention, mechanistic interpretability, representational divergence, Counterfactual Latent loss, DAS

## 一句话总结

系统性地揭示因果干预（activation patching、DAS、SAE 等）会将模型内部表征推离自然分布，理论区分"无害偏移"与"有害偏移"两类情况，并提出 Counterfactual Latent (CL) loss 来约束干预表征不偏离流形，在 7B LLM 上验证可减少偏移同时保持干预准确率。

## 研究背景与动机

**领域现状**：机械可解释性的核心方法论是因果干预——通过 activation patching、DAS、SAE 等方式操纵模型内部表征，观察行为变化，从而推断表征编码了什么。即使是 SAE、PCA 等相关性方法，也通常以因果干预作为验证特征是否真正有意义的最终裁判。因果干预在功能性机制声明中占据核心地位。

**关键痛点**：这些因果干预方法隐含一个未经检验的假设——干预产生的反事实模型状态对目标模型来说是"现实的"。例如，有些 activation patching 实验会将特征值放大 15 倍，这种情况下干预后的表征很可能已经严重偏离了模型的自然分布。

**核心矛盾**：如果干预后的表征是 out-of-distribution 的，那么后续层对这些 OOD 输入的响应可能激活训练中从未见过的"隐藏通路"(hidden pathways)，导致观察到的因果效应实际上是虚假的——我们以为发现了模型的自然机制，实则是干预制造的伪影。

**切入角度**：作者从理论和实验两个维度同时出发：(1) 先证明偏移是普遍现象；(2) 再区分偏移何时无害、何时有害；(3) 最后提出缓解方案。这是对整个可解释性方法论的元层面审视。

**核心 idea**：不是所有偏移都有害——行为零空间内的偏移是无害的，但激活隐藏通路或触发休眠行为变化的偏移是有害的。通过 CL loss 约束干预表征贴近自然流形，可以系统性地缓解有害偏移。

## 方法详解

### 整体框架

本文的逻辑链条分四步：

1. **证明偏移普遍存在**（Section 3）：理论证明 + 三种主流干预方法的实证
2. **区分无害 vs 有害偏移**（Section 4）：行为零空间理论 + 隐藏通路 + 休眠行为变化
3. **提出 CL loss 缓解方案**（Section 5.1）：应用于 Boundless DAS + 7B LLM
4. **改进 CL loss 用于 OOD 泛化**（Section 5.2）：修改版 CL loss 仅约束因果子空间

### 关键设计 1：偏移的理论保证

对于坐标级 patching，作者证明只要流形不是轴对齐的超矩形，偏移就必然发生。考虑圆形流形 $\mathcal{M}_K = \{c_K + u : \|u\|_2 \leq r_K\}$，将 $h^{\text{src}}$ 的第一个坐标和 $h^{\text{trg}}$ 的第二个坐标拼接：

$$\hat{h} = \begin{bmatrix} h_1^{\text{src}} \\ h_2^{\text{trg}} \end{bmatrix}, \quad \|\hat{h} - c_K\|_2^2 = u_1^2 + v_2^2$$

取边界点 $u = (r_K, 0)$，$v = (0, r_K)$ 可得 $\|\hat{h} - c_K\| = r_K\sqrt{2} > r_K$，干预后表征超出流形边界。

**定理 A.2** 进一步证明：一个非空凸集是 patch-closed 的当且仅当它是各坐标投影的笛卡尔积（即轴对齐超矩形）。因此球、椭球、一般多面体等常见流形几何在坐标 patching 下都会产生偏移。这是一个很强的负面结论。

### 关键设计 2：行为零空间与无害偏移

定义函数 $\psi: \mathbb{R}^d \to \mathbb{R}^{d'}$ 关于集合 $X$ 的行为零空间：

$$\mathcal{N}(\psi, X) = \{v \in \mathbb{R}^d \mid \forall x \in X,\ \psi(x+v) = \psi(x)\}$$

如果偏移 $v \in \mathcal{N}(\psi, X)$，即 $\psi(x+v) = \psi(x)$，则该偏移对 $\psi$ 的整体计算无害——等效于加了零向量。但作者强调：**无害性依赖于声明的粒度**——对整体函数无害的偏移可能对子计算有害，因为中间层的表征可能已经不同。

作者还引入了"行为二值子空间"(behaviorally binary subspace) 的概念：如果一个子空间仅通过其符号影响输出，那么只要 $\text{sign}(D_{\text{var}} \mathcal{A}(h))$ 不变，子空间内的值变化都是无害的，即使干预后的值组合在自然分布中从未出现过。

### 关键设计 3：隐藏通路与有害偏移

通过构造性证明展示有害偏移的两种形式：

**（a）隐藏通路激活**：构造一个两层 ReLU 网络，$s = \mathbf{1}^\top \text{ReLU}(W_\ell h^\ell + b_\ell)$，其中权重矩阵 $W_\ell \in \mathbb{R}^{3 \times 4}$。在自然表征下，第三个隐藏单元始终不激活（pre-activation 为负）。均值差 patching（$\delta_{B \to A} = \mu_A - \mu_B$）后的干预表征会使该单元激活，通过一个从未在自然输入下使用的通路翻转分类决策。将干预表征投影回 $\text{conv}(S_A)$ 后该效应消失，证实效果由偏移驱动而非因果机制。

**（b）休眠行为变化**：扩展上述网络加入上下文向量 $v$ 和第二层。干预在上下文 $v_4 < 0.75$ 时行为正常（预测 class A），但 $0.75 < v_4 < 1.0$ 时触发异常的 class C 预测——而自然表征下需要 $v_4 > 1$ 才会出现 C。休眠行为变化使干预安全性依赖上下文，穷举上下文不可行。形式化定义为 $\mathcal{V}(\psi, X, \mathcal{C}_1, \mathcal{C}) = \mathcal{N}(\psi, X, \mathcal{C}_1) \setminus \mathcal{N}(\psi, X, \mathcal{C})$。

### 损失函数：Counterfactual Latent (CL) Loss

**原始 CL loss**（来自 Grant 2025），结合 L2 距离和余弦距离：

$$\mathcal{L}_{\text{CL}}(\hat{h}, h_{\text{CL}}) = \frac{1}{2}\|\hat{h} - h_{\text{CL}}\|_2^2 - \frac{1}{2}\frac{\hat{h} \cdot h_{\text{CL}}}{\|\hat{h}\|_2 \|h_{\text{CL}}\|_2}$$

其中 $h_{\text{CL}}$ 是反事实潜在向量——从自然表征中取具有相同因果变量值的向量平均得到：$h_{\text{CL}} = \frac{1}{m} \sum_{i=1}^{m} h_{\text{CL}}^{(x_i)}$。总损失为 $\mathcal{L}_{\text{total}} = \epsilon \mathcal{L}_{\text{CL}} + \mathcal{L}_{\text{DAS}}$，其中 $\epsilon$ 是可调超参数。

**改进版 CL loss**，仅约束因果子空间维度，可独立于行为损失使用：

$$\mathcal{L}'_{\text{CL}} = \sum_{i=1}^{n} \left(\frac{1}{2}\|\hat{h}^{\text{var}_i} - h_{\text{CL}}^{\text{var}_i}\|_2^2 - \frac{1}{2}\frac{\hat{h}^{\text{var}_i} \cdot h_{\text{CL}}^{\text{var}_i}}{\|\hat{h}^{\text{var}_i}\|_2 \|h_{\text{CL}}^{\text{var}_i}\|_2}\right)$$

其中 $\hat{h}^{\text{var}_i} = \mathcal{A}^{-1}(D_{\text{var}_i} \mathcal{A}(\hat{h}))$ 是干预表征在因果子空间 $i$ 上的分量，$h_{\text{CL}}^{\text{var}_i}$ 用 stopgrad 处理防止梯度流回。

## 实验关键数据

### 主实验：偏移的普遍性（Section 3.2）

| 干预方法 | 模型 | 层 | EMD | 偏移显著 |
|---------|------|---|-----|---------|
| Mean Diff Vector Patching | Llama-3-8B-Instruct | L10 (最低 EMD 层) | 显著高于自然基线 | ✓ |
| SAE Reconstruction | Llama-3-8B-Instruct | L25 | 显著高于自然基线 | ✓ |
| Boundless DAS | wu2024 设置 | 指定层 | 显著高于自然基线 | ✓ |

三种主流方法在 PCA 可视化和 Earth Mover's Distance 量化上均显示干预表征明显偏离自然分布。作者还额外使用最近邻余弦距离、L2 配对距离、Local PCA Distance、KDE Density Score、Local Linear Reconstruction Error 等多种度量交叉确认，结论一致。

### CL Loss 在 Boundless DAS（7B LLM）上的效果（Section 5.1）

| CL 权重 $\epsilon$ | IIA (干预准确率) | EMD (偏移程度) | 说明 |
|---------------------|-----------------|---------------|------|
| 0（无 CL）| 基线 IIA | 较高 | 原始 DAS |
| 小 $\epsilon$ | 保持甚至略提升 | 明显降低 | **最优区间** |
| 大 $\epsilon$ | IIA 下降 | 最低 | CL 过强影响行为 |

关键发现：存在一个 sweet spot，小 $\epsilon$ 可在不牺牲 IIA 的前提下显著降低偏移。

### 改进 CL Loss 在合成任务上的效果（Section 5.2）

| 方法 | EMD (特征维度) | IIA | OOD 泛化 |
|------|---------------|-----|---------|
| DAS 行为损失 | 0.032 ± 0.003 | 0.997 ± 0.001 | 较低 |
| 改进 CL loss | **0.007 ± 0.001** | **0.9988 ± 0.0005** | **较高** |

CL loss 将 EMD 降低约 4.5 倍，IIA 略有提升。OOD 设置中（在 dense/sparse 子任务间迁移对齐矩阵），CL loss 训练的对齐显著优于行为损失。回归分析确认 EMD 与 OOD IIA 反相关（系数 -0.34，$R^2 = 0.73$，$p < 0.001$），证明减少偏移确实有实际价值。

### 关键发现

- 偏移不是个别方法的问题，而是因果干预的系统性问题
- 隐藏通路可在行为上看起来"正确"的同时完全使用非自然机制——最危险的情况
- 休眠行为变化使干预安全性依赖于上下文，而上下文空间不可穷举
- CL loss 提供简单有效的初步缓解方案，且有 OOD 泛化优势

## 亮点与洞察

1. **元方法学贡献**：不是在用可解释性工具分析模型，而是审视可解释性工具本身的可靠性。对整个领域的方法论基础有深远影响。

2. **"隐藏通路"概念**：干预可能激活自然状态下从未使用的计算路径，导致行为正确但机制错误的结论。直接挑战"高 IIA = 正确机制发现"的常见假设。

3. **无害 vs 有害的清晰框架**：通过行为零空间理论给出判断偏移有害性的原则方法，而非粗暴地视所有偏移为问题。

4. **定理 A.2 的优雅性**：只有轴对齐超矩形是 patch-closed 的——对几乎所有实际流形，坐标 patching 必然产生偏移。

5. **实用性**：CL loss 实现简单，可插入现有 DAS 流程，在 7B LLM 上验证有效。

## 局限性 / 可改进方向

1. **缺乏有害偏移的自动分类方法**：无法自动区分无害与有害偏移，限制实用性。
2. **CL loss 是"广撒网"策略**：同时减少所有偏移（含无害的），非精准消除有害偏移。
3. **改进版 CL loss 仅在简单合成任务验证**：10 类分类的合成数据集距真实 LLM 场景较远。
4. **限于线性对齐函数**：Sutter et al. 指出非线性 AF 有更根本的问题，本文未覆盖。
5. **CL 向量获取依赖标注**：需知道哪些自然表征具有相同因果变量值，复杂场景难获取。
6. **可探索方向**：(a) ReLU 激活模式审计的在线偏移检测；(b) 流形投影与 CL loss 结合；(c) 自监督发现有害偏移。

## 相关工作与启发

- **Makelov et al. (2023)**：先前指出 DAS 中零空间与休眠子空间的交互问题，本文推广到更广泛的因果干预方法。
- **Zhang et al. (2024) / Heimersheim (2024)**：指出 patching 结果易被误解，本文从表征偏移角度提供新的理论解释。
- **Sutter et al. (2025)**：质疑非线性 AF 下因果干预意义，与本文发现互补。
- **Grant (2025)**：原始 CL loss 来源，本文拓展到因果子空间级别。
- **对 SAE 研究的启示**：SAE 重建本身就是干预并会产生偏移，对 SAE 特征的"因果验证"步骤提出质疑。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 质疑可解释性研究的基本方法论假设，元层面重要贡献
- **实验充分度**: ⭐⭐⭐⭐ — 理论证明扎实，LLM 实验有意义，但改进方法仅在合成数据验证
- **写作质量**: ⭐⭐⭐⭐⭐ — 问题定义精准，逻辑清晰，理论与实验结合紧密
- **价值**: ⭐⭐⭐⭐⭐ — 对 mech interp 领域的因果干预实验有广泛影响，Oral 当之无愧
