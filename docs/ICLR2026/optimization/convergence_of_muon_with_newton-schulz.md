# Convergence of Muon with Newton-Schulz

**会议**: ICLR2026  
**arXiv**: [2601.19156](https://arxiv.org/abs/2601.19156)  
**代码**: 待确认  
**领域**: 优化/理论  
**关键词**: Muon optimizer, Newton-Schulz, polar decomposition, matrix optimization, convergence analysis

## 一句话总结
首次为实际使用的 Muon 优化器（使用 Newton-Schulz 近似而非精确 SVD 极坐标分解）提供非凸收敛保证：证明收敛速率匹配 SVD 理想化版本（差一个常数因子），该因子随 Newton-Schulz 步数 $q$ 双指数衰减，且 Muon 比向量对应物 SGD-M 少 $\sqrt{r}$ 倍秩损失。

## 研究背景与动机

1. **领域现状**：Muon 优化器通过正交化动量矩阵（而非像 Adam 那样向量化处理）来更新矩阵参数，在 LLM 训练中表现优异。实际使用 Newton-Schulz (NS) 迭代近似极坐标分解，避免昂贵的 SVD。
2. **现有痛点**：现有 Muon 理论分析（Shen et al., Li & Hong）都将 NS 替换为精确 SVD——但实际中从不用 SVD。NS 近似误差如何影响收敛？几步 NS 就够吗？why？
3. **核心矛盾**：实践中 Muon 用少量 NS 步就达到了 SVD 级别效果（更快的 wall-clock），但理论空白——实践远超理论。
4. **切入角度**：直接分析 NS 近似的极坐标误差 $\varepsilon_q$，证明它随步数双指数衰减。
5. **核心idea一句话**：NS 近似误差 $\varepsilon_q$ 双指数衰减→几步 NS 就将 Muon 收敛率拉到 SVD 级别→每步计算远低于 SVD→wall-clock 更快。

## 方法详解

### 整体框架
Muon 每步：(1) 计算随机梯度 $G_t$；(2) 动量更新 $M_t = \beta M_{t-1} + G_t$；(3) 预缩放 $X_{t,0} = M_t/\alpha_t$；(4) $q$ 步 NS 迭代 $X_{t,j} = p_\kappa(X_{t,j-1}X_{t,j-1}^\top)X_{t,j-1}$ 近似正交化；(5) 更新 $W_t = W_{t-1} - \eta O_t$。分析目标：$\frac{1}{T}\sum_t \mathbb{E}[\|\nabla f(W_{t-1})\|_*] \leq \epsilon$。

### 关键设计（理论贡献）

1. **Theorem 1: NS-Muon 非凸收敛**:
   - Muon with $q$ 步 NS 达到 $\epsilon$-平稳点的迭代数：$T = O\left(\frac{C_q \cdot L D}{\epsilon^2}\right)$
   - $C_q$ 是唯一依赖 NS 近似的常数因子

2. **Theorem 2: 极坐标近似误差双指数衰减**:
   - $\varepsilon_q \leq \varepsilon_0^{(2\kappa+1)^q}$——随步数 $q$ 双指数衰减，随多项式阶 $\kappa$ 也衰减
   - 意义：$q = 3-5$ 步，$\kappa = 2-3$ 就足以让 $C_q \approx 1$（匹配 SVD）
   - Wall-clock 优势：NS 只需矩阵乘法（GPU 高效），SVD 是 $O(mn \min(m,n))$

3. **Theorem 3: 比 SGD-M 的秩优势**:
   - Muon 收敛率比 SGD-M 快 $\sqrt{r}$ 倍（$r = \min(m,n)$，矩阵秩）
   - 原因：Muon 在 nuclear norm 下工作→利用了矩阵低秩结构→更高效的搜索方向

## 实验关键数据

### 主实验（收敛对比）

| 方法 | 度量 | 收敛率 | 秩依赖 |
|------|------|--------|--------|
| SGD-M | Frobenius 梯度 | $O(1/\sqrt{T})$ | $\sqrt{r}$ 损失 |
| Muon (SVD) | Nuclear 梯度 | $O(1/\sqrt{T})$ | 无 $\sqrt{r}$ 损失 |
| **Muon (NS, $q$ 步)** | Nuclear 梯度 | $O(C_q/\sqrt{T})$ | 无 $\sqrt{r}$ 损失 |

### 消融（$C_q$ vs 步数 $q$）

| NS 步数 $q$ | $\kappa=2$ 时 $C_q$ | $\kappa=3$ 时 $C_q$ |
|-------------|---------------------|---------------------|
| 1 | 大 | 中等 |
| 3 | $\approx 1.01$ | $\approx 1.001$ |
| 5 | $\approx 1.0$ | $\approx 1.0$ |

### 关键发现
- **3-5 步 NS 就匹配 SVD**：$C_q$ 双指数收敛到 1，实际中的选择有充分理论根据
- **Muon vs SGD-M 的 $\sqrt{r}$ 优势**：对高秩矩阵参数（如大 attention 层），优势显著
- **wall-clock 优势解释**：NS 每步成本远低于 SVD，迭代数几乎相同→总时间更少

## 亮点与洞察
- **首次为实际 Muon 提供理论保证**：关闭了 practice-theory gap。之前所有理论都"假装"用 SVD
- **双指数衰减是关键 insight**：$\varepsilon_q \leq \varepsilon_0^{5^q}$（$\kappa=2$ 时）——3步误差 $< 10^{-100}$
- **Nuclear norm 度量的选择**：在矩阵空间用 nuclear norm 而非 Frobenius——自然匹配极坐标分解，揭示秩优势
- **对未来矩阵优化器的启发**：NS 近似的通用分析框架可扩展到其他矩阵优化器

## 局限性 / 可改进方向
- 纯理论贡献，无新实验（但论文目标就是解释已有实践）
- 假设标准光滑+有界方差，未覆盖 Adam 风格自适应
- 未分析 Muon 与 Shampoo/SOAP 等二阶方法的比较

## 相关工作与启发
- **vs Shen et al. / Li & Hong**：他们分析 SVD-Muon。本文首次分析 NS-Muon——唯一匹配实践的理论
- **vs Shampoo/SOAP**：二阶预条件器，维护曲率。Muon 非二阶——正交化动量，机制不同可互补
- **vs Orthogonal-SGDM**：先正交化再动量。Muon 先动量再正交化+用 NS 替代 SVD

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次为实际使用的 NS-Muon 提供收敛理论，双指数衰减结果优美
- 实验充分度: ⭐⭐⭐ 纯理论，无新实验（但合理——补充已有实践的理论解释）
- 写作质量: ⭐⭐⭐⭐⭐ 研究问题清晰、定理层层递进、叙述严谨流畅
- 价值: ⭐⭐⭐⭐⭐ 为当下最热门的矩阵优化器提供了急需的理论基础
