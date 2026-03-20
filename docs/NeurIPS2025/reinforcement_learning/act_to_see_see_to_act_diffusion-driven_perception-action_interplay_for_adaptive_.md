# Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies

**会议**: NeurIPS 2025  
**arXiv**: [2509.25822](https://arxiv.org/abs/2509.25822)  
**代码**: [项目主页](https://jingwang18.github.io/dp-ag.github.io/)  
**领域**: 扩散模型 / 机器人学习 / 模仿学习  
**关键词**: diffusion policy, perception-action loop, VJP, variational inference, imitation learning  

## 一句话总结
提出 DP-AG（Action-Guided Diffusion Policy），通过将扩散策略的噪声预测的 Vector-Jacobian Product (VJP) 作为结构化随机力驱动隐观测特征在扩散步骤间动态演化，并用循环一致对比损失闭合感知-动作环路，在 Push-T 上提升 6%、Dynamic Push-T 上提升 13%、真实 UR5 机器人上成功率提升 23%+。

## 研究背景与动机

1. **领域现状**：模仿学习（IL）方法中，Diffusion Policy (DP) 通过扩散去噪过程建模动作分布，在机器人操控中取得 SOTA。但 DP 和其他方法都将感知和动作**解耦**——观测特征在一次动作序列生成过程中保持静态不变。

2. **现有痛点**：静态特征编码在整个动作序列生成期间"冻结"，忽略了动作生成过程中的中间反馈——不能像人类那样根据正在执行的动作动态调整对环境的理解。这导致动作序列不够连贯平滑，尤其在动态环境中。

3. **核心矛盾**：扩散策略的去噪过程内已蕴含丰富的中间动作信息（每步的噪声预测），但这些信息未被用来改进感知表示——感知和动作是单向的（感知→动作），缺少动作→感知的反馈。

4. **本文要解决什么？**
   - 如何建立感知和动作之间的双向闭环？
   - 如何利用扩散过程中的中间噪声预测来动态更新观测特征？

5. **切入角度**：受人类感知-动作耦合的启发（"Act to See, See to Act"），将扩散噪声预测对隐特征的 VJP 作为"结构化随机力"驱动隐特征演化，使感知表示与动作精炼同步进化。

6. **核心 idea 一句话**：用扩散策略噪声预测的 VJP 驱动隐观测特征的 SDE 演化，闭合感知-动作环路。

## 方法详解

### 整体框架
DP-AG 在标准 Diffusion Policy 基础上增加三个组件：(1) **变分推断**：将观测特征编码为高斯后验 $q_\phi(z_t|o_t) = \mathcal{N}(\mu_\phi, \sigma_\phi^2)$；(2) **动作引导 SDE**：隐特征在扩散步 $k$ 间演化 $d\tilde{z}_t^k = \text{VJP}(\hat{a}_t^k, z_t) dt + \sigma_\phi dW_t$；(3) **循环一致对比损失**：对齐静态和演化隐特征条件下的噪声预测。动作生成过程中，观测特征不再静止，而是随动作精炼同步更新。

### 关键设计

1. **VJP 驱动的隐特征演化**
   - 做什么：让隐观测特征在每个扩散去噪步骤中根据动作反馈动态更新。
   - 核心思路：VJP 计算噪声预测关于隐特征的"向后传播"方向：$\text{VJP}(\hat{a}_t^k, z_t) = (\frac{\partial \epsilon_\theta}{\partial z_t})^\top \epsilon_\theta$。这个方向指向最能减少动作不确定性的特征调整方向。离散化后：$\tilde{z}_t^k = \mu_\phi(z_t) + \gamma \sigma_\phi(z_t) \odot \text{VJP}(\hat{a}_t^k, z_t)$。
   - 设计动机：VJP 提供"任务驱动的注意力"——引导隐特征关注观测中对当前动作最重要的部分。类比开车：虽然窗外景色不变，但转弯时注意力集中在弯道边缘，加速时关注前车。VJP 就是这种动态注意力的计算实现。

2. **循环一致对比损失（Cycle-Consistent InfoNCE）**
   - 做什么：确保隐特征演化与动作扩散保持一致，防止过度漂移。
   - 核心思路：在每步 $k$，计算两个噪声预测：$\varepsilon_k = \epsilon_\theta(\hat{a}_t^k, z_t, k)$（静态隐特征）和 $\tilde{\varepsilon}_k = \epsilon_\theta(\hat{a}_t^k, \tilde{z}_t^k, k)$（演化隐特征）。用 InfoNCE 损失拉近匹配对、推远不匹配对：$\mathcal{L}_{\text{cont}} = -\frac{1}{B}\sum_i \log \frac{\exp(\text{sim}(\varepsilon_k^i, \tilde{\varepsilon}_k^i)/\tau)}{\sum_{j \neq i} \exp(\text{sim}(\varepsilon_k^i, \tilde{\varepsilon}_k^j)/\tau)}$。
   - 设计动机：VJP 引导的演化可能导致隐特征漂移太远。对比损失将演化特征"锚定"在静态特征附近，保持语义一致性。理论证明（Theorem 1）：在 Lipschitz 条件下，对比损失的最小化为隐特征漂移提供上界：$\|\tilde{z}_t^{k+1} - \tilde{z}_t^k\|^2 \leq 2L^2(2 - \tau\ln(B-1) + \tau\alpha)$。

3. **变分推断基础**
   - 做什么：为隐特征演化提供概率框架。
   - 核心思路：推导动作引导 ELBO（Eq. 12）：$\log p(\varepsilon_k|z_t) \geq \mathbb{E}_{q_\phi}[\log p(\tilde{\varepsilon}_k|\tilde{z}_t^k)] - \text{KL}(q_\phi \| p)$。KL 项鼓励演化隐特征接近先验 $p(\tilde{z}_t^k|z_t) = \mathcal{N}(z_t, I)$。

### 损失函数 / 训练策略
- 总损失：$\mathcal{L}_{\text{DP-AG}} = \mathcal{L}_{\text{DP}} + \lambda_{\text{cont}} \mathcal{L}_{\text{cont}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}}$
- 噪声匹配项保持动作预测能力，对比项闭合感知-动作环，KL 项防止隐特征漂移

## 实验关键数据

### 主实验：Push-T 和 Dynamic Push-T

| 方法 | Push-T (img) | Push-T (kp) | Dynamic Push-T |
|------|-------------|-------------|----------------|
| IBC | 0.75 | 0.90 | 0.52 |
| AdaFlow | 0.87 | 0.91 | 0.67 |
| DP (baseline) | 0.87 | 0.95 | 0.65 |
| **DP-AG (ours)** | **0.93** | **0.99** | **0.80** |

Push-T 图像模态：0.87 → **0.93** (+6%)；Dynamic Push-T：0.65 → **0.80** (+13%)。

### 真实机器人 UR5 实验

| 任务 | DP 成功率 | DP-AG 成功率 | 提升 |
|------|----------|-------------|------|
| 操控任务 | ~60% | ~83% | **+23%** |
| 动作平滑度 | 基线 | **↓60%** 更平滑 | 显著 |

### 消融实验：各组件贡献

| 配置 | 不规则螺旋 MSE |
|------|---------------|
| Base Flow (无 VJP) | 0.0095 |
| **VJP-Guided Flow** | **0.0052** (↓45.3%) |

### 关键发现
- **VJP 引导的隐特征演化形成结构化流形**：可视化显示 Base Flow 的隐状态散乱，VJP 引导后形成与输出对齐的结构化流形
- **动态环境中优势更大**：Dynamic Push-T（+13%）比静态 Push-T（+6%）改善更显著——因为动态场景更需要自适应感知
- **对比损失优于 MSE**：用对比损失（相对相似性）替代 MSE（绝对匹配）避免了过度刚性约束，允许有界的自适应
- **VJP 计算开销极小**：利用现代自动微分框架，VJP 计算可忽略不计

## 亮点与洞察
- **"Act to See, See to Act"的生物学启发**非常直觉且有深度：将人类主动感知的循环本质引入策略学习，不是获取新观测，而是根据动作**重新解读**同一观测——这是认知科学中"enactive perception"的计算实现。
- **VJP 作为"任务驱动注意力"**的解释框架很精彩：VJP 的方向指向最能减少动作不确定性的特征调整方向，大小反映不确定性程度——不确定时大幅调整，确定时微调。
- **理论与实验的完美配合**：Lemma 1 和 Theorem 1 不是空泛的理论保证，而是直接预测了对比损失最小化→隐特征漂移有界→轨迹连续性提升的因果链，并被实验验证。

## 局限性 / 可改进方向
- **VJP 计算需要通过噪声预测器反向传播**：虽说开销小，但在大模型上可能不可忽略
- **仅在 DDPM 上验证**：虽声称可扩展到 flow matching 等，但实验未验证
- **隐特征演化的 γ 超参数**：VJP 强度需要调节，不同任务可能需要不同设置
- **单视角观测**：未验证多视角或触觉等其他模态的效果

## 相关工作与启发
- **vs Diffusion Policy (Chi et al.)**：DP 保持静态特征，DP-AG 动态演化特征——核心区别。DP-AG 在 DP 的动作连续性基础上扩展到感知连续性
- **vs PlaNet/Dreamer (隐状态模型)**：这些方法用 VAE 预测未来状态用于规划，但不在动作生成过程中动态更新特征——维度不同
- **vs VLA (OpenVLA, π₀)**：VLA 通过视觉-语言模型增强感知，但仍是静态的；DP-AG 通过动作反馈动态增强感知，两者可能互补
- **可迁移启示**：VJP 作为隐特征演化驱动力的思路可迁移到任何条件生成过程——如文本生成中用生成内容的反馈调整上下文表示

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 感知-动作闭环的设计理念独特，VJP 驱动隐特征演化是全新的技术路径
- 实验充分度: ⭐⭐⭐⭐⭐ 理论验证(螺旋)+仿真(Push-T/Robomimic/Kitchen)+真实机器人(UR5)，层层递进
- 写作质量: ⭐⭐⭐⭐⭐ 生物学启发→数学框架→理论保证→实验验证的叙事链条极其流畅
- 价值: ⭐⭐⭐⭐⭐ 为扩散策略引入动态感知机制，在机器人操控上有显著实际改善
