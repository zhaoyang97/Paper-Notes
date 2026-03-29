# ComPO: Preference Alignment via Comparison Oracles

**会议**: NeurIPS 2025  
**arXiv**: [2505.05465](https://arxiv.org/abs/2505.05465)  
**代码**: https://github.com/PeterLauLukChen/ComparisonPO  
**领域**: LLM/NLP  
**关键词**: 偏好对齐, DPO, 比较oracle, 似然位移, 零阶优化

## 一句话总结
针对DPO中噪声偏好对（preferred和dispreferred响应相似）导致的似然位移和冗长问题，提出基于比较oracle的零阶偏好对齐方法ComPO，将数据分为干净/噪声子集，用DPO处理干净数据、用ComPO提取噪声数据中的信号，在AlpacaEval 2等benchmark上持续提升LC win rate。

## 研究背景与动机

1. **领域现状**：直接偏好对齐方法（DPO及其变体）因简单高效已广泛应用于LLM对齐。DPO通过最大化preferred和dispreferred响应的log-likelihood差值来优化策略。
2. **现有痛点**：DPO存在两个关键问题——(a) **似然位移（likelihood displacement）**：训练过程中preferred响应的绝对概率反而下降，概率质量被转移到意义相反的响应上（如从"No"转移到"Yes"），甚至导致原本安全的模型变得不安全；(b) **冗长性（verbosity）**：模型倾向生成更长但不一定更好的回复。这两个问题主要由噪声偏好对引起——当preferred和dispreferred响应很相似时，DPO的代理目标函数无法正确提取信息。
3. **核心矛盾**：噪声偏好对（相似的response pair）确实包含有用信息，但DPO的log-likelihood margin代理目标无法有效提取。直接丢弃噪声数据（如CHES score过滤）是一种浪费。
4. **本文要解决什么？** 设计一种方法从噪声偏好对中提取有用的对齐信号，而不是通过代理目标函数间接优化。
5. **切入角度**：借鉴优化理论中的**比较oracle**（只需"A比B好还是差"的二值判断，不需要目标函数值或梯度），将偏好对视为比较oracle的输出，直接用比较信号引导参数更新。
6. **核心idea一句话**：不定义显式的代理目标函数，而是用偏好对作为比较oracle，通过零阶优化直接引导模型参数向"preferred likelihood更高、dispreferred likelihood更低"的方向移动。

## 方法详解

### 整体框架

ComPO采用三阶段流程：
1. 用参考模型将数据集按log-likelihood margin分为**干净子集**（margin > δ）和**噪声子集**（margin ≤ δ）
2. 在干净子集上运行标准DPO（DPO_clean）
3. 从DPO_clean的输出策略出发，在噪声子集上运行ComPO（Algorithm 2）

### 关键设计

1. **偏好比较oracle（Preference Comparison Oracle）**:
   - 做什么：定义一种新的参数比较方式，不依赖显式目标函数
   - 核心思路：给定参数 $\theta$ 和 $\theta'$，比较oracle $\mathcal{C}_\pi(\theta, \theta')$ 返回-1当且仅当 $\pi_{\theta'}(\mathbf{y}^+|\mathbf{x}) > \pi_\theta(\mathbf{y}^+|\mathbf{x})$ 且 $\pi_{\theta'}(\mathbf{y}^-|\mathbf{x}) < \pi_\theta(\mathbf{y}^-|\mathbf{x})$，即 $\theta'$ 同时满足preferred likelihood更高和dispreferred likelihood更低
   - 设计动机：DPO只优化likelihood margin（差值），可能增大差值但两个likelihood都在下降。比较oracle要求两个likelihood同时朝正确方向移动，严格避免似然位移

2. **基本方案（Algorithm 1）+ 收敛保证**:
   - 做什么：基于比较oracle的零阶优化基本框架
   - 核心思路：生成m个随机扰动 $\mathbf{z}_i$，查询oracle $y_i = \mathcal{C}_\pi(\theta_t, \theta_t + r\mathbf{z}_i)$，利用1-bit compressed sensing原理恢复归一化稀疏梯度 $\hat{\mathbf{g}}_t = \arg\max_{\|\mathbf{g}\|_1 \leq \sqrt{s}, \|\mathbf{g}\| \leq 1} \sum_i y_i \mathbf{z}_i^\top \mathbf{g}$，然后更新 $\theta_{t+1} = \theta_t - \eta \hat{\mathbf{g}}_t$
   - 在非凸光滑函数假设下提供收敛保证，oracle调用次数为 $O(\frac{\ell \Delta}{\epsilon^2}(s\log(\frac{2d}{s}) + \log(\frac{\ell\Delta}{\Lambda\epsilon^2})))$，依赖梯度稀疏度s而非维度d
   - 设计动机：虽然对齐的真实目标函数未知（无法求值或求梯度），但比较信号足以估计归一化梯度方向

3. **实用方案（Algorithm 2）**:
   - 做什么：将理论方案适配到十亿参数规模
   - 三个关键近似：(a) **只扰动输出层权重**——将扰动维度从d（数十亿）降到 $d^o$（输出层维度），大幅减少计算/内存开销；(b) **用归一化+裁剪近似稀疏梯度估计**——先计算 $\hat{\mathbf{g}}^o = \frac{\sum y_i \mathbf{z}_i}{\|\sum y_i \mathbf{z}_i\|}$，然后将幅值小于 $\lambda_g$ 的分量置零；(c) **自适应步长**——按"改进"比例 $\frac{|\{i: y_i=-1\}|}{m}$ 调整步长，信息不足时跳过更新
   - 设计动机：让零阶方法在LLM规模下可行，同时保留核心的比较oracle思想

4. **数据划分策略**:
   - 噪声判定：$|\log\pi_{ref}(\mathbf{y}^+|\mathbf{x}) - \log\pi_{ref}(\mathbf{y}^-|\mathbf{x})| \leq \delta$，使用 $\delta=3$
   - 虽然作者承认log-likelihood margin不如CHES score精确，但计算简单且实验表明ComPO能有效利用这些"被抛弃"的噪声数据

### 训练策略
- 模型：Mistral-7B, Llama-3-8B, Gemma-2-9B（base和instruct变体）
- ComPO超参：r=0.0005~0.00075, m=1600~1800, $\lambda_g$=0.00008~0.00022, $\lambda$=0.2
- 评估：AlpacaEval 2, Arena-Hard, MT-Bench
- 30×NVIDIA A40 GPU

## 实验关键数据

### 主实验（Mistral-Instruct-7B为例）

| 方法 | AlpacaEval 2 LC% | AlpacaEval 2 WR% | Arena-Hard WR% | MT-Bench Avg |
|------|------------------|-------------------|-----------------|-------------|
| DPO | 24.14 | 16.71 | 14.4 | 5.86 |
| DPO_clean | 23.89 | 16.15 | 14.2 | 5.73 |
| **DPO_clean + ComPO** | **26.17** | **18.32** | 10.5 | **7.69** |

ComPO在LC（长度控制）win rate上持续提升（+2.03~2.28%），说明减少了冗长性。

### 消融：似然位移缓解

| 方法 | 似然位移比例 | 说明 |
|------|-------------|------|
| DPO (全数据) | 较高 | 噪声数据导致严重位移 |
| DPO_clean | 降低 | 过滤噪声后减少 |
| DPO_clean + ComPO | **最低** | ComPO有效利用噪声数据且不引起位移 |

### 关键发现
- **噪声数据有价值但需要正确利用**：直接过滤（DPO_clean）反而可能略降性能，用ComPO处理噪声数据则能提升
- **LC win rate一致改善**：跨所有模型和基准，ComPO特别擅长提升LC（相对于raw WR），说明减少了冗长性
- **比较oracle避免了似然位移的根源**：因为oracle要求preferred/dispreferred likelihood同时朝正确方向变化，不允许"差值增大但都在下降"
- 仅扰动输出层权重就足够有效，大幅降低了零阶方法的成本

## 亮点与洞察
- **将优化理论的比较oracle引入LLM对齐**的跨领域思路非常新颖：不定义代理目标函数，直接用偏好对的二值比较信号做优化，从理论上规避了DPO代理函数不准确的问题
- **干净/噪声数据分治策略**实用：DPO处理容易的数据、ComPO处理困难的数据，两者互补
- **收敛性证明**扩展到了非凸情况，比原始比较oracle方法（仅证凸情况）有理论贡献

## 局限性 / 可改进方向
- 噪声/干净的划分阈值δ=3是手动设定的，对不同数据集可能需要调参
- 实用方案只扰动输出层——信息是否充分？对于更深层表示的对齐可能不够
- 只报告了5次试验中的最佳结果，缺少mean±std，可能有较大方差
- Arena-Hard上有时ComPO反而低于DPO（如Mistral-Instruct），说明方法不是在所有维度都稳定提升
- 可探索与CHES score结合——用CHES做噪声划分可能比log-likelihood margin更准确

## 相关工作与启发
- **vs DPO**: DPO优化log-likelihood margin代理目标，ComPO绕过代理直接用比较信号。两者互补——干净数据用DPO，噪声数据用ComPO
- **vs Razin et al. (CHES score过滤)**: CHES直接丢弃噪声数据，ComPO则尝试挖掘噪声数据的价值
- **vs SimPO**: SimPO去除了参考模型，ComPO可以在SimPO基础上做进一步改进

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将比较oracle引入偏好对齐是全新思路，理论和方法都有实质贡献
- 实验充分度: ⭐⭐⭐⭐ 多模型多benchmark，但缺少mean±std和更大模型实验
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，但从基本方案到实用方案的gap较大
- 价值: ⭐⭐⭐⭐ 提供了处理噪声偏好数据的新视角，对RLHF/DPO社区有启发
