# DEPO: Dual-Efficiency Preference Optimization for LLM Agents

**会议**: AAAI 2026  
**arXiv**: [2511.15392](https://arxiv.org/abs/2511.15392)  
**代码**: [https://opencausalab.github.io/DEPO](https://opencausalab.github.io/DEPO)  
**领域**: Agent / LLM  
**关键词**: LLM Agent 效率优化, 偏好优化, KTO, 双重效率, 强化学习  

## 一句话总结

提出双重效率（dual-efficiency）的概念，将 LLM Agent 的效率分解为 step 级（减少每步 token 数）和 trajectory 级（减少总步数），并基于 KTO 设计了 DEPO 方法，通过在 desirable 样本的 reward 中加入效率 bonus 来联合优化效率与性能。

## 背景与动机

随着 LLM 推理能力增强，CoT 越来越长，导致 Agent 与环境交互时效率低下。现有效率研究主要关注单次响应的 token 压缩，忽视了 Agent 场景下的两个关键开销来源：
1. **每步生成的 token 数量**：过度思考导致单步响应冗长（step-level 低效）
2. **完成任务的总步数**：推理不精确导致需要更多交互步骤（trajectory-level 低效）

现有 RL 方法（PPO、DPO、GRPO、KTO）主要关注学习动态和性能提升，缺乏对 Agent 交互效率的显式优化。

## 核心问题

如何在不牺牲任务性能的前提下，同时减少 LLM Agent 的每步 token 数和总交互步数？

## 方法详解

### 整体框架

三阶段流水线：
1. **MCTS 数据生成**：使用 DeepSeek-V3 进行蒙特卡洛树搜索，生成 ReAct 格式（Thought + Action）的轨迹数据
2. **行为克隆（BC）SFT**：在高质量 desirable 轨迹上做标准 SFT，学习基础策略 $\pi_{\text{BC}}$
3. **DEPO 偏好优化**：在 BC 基础上做效率感知的偏好学习

### 关键设计

**双重效率定义**：
- Step-level 效率：最小化每步生成的 token 数
- Trajectory-level 效率：最小化完成任务所需的总步数

**数据标注与过滤**：
- Desirable ($\mathcal{D}$)：reward $r(\tau) \geq \kappa_0$（BabyAI: $\geq 0.9$; Webshop: $= 1.0$）
- Undesirable ($\mathcal{U}$)：$\kappa_2 \leq r(\tau) < \kappa_1$（中间质量区间 $[0.7, 0.9)$）
- 过低质量直接丢弃，确保 desirable 和 undesirable 之间有 margin
- 额外按步数过滤：$<7$ 步归入 $\mathcal{D}$，$\geq 7$ 步归入 $\mathcal{U}$
- 使用 GPT-4.1 mini 对 Thought 部分做 rephrasing，使 desirable 轨迹每步 token 更少

**效率 Bonus 设计**（核心创新）：

在 KTO 的 implied reward 中加入参数无关的效率偏移量：

$$r_\theta(\tau) = \log \frac{\pi_\theta(a_t | \tau_t)}{\pi_{\text{BC}}(a_t | \tau_t)} + b(\tau)$$

其中 bonus $b(\tau)$ 定义为：

$$b(\tau) = \begin{cases} \frac{\alpha_1}{\bar{T}_{\text{token}}(\tau)} + \frac{\alpha_2}{T_{\text{step}}(\tau)}, & \text{if } \tau \in \mathcal{D} \\ 0, & \text{if } \tau \in \mathcal{U} \end{cases}$$

- $\bar{T}_{\text{token}}$：每步平均 token 数（越大 → bonus 越小 → 惩罚冗长）
- $T_{\text{step}}$：总步数（越多 → bonus 越小 → 惩罚低效轨迹）
- **仅对 desirable 样本施加 bonus**，undesirable 不加惩罚（消融实验验证加惩罚反而有害）

### 损失函数 / 训练策略

基于 KTO（Kahneman-Tversky Optimization）框架：

$$\mathcal{L}_{\text{KTO}}(\theta) = \mathbb{E}_{\tau \sim \mathcal{D}, \mathcal{U}} [\lambda(\tau) - v(\tau)]$$

其中 value function 对 desirable 和 undesirable 分别使用 sigmoid 处理：
- Desirable: $v(\tau) = \lambda_D \cdot \sigma(\beta(r_\theta(\tau) - z_0(\tau)))$
- Undesirable: $v(\tau) = \lambda_U \cdot \sigma(\beta(z_0(\tau) - r_\theta(\tau)))$

$z_0(\tau)$ 为当前策略与 BC 策略的 KL 散度正则项。

**训练配置**：
- LoRA 微调，BC 阶段 lr=1e-4，DEPO 阶段 lr=2e-5，各 3 epochs
- $\beta=0.2$，$\lambda_D = \lambda_U = 1$
- Llama3.1-8B: $\alpha_1 = \alpha_2 = 3$；Qwen2.5-7B: $\alpha_1 = \alpha_2 = 2$
- BabyAI: 512 desirable + 471 undesirable；Webshop: 各 1567 条
- 8 × A800 80GB GPU

## 实验关键数据

**主实验（Table 1）**：

| 模型 | 指标 | Webshop Succ.↑ | Webshop T@All↓ | BabyAI Succ.↑ | BabyAI T@All↓ |
|------|------|------|------|------|------|
| Llama3.1-8B-BC | baseline | 0.47 | 840 | 0.77 | 836 |
| + KTO | | 0.48 | 776 | 0.87 | 342 |
| **+ DEPO** | | **0.50** | **633** | **0.88** | **327** |
| Qwen2.5-7B-BC | baseline | 0.44 | 1014 | 0.47 | 2062 |
| + KTO | | 0.54 | 886 | 0.58 | 1199 |
| **+ DEPO** | | **0.56** | **726** | **0.75** | **893** |

**效率提升汇总**（相对 BC baseline）：
- Token 使用减少最高 **60.9%**（Llama, BabyAI T@All）
- 步数减少最高 **26.9%**（Qwen, BabyAI S@All vs KTO）
- 性能提升最高 **29.3%**（Qwen, BabyAI Succ. vs BC）

**泛化性（Figure 2）**：
- 在 GSM8K、MATH、SimulEq 三个 OOD 数学基准上，DEPO 训练的模型平均准确率提升，token 使用减少
- Llama3.1-8B-BC+DEPO 跨域效率提升明显；Qwen2.5-7B-BC+DEPO token 略有增加

**样本效率（Figure 3）**：
- 仅用 25% 训练数据（BabyAI 245 条，Webshop 783 条），T@All 效率提升超过 10%
- 100% 数据时 T@All 提升近 60%

### 消融实验要点

1. **$\alpha_1$ 和 $\alpha_2$ 联合设置最优**：单独优化某一维度可能提升该维指标但损害性能；如 $\alpha_1=0, \alpha_2>0$ 时 Qwen 的 Succ. 和 Reward 明显下降
2. **undesirable 惩罚无效**：对 undesirable 轨迹加等强度惩罚后，Llama 在 BabyAI 上 T@All +46.5%、S@All +39.4%（效率大幅恶化），性能也下降

## 亮点

1. **概念清晰**：将 Agent 效率解耦为 step-level 和 trajectory-level 两个维度，方便针对性优化
2. **方法简洁优雅**：仅在 KTO reward 中加一个效率 bonus 项，无需额外 reward model、配对标注、on-policy 采样，实现简单且训练稳定
3. **效率与性能双赢**：不是以性能换效率，而是在提升效率的同时性能也有提升（最高 29.3%）
4. **样本高效**：25% 数据即可获得显著效率增益，适合数据稀缺场景
5. **跨域泛化**：在 Webshop/BabyAI 上训练，可迁移到数学推理任务

## 局限性 / 可改进方向

1. **评估场景有限**：仅在 Webshop（网购）和 BabyAI（网格世界）两个相对简单的环境上训练和主评估，缺少更复杂的真实场景（如 web browsing、code execution）
2. **数据生成依赖强模型**：MCTS 搜索使用 DeepSeek-V3，rephrasing 使用 GPT-4.1 mini，数据生成成本不低
3. **效率 bonus 设计偏简单**：使用倒数形式 $1/T$ 的 bonus，没有考虑任务难度的归一化——简单任务天然步数少应获得更少奖励、难任务步数多但合理不应被过度惩罚
4. **缺少与其他效率方法的对比**：仅与 Token Budget (TB) 做了比较，缺少与 L1 regularization、DAST 等 RL-based 高效推理方法的直接比较
5. **OOD 泛化评估存疑**：在数学任务上的泛化测试仅报告 accuracy 和 avg tokens，没有步数对比；且 Qwen 的 token 并未显著减少
6. **未讨论对推理质量的影响**：压缩 Thought 可能导致推理质量下降，论文未分析 Thought 质量变化

## 与相关工作的对比

| 方法 | 类型 | 效率优化 | 需配对数据 | 需 Reward Model | 在线采样 |
|------|------|------|------|------|------|
| ETO | 离线 RL (DPO) | ✗ | ✓ 对比对 | ✗ | ✗ |
| DMPO | 离线 RL (DPO) | ✗（仅长度归一化）| ✓ | ✗ | ✗ |
| RAGen/StarPO | 在线 RL | ✗ | ✗ | ✗ | ✓ |
| GiGPO | 在线 RL | ✗ | ✗ | ✗ | ✓ |
| L1/DAST | RL + 长度惩罚 | step-level | 视方法 | 视方法 | 视方法 |
| **DEPO** | 离线 RL (KTO) | **双重** | **✗** | **✗** | **✗** |

DEPO 的优势在于：(1) 同时优化两个维度的效率；(2) 基于 KTO 无需配对数据；(3) 纯离线训练。

## 启发与关联

1. **效率 bonus 的设计思路可迁移**：类似的效率奖励可以嵌入 DPO/GRPO 等其他偏好优化框架中
2. **与 "思考预算" 方向互补**：Token Budget 控制输出上限，DEPO 从训练侧优化输出分布，两者可结合
3. **对 Agent 系统设计的启示**：实际部署中 API 调用延迟和成本随步数线性增长，trajectory-level 效率可能比 step-level 效率更重要
4. **数据质量工程值得关注**：MCTS + rephrasing 的数据构造流程虽然有效，但引入了对外部强模型的依赖，如何自举（self-play）生成高效数据是有价值的后续方向

## 评分 (⭐ 1-5)

⭐⭐⭐ (3/5)

**优点**：问题定义清晰，方法简洁实用，实验充分（消融、泛化、样本效率都有覆盖），效率提升显著。

**不足**：核心创新有限——本质是在 KTO reward 上加了一个手工设计的效率 bonus 项，技术深度一般。评估场景偏简单（Webshop 和 BabyAI 不够复杂），缺少与更多效率优化方法的对比。效率 bonus 未考虑任务难度归一化，可能在异构任务集上表现不稳定。属于扎实但缺乏惊喜的工作。
