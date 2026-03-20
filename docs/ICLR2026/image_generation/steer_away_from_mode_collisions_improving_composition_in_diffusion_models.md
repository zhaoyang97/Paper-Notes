# Steer Away From Mode Collisions: Improving Composition In Diffusion Models

**会议**: ICLR 2026  
**arXiv**: [2509.25940](https://arxiv.org/abs/2509.25940)  
**代码**: [https://github.com/debottam-dutta7/co3](https://github.com/debottam-dutta7/co3)  
**领域**: 扩散模型 / 组合生成  
**关键词**: 组合生成, Mode Collision, Tweedie Mean 组合, 无梯度校正, 即插即用  

## 一句话总结

针对扩散模型多概念 prompt 中的概念缺失/碰撞问题，提出"模式碰撞"假说（联合分布与单概念分布的模式重叠），设计 CO3（Concept Contrasting Corrector）通过在 Tweedie 均值空间中组合校正分布 $\tilde{p}(x|C) \propto p(x|C) / \prod_i p(x|c_i)$ 来远离退化模式，实现即插即用、无梯度、模型无关的组合生成改进。

## 研究背景与动机

1. **领域现状**：扩散模型在文本到图像生成方面取得巨大突破，但即使是简单的多概念 prompt（如"a cat and a dog"），也时常出现概念缺失、模糊或不自然融合的问题。
2. **现有痛点**：
   - 基于优化的校正方法（Attend-Excite, SynGen, ToMe）需要对模型做梯度计算，是模型相关的
   - 可组合扩散方法（Composable Diffusion）模型无关但效果差，因为线性分数组合在 $t>0$ 时不对应正确的前向分布
   - 两者各有长短但无统一框架
3. **核心矛盾**：多概念 prompt 的联合分布 $p(x|C)$ 存在"问题模式"——这些模式与单概念分布 $p(x|c_i)$ 高度重叠，导致生成偏向某个强势概念。
4. **本文要解决什么？** (i) 理论上分析并统一现有方法（校正 vs 组合扩散）；(ii) 设计一个无训练、无梯度、模型无关的采样校正策略来改善多概念组合。
5. **切入角度**：提出"模式碰撞"假说——当 $p(x|C)$ 的某些模式与 $p(x|c_i)$ 重叠时，采样会倾向该单概念。解决思路是设计一个校正分布来抑制这些重叠区域。
6. **核心 idea 一句话**：通过 $\tilde{p}(x|C) \propto p(x|C) / \prod_i p(x|c_i)$ 远离单概念主导的退化模式，并证明 Tweedie 均值组合是一个统一框架。

## 方法详解

### 整体框架

CO3 是一个采样阶段的校正器，在 DDIM 采样过程中嵌入。在去噪的前 20% 步骤中执行校正，分为两个阶段：(1) CO3-resampler：前 3 步进行噪声重采样（权重和为 0）；(2) CO3-corrector：后续步骤进行潜码校正（权重和为 1）。不修改模型参数，不需要梯度计算。

### 关键设计

1. **Tweedie 均值组合框架**:
   - 做什么：统一校正方法和可组合扩散方法，提供理论基础
   - 核心思路：将分布组合从分数空间迁移到 Tweedie 均值空间 $\tilde{x}_{\text{tweedie}} = w_0 \hat{x}_{\text{tweedie}}[\epsilon_t^{\lambda,C}] + \sum_{k=1}^K w_k \hat{x}_{\text{tweedie}}[\epsilon_t^{\lambda,c_k}]$，其中 $\hat{x}_{\text{tweedie}}[\epsilon_t^{\lambda,c}] = x_t - \sigma_t \epsilon_t^{\lambda,c}$
   - 设计动机：Proposition 1 证明——当 $\sum_k w_k = 1$ 时组合的 Tweedie 均值仍是合法的 Tweedie 均值（CO3-corrector）；当 $\sum_k w_k = 0$ 时退化为加权噪声（CO3-resampler，仅在 $t=T$ 处理论有效）

2. **CO3-resampler（权重和为 0）**:
   - 做什么：在早期高噪声步骤中替换初始噪声
   - 核心思路：将当前 $x_t$ 替换为概念噪声的加权组合，本质上是从概念抑制后的分布中重新采样起始噪声
   - 设计动机：实验显示重采样在高 $t$ 值时效果最好，可抑制概念主导问题

3. **CO3-corrector（权重和为 1）**:
   - 做什么：在中期步骤校正 Tweedie 均值
   - 核心思路：$w_0 > 0$（联合 prompt 权重），$w_1,...,w_K < 0$（单概念为负权重，起抑制作用）。关键是此时组合后仍保持 CFG 的形式 $\tilde{\epsilon}_t^{\tilde{\lambda},C} = \epsilon_t^\phi + \lambda(\sum_k w_k \epsilon_t^{c_k} - \epsilon_t^\phi)$
   - 设计动机：与可组合扩散使用任意 $\lambda_i$ 不同，CO3-corrector 保持了 CFG 的无条件-条件比例，避免产生分布外样本

4. **Closeness-Aware Concept Weight Modulation**:
   - 做什么：自适应调整各概念的抑制权重
   - 核心思路：计算当前噪声预测 $\epsilon^C$ 与各概念噪声 $\epsilon^{c_k}$ 的距离 $d_k$，通过指数核 $a_k = \exp(-\beta d_k)$ 转换为亲和度，归一化后作为负权重 $w_k = -a_k / \sum_j a_j$
   - 设计动机：离当前样本越近的概念应被更强地抑制，实现动态平衡

### 损失函数 / 训练策略

无训练方法。基于 SDXL，50步 DDIM 采样，guidance scale $\lambda=5.0$，校正应用在前 20% 步骤中，前 3 步用 resampler，后续用 corrector。$\beta=0.8$。

## 实验关键数据

### 主实验（两概念 prompt，Attend-Excite 基准）

| 方法 | 无训练 | 无梯度 | 模型无关 | ImageReward (A-A) | ImageReward (A-O) | ImageReward (O-O) |
|------|--------|--------|---------|----------|----------|----------|
| SDXL | ✓ | ✓ | - | 0.782 | 1.547 | 0.679 |
| Attend-Excite | ✓ | ✗ | ✗ | 0.824 | 1.238 | 0.874 |
| InitNO | ✓ | ✗ | ✓ | 1.008 | 1.393 | 1.138 |
| Tweediemix | ✓ | ✓ | ✓ | 1.002 | 1.313 | 0.796 |
| **CO3 (ours)** | ✓ | ✓ | ✓ | **1.234** | **1.674** | **1.016** |

### 消融实验（逐步加入组件）

| 配置 | ImageReward Avg | BLIP-VQA Avg |
|------|----------------|-------------|
| SDXL base | 0.843 | — |
| + Resampling | 0.944 | — |
| + Corrector | 0.946 | — |
| + Weight modulation | **1.012** | — |

### 关键发现

- CO3 在是唯一同时满足无训练、无梯度、模型无关三个属性的高性能方法，但在所有指标上匹敌或超过需要梯度的方法
- 权重和为 1 的约束至关重要——它保持了 CFG 形式，而任意权重的可组合扩散容易产生分布外样本
- Resampler 和 Corrector 发挥互补作用：resampler 在高噪声阶段有效，corrector 在中间阶段生效
- 权重调制贡献显著（Avg 从 0.946→1.012），说明自适应抑制比固定权重更有效
- 在复杂 prompt（T2ICompBench）和稀有概念（RareBench）上也优于专门设计的方法（如 R2F）

## 亮点与洞察

- **理论统一性强**：Proposition 1 将现有的校正方法和可组合扩散统一到 Tweedie 均值组合框架下，揭示了权重约束的关键作用（$\sum w_k = 1$ 保持 CFG 形式 vs $\sum w_k = 0$ 做重采样）。这个统一视角本身就很有价值
- **模式碰撞假说**：用 $p(x|C) / \prod_i p(x|c_i)$ 抑制与单概念重叠的退化模式，直觉清晰且有实验支持。实验结果间接验证了这个假说
- **完全即插即用**：不需要修改模型、不需要梯度、不需要额外训练，可直接应用到任何扩散模型上

## 局限性 / 可改进方向

- 每个去噪步需要对 $K+1$ 个条件分别运行噪声预测，推理开销随概念数线性增长
- 仅在 SDXL 上验证，虽然声称模型无关但未在 DiT 架构（Flux/SD3）上测试
- CO3-resampler 在 $t < T$ 时理论上不严格成立，虽然实验中 $t \approx 0.9T$ 前有效但缺乏严格保证
- 权重调制中的 $\beta$ 参数和校正步数比例需要手动设定
- 概念分解依赖于 prompt 解析，对复杂自然语言 prompt 的分解质量未讨论

## 相关工作与启发

- **vs Attend-Excite**: AE 通过注意力图优化来绑定属性，需要梯度且架构相关；CO3 从模式层面解决问题，理论视角更高
- **vs Composable Diffusion**: 同样是模型无关的分数组合，但可组合扩散用任意权重的线性组合，CO3 证明了权重约束是关键
- **vs Tweediemix**: 同样在 Tweedie 空间操作，但 Tweediemix 需要 LoRA 微调阶段，CO3 纯采样时间校正
- 模式碰撞假说可能启发对扩散模型训练动态的更深入理解

## 评分

- 新颖性: ⭐⭐⭐⭐ 模式碰撞假说新颖，Tweedie 均值组合的统一理论有洞察力
- 实验充分度: ⭐⭐⭐⭐ 多基准（简单/复杂/稀有 prompt），消融完整，人工评估
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，但符号较多，阅读门槛较高
- 价值: ⭐⭐⭐⭐ 即插即用的组合生成改进有很高实用价值
