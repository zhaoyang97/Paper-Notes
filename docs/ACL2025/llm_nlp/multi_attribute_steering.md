# Multi-Attribute Steering of Language Models via Targeted Intervention

**会议**: ACL 2025  
**arXiv**: [2502.12446](https://arxiv.org/abs/2502.12446)  
**代码**: https://github.com/duykhuongnguyen/MAT-Steer  
**领域**: LLM / NLP  
**关键词**: inference-time intervention, steering vectors, multi-attribute alignment, gating mechanism, representation alignment

## 一句话总结
提出 MAT-Steer，通过属性感知的 token 级 gating 机制和正交性约束，实现推理时对 LLM 多属性（如真实性、毒性、偏见）的同时精准干预，在 QA 和生成任务上全面超越现有 ITI 和微调方法。

## 研究背景与动机
1. **领域现状**：推理时干预（ITI）通过在模型中间层加入 steering vector 来调整 LLM 行为，无需更新参数，成本低且避免灾难性遗忘。
2. **现有痛点**：现有 ITI 方法（如 Li et al. 2024, LITO）对所有 token 均匀施加同一干预向量，在多属性场景下会产生属性间冲突——例如同时提升 helpfulness 和降低 bias 时，一个方向的干预可能恶化另一个属性。
3. **核心矛盾**：多属性 steering vector 之间存在方向冲突，均匀干预导致过度校正（overcorrection），且无法区分哪些 token 与哪个属性相关。
4. **本文要解决什么**：(1) 如何在 token 级别精确判断对哪些 token 施加哪个属性的干预？(2) 如何避免多个 steering vector 之间的冲突？
5. **切入角度**：观察到不同 token 与不同属性的相关性差异很大（如 "harmed" 与 bias 相关，"the" 与任何属性都无关），因此应该选择性地、按需干预。
6. **核心 idea 一句话**：用属性特定的 gating function 实现 token 级选择性干预，配合 MMD 表示对齐 + 稀疏性 + 正交性约束，解决多属性 steering 的冲突问题。

## 方法详解

### 整体框架
输入为 LLM 的中间层 activation，对每个 token 的激活向量，通过 T 个属性特定的 gating function 判断该 token 是否需要针对每个属性进行干预，然后加权叠加 T 个 steering vector 完成编辑。训练阶段使用正/负样本对的激活分布，通过 MMD loss 对齐 + 正交性/稀疏性正则化学习 steering vector 和 gating 参数。

### 关键设计

1. **属性感知 Gating Function**:
   - 做什么：为每个属性 $t$ 学习一个 gating function $G_t(a_i) = \sigma(w_t a_i + b_t)$，输出 0-1 标量表示对该 token 激活的干预强度
   - 核心思路：整体 steering function 为 $f(a_i | \theta_1,...,\theta_T) = a_i + \sum_{t=1}^T G_t(a_i) \theta_t$，每个属性的 steering vector $\theta_t$ 乘以其 gating 权重后叠加
   - 设计动机：不同于均匀干预，gating 让模型只在相关 token 上施加干预，避免对已对齐的 token 过度校正。例如毒性相关词 gating weight 高，中性词 gating weight 接近 0

2. **MMD 表示对齐 (Representation Alignment)**:
   - 做什么：通过 Maximum Mean Discrepancy (MMD) loss 将编辑后的负样本激活分布对齐到正样本激活分布
   - 核心思路：$\mathcal{L}_{MMD} = \sum_{t=1}^T \| \frac{1}{|\mathcal{A}_t^p|}\sum \phi(a_i^p) - \frac{1}{|\mathcal{A}_t^n|}\sum \phi(f(a_i^n)) \|_{\mathcal{H}}^2$，使用 RKHS 映射捕获高阶分布差异
   - 设计动机：相比之前 ITI 工作只匹配均值，MMD 能捕获方差等高阶矩信息，且不需要配对数据（可以处理非配对的正负样本）

3. **冲突避免正则化 (Conflict Avoidance)**:
   - 做什么：三项正则化确保多属性干预不冲突
   - 核心思路：
     - 正样本保护 $\mathcal{L}_{pos}$：对正样本激活的 gating 权重施加 L2 惩罚，确保已对齐样本不被干预
     - 稀疏性约束 $\mathcal{L}_{sparse}$：对负样本激活的 gating 权重施加 L1 惩罚，确保只有最相关的属性向量被激活
     - 正交性约束 $\mathcal{L}_{ortho}$：惩罚不同属性 steering vector 间的余弦相似度，使它们在不同方向操作避免相互干扰
   - 设计动机：LLM 激活空间维度很高（d=4096），因此有足够空间让多个属性向量保持正交

### 损失函数 / 训练策略
总损失 $\mathcal{L} = \mathcal{L}_{MMD} + \lambda_1 \mathcal{L}_{pos} + \lambda_2 \mathcal{L}_{sparse} + \lambda_3 \mathcal{L}_{ortho}$。编辑后的激活还会做归一化，保持与原始激活相同的幅度（防止干预改变激活的 scale）。训练数据来自各属性对应的正负样本对（如 TruthfulQA/Toxigen/BBQ），分别提取中间层激活。

## 实验关键数据

### 主实验

| 方法 | TruthfulQA | Toxigen | BBQ |
|------|-----------|---------|-----|
| Llama-3.1-8B (base) | 49.91 | 48.10 | 51.77 |
| ICL | 55.32 | 51.26 | 56.46 |
| SFT | 54.02 | 55.51 | 57.29 |
| DPO | 56.10 | 55.94 | 57.51 |
| LITO (best ITI baseline) | 58.63 | 54.08 | 58.14 |
| **MAT-Steer** | **61.94** | **57.59** | **60.32** |

MAT-Steer 在三个数据集上全面超越所有基线，比最强 ITI 基线 LITO 分别高 3.31%、3.51%、2.18%。

### 消融实验

| 配置 | TruthfulQA |
|------|-----------|
| Base model | 49.91 |
| + Alignment (MMD) | 53.82 |
| + Alignment + Pos preservation | 55.48 |
| + Alignment + Sparse | 56.73 |
| + Alignment + Orthogonality | 54.37 |
| MAT-Steer w/o Pos | 57.37 |
| MAT-Steer w/o Sparse | 58.08 |
| MAT-Steer w/o Orth | 59.69 |
| MAT-Steer w/o Normalization | 59.88 |
| **MAT-Steer (full)** | **61.94** |

### 关键发现
- 每个组件都有贡献，稀疏性约束和正交性约束分别贡献约 3.86% 和 2.25% 的提升
- 归一化很关键（+2.06%），防止干预后激活的 scale 偏移
- 在 ParaDetox 毒性分析中，MAT-Steer 对有毒样本的 gating weight 为 0.61 而无关属性为 0.14，证明选择性干预有效
- 仅需不到 20% 的训练数据即可达到 SFT/DPO 全量数据的性能
- 方法可泛化到不同模型（Qwen2.5-7B, Llama-3.1-8B-Chat）和不同任务（HH-RLHF, FaithEval, OBQA）

## 亮点与洞察
- **token 级选择性干预**是核心创新：不同于对所有 token 均匀"加偏移"，MAT-Steer 只在需要的 token 上操作，大幅减少副作用。这个思路可以迁移到任何需要精细控制模型行为的场景。
- **正交性约束**将多属性冲突问题转化为高维空间中的方向分离问题，利用了 LLM 激活空间的高维特性（d=4096 >> T），非常优雅。
- **MMD loss 替代 MSE**捕获分布级别的对齐，比点对点匹配更鲁棒，对无配对数据场景尤其友好。
- **极高的数据效率**：10% 数据就超过 SFT/DPO 100% 数据，说明 steering vector 方法在数据受限时极具优势。

## 局限性 / 可改进方向
- 当前 gating function 是简单的线性+sigmoid，可能在复杂场景下表达力不足，可以尝试更复杂的 gating（如 MLP 或 attention-based gating）
- 正交性是软约束，属性数量增加时是否仍能保持性能有待验证
- 实验主要在 7-8B 模型上进行，更大模型（70B+）上的效果未知
- steering vector 在不同层的效果可能不同，目前选择固定层，可以探索多层联合干预

## 相关工作与启发
- **vs LITO (Bayat et al. 2024)**：LITO 是之前最强的 ITI 方法，但对所有 token 均匀干预，无法处理属性冲突。MAT-Steer 通过 gating + 正交性全面超越
- **vs SFT/DPO 微调**：微调需要更多数据且有灾难性遗忘风险，MAT-Steer 作为推理时方法无需修改模型参数，且可与微调方法叠加使用
- **vs ICV (Liu et al. 2024b)**：ICV 也用均值方向作为 steering vector，但没有 token 级 gating 和多属性冲突处理

## 评分
- 新颖性: ⭐⭐⭐⭐ token 级 gating + 多属性正交性约束的组合有新意，但单个组件并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 三个 QA 数据集 + 生成任务 + 多模型 + 详细消融 + 泛化实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观，方法叙述完整
- 价值: ⭐⭐⭐⭐ 推理时多属性对齐是实际需求，方法实用且高效
