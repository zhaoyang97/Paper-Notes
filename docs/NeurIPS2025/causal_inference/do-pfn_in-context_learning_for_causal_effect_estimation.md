<!-- 由 src/gen_stubs.py 自动生成 -->
# Do-PFN: In-Context Learning for Causal Effect Estimation

**会议**: NeurIPS 2025  
**arXiv**: [2506.06039](https://arxiv.org/abs/2506.06039)  
**代码**: https://github.com/jr2021/Do-PFN (有)  
**领域**: 因果推断 / 基础模型  
**关键词**: Causal Effect Estimation, PFN, In-Context Learning, SCM, CATE, Amortized Inference

## 一句话总结
提出 Do-PFN，将 Prior-data Fitted Networks (PFN) 扩展到因果效应估计，在大量合成 SCM 数据上预训练 Transformer 进行 in-context 因果推理，仅需观测数据即可预测干预分布（CID）和 CATE，无需因果图知识或不混杂假设，在合成和半合成实验中表现出色。

## 研究背景与动机

1. **领域现状**：因果效应估计是科学核心任务。RCT 是金标准但常不可行。从观测数据估计因果效应通常需要不混杂假设（unconfoundedness），但该假设难以验证。TabPFN 已在表格 ML 中展示了 in-context learning 的惊人效果。
2. **现有痛点**：(a) 现有方法依赖因果图知识或不混杂假设；(b) 元学习者（T-/S-/X-learner）在不混杂假设不满足时失效；(c) 深度学习方法（DragonNet/TARNet）同样依赖该假设。
3. **核心矛盾**：能否通过大规模预训练让模型"meta-learn"出因果推理能力，从而无需显式的因果图或不混杂假设？
4. **切入角度**：受 TabPFN 启发——如果预训练在合成因果数据上（包括干预），模型可以学到从观测数据预测干预结果的能力。
5. **核心 idea**：在百万级 SCM 上预训练 Transformer，输入完整观测数据集+干预查询，输出条件干预分布 $p(y|do(t),\mathbf{x})$。

## 方法详解

### 整体框架
预训练阶段：采样 SCM → 生成观测数据 $\mathcal{D}^{ob}$ + 干预数据 $\mathcal{D}^{in}$ → 训练 Transformer 预测 $y^{in}$ given $(t^{in}, \mathbf{x}^{in}, \mathcal{D}^{ob})$  
推理阶段：给定真实观测数据 + 干预查询 → Do-PFN 直接输出 CID

### 关键设计

1. **SCM Prior 设计**:
   - 采样多样的 DAG 结构（4-60 节点）、非线性函数、噪声分布
   - 同时生成观测数据和干预数据对
   - 先验覆盖可识别和不可识别的因果场景

2. **Proposition 1（理论保证）**:
   - 证明 Algorithm 1 的 SGD 等价于最小化 CID 与模型预测分布之间的 forward KL 散度
   - 意味着模型学到的是 CID 的最优近似

3. **三类不确定性分解**:
   - 随机不确定性：SCM 噪声项导致的
   - 不可识别性不确定性：观测等价的 SCM 间的
   - 认知不确定性：有限数据导致的（随数据量增加消失）

4. **一致性保证**:
   - 当 $|\mathcal{D}^{ob}| \to \infty$ 时，后验分布收敛到马尔可夫等价类

### 损失函数 / 训练策略
- 负对数似然 $-\log q_\theta(y^{in}|do(t^{in}), \mathbf{x}^{in}, \mathcal{D}^{ob})$
- 7.3M 参数 Transformer，在单 RTX 2080 上训练 48-96 小时
- Bar distribution 参数化输出

## 实验关键数据

### 主实验 — CID/CATE/ATE 估计
| 方法 | CID (MSE↓) | CATE (MSE↓) | 图知识需求 |
|------|-----------|-------------|-----------|
| Do-PFN | **最优** | **最优** | 无 |
| TabPFN v2 | 差 | 差 | 无 |
| Causal Forest | 中 | 中 | 需要不混杂 |
| DragonNet | 中 | 中 | 需要不混杂 |
| DoWhy (Graph) | 参考标准 | 参考标准 | 需要因果图 |

### 消融实验
| 配置 | 关键发现 |
|------|---------|
| Dont-PFN (仅观测预训练) | 远劣于 Do-PFN，证明干预预训练学到了不同于回归的能力 |
| Do-PFN-Graph (给图信息) | 与不给图信息的 Do-PFN 性能接近，说明模型自动学会了调整 |
| 不满足不混杂假设 | Do-PFN 稳健，baseline 方法性能下降 |
| 大图 (21-50 节点) | v1 性能下降，v1.1（扩展预训练）恢复 |

### 关键发现
- Do-PFN 自动执行前门/后门调整，无需图知识
- 在 RealCause 基准上与专门 CATE 估计器竞争力强
- 不确定性校准良好，尤其在不可识别场景中不确定性正确增加

## 亮点与洞察
- **因果推理的 foundation model 路线**：将 TabPFN 的 in-context learning 成功扩展到因果推断，开辟了 amortized causal inference 新方向。
- **无需因果图和不混杂假设**：这是一个重大突破——大多数因果效应估计方法至少需要其中之一。
- **三类不确定性的漂亮分解**（公式 4）：随机、不可识别性、认知三者的来源和消除条件清晰。
- **Dont-PFN 对照实验非常有说服力**：证明干预预训练确实学到了因果能力而非仅是回归。

## 局限性 / 可改进方向
- **仅处理二元处理变量**：连续处理和多值处理未覆盖
- **依赖 SCM prior 的覆盖度**：如果真实数据生成过程超出先验范围，性能可能下降
- **7.3M 参数较小**：更大模型+更多预训练数据可能进一步提升
- **可改进**：扩展到连续处理；多处理变量联合估计；与 LLM 结合增强先验

## 相关工作与启发
- **vs 元学习者 (T/S/X-learner)**: 需要不混杂假设，Do-PFN 不需要
- **vs DoWhy**: DoWhy 需要因果图，Do-PFN 不需要
- **vs TabPFN**: TabPFN 做预测，Do-PFN 做因果推断——从"条件"到"干预"的关键跨越

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将 PFN 扩展到因果推断是全新方向
- 实验充分度: ⭐⭐⭐⭐⭐ 合成+半合成+RealCause+OOD分析+校准分析
- 写作质量: ⭐⭐⭐⭐⭐ 理论严谨，实验设计巧妙
- 价值: ⭐⭐⭐⭐⭐ 开辟因果推断基础模型新方向
