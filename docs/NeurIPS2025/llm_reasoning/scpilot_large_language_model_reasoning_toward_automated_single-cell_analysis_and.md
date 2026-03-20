# scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery

**会议**: NeurIPS 2025  
**arXiv**: [2602.11609](https://arxiv.org/abs/2602.11609)  
**代码**: https://github.com/maitrix-org/scPilot  
**领域**: LLM推理 / 生物信息学  
**关键词**: single-cell RNA-seq, LLM reasoning, omics-native reasoning, cell-type annotation, trajectory inference

## 一句话总结
提出 scPilot 框架和 scBench 基准，让LLM直接在单细胞RNA-seq数据上进行"组学原生推理"（读取标记基因→提出假设→调用工具验证→迭代修正），实现细胞类型标注准确率提升11%、轨迹推断graph-edit distance降低30%。

## 研究背景与动机

1. **领域现状**：单细胞RNA-seq分析依赖固定pipeline（Scanpy, Seurat），大量隐式人工推理（如差异基因→细胞类型的判断）未被自动化。现有LLM应用仅把LLM当"代码生成器"来调用现有工具。
2. **现有痛点**：(a) 单细胞基础模型（scGPT等）将基因表达嵌入向量空间，缺乏可解释性；(b) LLM代码agent只包装工具默认参数，不做生物学推理；(c) 分析过程的生物学逻辑不透明。
3. **核心矛盾**：单细胞分析需要大量专家推理（从标记基因识别细胞类型，从谱系轨迹推断发育关系），但现有自动化工具不做推理，只做计算。
4. **本文要解决什么？** 让LLM不仅调用工具，还要像生物学家一样解释数据、提出假设、收集证据、迭代修正。
5. **切入角度**：定义"组学原生推理"(ONR)范式——LLM接收单细胞数据的文本摘要，显式推理，调用工具获取数值证据，迭代到得出生物学结论。
6. **核心idea一句话**：将单细胞分析形式化为自然语言推理问题，LLM在每一步产出(声明, 操作)对构成"言语+计算"双轨证明。

## 方法详解

### 整体框架
三个核心组件：(1) **Problem-to-Text Converter** $\mathcal{C}$：将 $10^5$-$10^6$ 细胞的表达矩阵压缩为LLM可消化的文本摘要（如cluster大小、top-k标记基因等）；(2) **Bio-Tool Library** $\mathcal{T}$：封装Scanpy、Monocle、pySCENIC等工具为可调用的结构化API；(3) **LLM Reasoner** $\mathcal{R}_\phi$：以o1/Gemini等推理LLM为核心，执行闭环推理 $\mathbf{X} \to \text{Prompt} \to \{(\text{Thought}_k, \text{Call}_k)\}_{k=1}^K \to \hat{y}$。

### 关键设计

1. **组学原生推理 (ONR) 形式化**:
   - 做什么：将生物信息分析任务定义为推理序列 $\mathcal{R} = [(c_1,o_1), \ldots, (c_K,o_K)]$
   - 核心思路：每步LLM产出自然语言声明 $c_k$（如"cluster 5高表达CD3D和CD3E，可能是T细胞"）和操作 $o_k$（如"检查NK细胞标记基因"），各操作改变数据状态 $S_k = o_k(S_{k-1})$
   - 设计动机：与代码agent的关键区别——推理trace是可审计的生物学论证，不只是代码+输出

2. **Problem-to-Text压缩**:
   - 做什么：将百万级细胞矩阵压缩为LLM上下文窗口内可处理的文本
   - 核心思路：针对不同任务设计不同压缩：细胞标注用Leiden聚类+top-10标记基因；轨迹推断用PAGA图+pseudotime；GRN用top-150 TF-gene对
   - 设计动机：保留生物学显著信息的同时大幅降维，使得LLM可以在文本域操作

3. **scBench 基准**:
   - 做什么：覆盖三大任务（细胞标注、轨迹推断、基因调控网络预测）的9个数据集
   - 核心思路：每个任务有expert-verified ground truth和自动化评测指标（准确率、graph-edit distance、AUROC）
   - 设计动机：现有单细胞基准只评embedding质量或数值指标，不评估推理的生物学意义

### 损失函数 / 训练策略
scPilot是training-free框架，不微调LLM。所有推理能力来自prompt engineering和迭代推理策略。核心设计原则：(a) 生物学上下文优先；(b) 迭代推理；(c) 最小人工启发式。

## 实验关键数据

### 主实验
| 任务 | 数据集 | scPilot (o1) | 最佳baseline | 提升 |
|------|--------|-------------|-------------|------|
| 细胞标注 | PBMC3k | ~0.76 | CellTypist 0.563 | +35% |
| 细胞标注 | Liver | ~0.50 | CellTypist 0.464 | +8% |
| 细胞标注 | Retina | ~0.49 | CellTypist 0.388 | +26% |
| 轨迹推断 | Pancreas | GED降低30% | 传统pipeline | Gemini-2.5-Pro最优 |
| GRN预测 | 多器官 | AUROC提升0.03 | pySCENIC直接输出 | 迭代推理增益 |

### 消融实验
| 配置 | 效果 | 说明 |
|------|------|------|
| Direct prompting (无迭代) | 基线 | 一次性推理 |
| 迭代推理 (2-3轮) | +11% avg accuracy | 迭代修正假设 |
| 无生物学context | 显著下降 | 物种/组织信息关键 |
| 不同LLM比较 | o1最适合标注, Gemini最适合轨迹 | LLM能力有任务特异性 |

### 关键发现
- 迭代推理是关键——LLM在首轮经常犯错（如混淆NK和T细胞），但通过查看额外标记基因在第二轮纠正
- LLM能发现专家标注中的潜在问题——某些情况下scPilot的推理比原始标注更合理
- 不同LLM在不同任务上有优势：o1推理能力强适合标注，Gemini上下文窗口大适合轨迹
- 推理trace具有高度可解释性——生物学家可以审计每一步的逻辑

## 亮点与洞察
- **范式转变**：从"LLM调用工具"到"LLM做生物学推理"。scPilot不只是自动化pipeline，而是自动化专家思维过程
- **推理trace的科学价值**：生成的trace暴露了标记基因歧义性、组织特异性表达模式等，对生物学家有独立的分析价值
- **可推广的ONR框架**：同样的"数据→文本摘要→LLM推理→工具验证"范式可以迁移到蛋白质组学、代谢组学等其他组学领域

## 局限性 / 可改进方向
- 依赖Problem-to-Text压缩的质量——信息丢失可能导致推理偏差
- 当前只覆盖三个核心任务，空间转录组学、多组学整合等未涉及
- 完全依赖LLM的生物学知识，对于非常新的或罕见的细胞类型可能知识不足
- 每次分析需要多轮LLM推理，计算成本较高（o1 API费用）

## 相关工作与启发
- **vs CellAgent**: CellAgent让LLM写代码调用Scanpy，但不做生物学推理；scPilot让LLM解释差异基因并提出假设
- **vs scGPT/Geneformer**: 这些模型在向量空间操作，不产生自然语言推理；scPilot的trace是可审计的论证
- **vs GPTCellType**: 仅做细胞标注一个任务的直接prompting；scPilot覆盖多任务+迭代推理

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "组学原生推理"的形式化定义和系统框架是全新的，开辟了LLM在计算生物学中的新范式
- 实验充分度: ⭐⭐⭐⭐ 9个数据集、3个任务、多个LLM和baseline，但GRN任务的提升幅度有限
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，但数学形式化有些过度符号化
- 价值: ⭐⭐⭐⭐⭐ 对计算生物学社区有变革性影响——展示了LLM作为"科学推理伙伴"而非"代码生成器"的可能性
