# NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering

**会议**: ACL 2025  
**arXiv**: [2505.19754](https://arxiv.org/abs/2505.19754)  
**代码**: https://github.com/X-LANCE/NeuSym-RAG  
**领域**: LLM Agent / RAG  
**关键词**: RAG, PDF QA, neural-symbolic retrieval, text-to-SQL, multiview chunking

## 一句话总结
NeuSym-RAG 提出了一个混合神经-符号检索框架，将 PDF 文档通过多视角分块解析同时存入关系数据库和向量库，LLM Agent 通过可执行动作（SQL 查询 + 向量检索 + 查看图片等）迭代式交互检索，在学术论文 QA 上比经典 RAG 提升 17.3%。

## 研究背景与动机

1. **领域现状**：RAG 是 LLM 知识密集型问答的主流方案，但通常仅使用基于向量的神经检索（embedding + 相似度搜索）。另一方面，Text-to-SQL 等符号检索方法适合精确查询但不擅长模糊匹配。
2. **现有痛点**：
   - **神经检索和符号检索被孤立研究**：向量检索擅长语义模糊匹配但难以处理聚合/比较查询（如"论文中共有几个表格"）；符号检索擅长精确查询但遇到同义词变体（如"graph-based RAG" vs "GraphRAG"）就失败
   - **固定长度分块忽略 PDF 结构**：研究论文有丰富的内在结构（章节、表格、图片、公式），简单切分无法利用
   - **现有学术 QA 数据集不够真实**：多数只基于单页或摘要，不涉及完整 PDF 多文档分析
3. **核心矛盾**：单一检索范式无法覆盖真实用户查询的多样性（语义理解 vs 精确计算）
4. **本文要解决什么？** 将神经检索和符号检索统一到一个交互式框架中，充分利用 PDF 的多视角结构信息
5. **切入角度**：将 PDF 解析为关系数据库（精确查询）和向量库（语义匹配）两套并行后端，用 Agent 根据问题类型自适应选择检索方式
6. **核心 idea 一句话**：用 schema-constrained 数据库连接 PDF 的多视角分块和向量编码，LLM Agent 通过 ReAct 框架在两种检索后端之间迭代交互，直到收集足够信息生成答案。

## 方法详解

### 整体框架
PDF 输入 → **Stage 1: Multiview Parsing**（解析为关系数据库 DuckDB）→ **Stage 2: Multimodal Encoding**（可编码列值向量化存入 Milvus 向量库）→ **Stage 3: Iterative Agent Interaction**（Agent 用 5 种动作从 DB/VS 交互检索→生成答案）。

### 关键设计

1. **多视角文档解析（Multiview Document Parsing）**：
   - 做什么：从多个粒度解析 PDF 内容存入关系数据库
   - 核心思路：(1) 查询 arXiv API 获取元数据（作者、会议等）(2) 用 PyMuPDF 按页面、章节、定长三种粒度切分文本 (3) 用 OCR 模型 MinerU 提取表格和图片 (4) 用 LLM/VLM 对各元素生成摘要
   - 设计动机：不同查询需要不同粒度——"这篇论文几个表格"需要表格级别，"某个方法的细节"需要章节级别。多视角 = 多粒度

2. **多模态向量编码（Multimodal Vector Encoding）**：
   - 做什么：将数据库中可编码的列值向量化，建立 DB 和 VS 的一一映射
   - 核心思路：标记 DB schema 中的 varchar 长文本列为"可编码"，用 3 种文本编码器（BM25、MiniLM、BGE）+ 1 种图像编码器（CLIP）分别编码，每个向量附带 (table_name, column_name, primary_key) 三元组用于反查 DB
   - 设计动机：DB schema 作为"骨架"组织 VS 中的向量，不同编码器覆盖不同匹配需求

3. **5 种可执行动作的 Agent 交互**：
   - **RetrieveFromVectorstore**：Agent 重写查询、选择编码模型和视角（表名+列名），支持元数据过滤
   - **RetrieveFromDatabase**：Agent 生成 SQL 查询执行精确检索
   - **ViewImage**：Agent 指定坐标裁剪 PDF 页面图像，送入 VLM 推理
   - **CalculateExpr**：执行 Python 数学表达式，减少数学幻觉
   - **GenerateAnswer**：终止动作，输出最终答案
   - 设计动机：Agent 可自由组合两种检索方式——先 SQL 过滤再向量匹配，或先向量检索再 SQL 精化

### 混合检索协作模式
- **DB → VS**：先用 SQL 选出符合结构化条件的行，提取主键，作为 filter 传入向量检索进行语义匹配
- **VS → DB**：先用向量搜索找到语义相关条目，转换为临时表或 SQL 条件，进一步精确查询
- ReAct 框架：每轮 Agent 输出 thought → action → observation，迭代直到 GenerateAnswer

## 实验关键数据

### 主实验（AirQA-Real 数据集）

| 方法 | Text | Table | Image | Formula | Metadata | AVG |
|------|------|-------|-------|---------|----------|-----|
| Classic-RAG (GPT-4o-mini) | 12.3 | 11.9 | - | - | - | ~25 |
| HybridRAG | - | - | - | - | - | ~30 |
| GraphRAG | - | - | - | - | - | ~28 |
| **NeuSym-RAG (GPT-4o-mini)** | - | - | - | - | - | **~42** |
| NeuSym-RAG vs Classic-RAG | | | | | | **+17.3%** |

跨数据集结果：

| 数据集 | Classic-RAG | NeuSym-RAG | 提升 |
|--------|-----------|-----------|------|
| AirQA-Real | 25.0 | **42.3** | +17.3 |
| M3SciQA | 39.2 | **47.5** | +8.3 |
| SciDQA | 44.1 | **49.8** | +5.7 |

### 消融实验

| 配置 | AirQA-Real AVG |
|------|---------------|
| Full NeuSym-RAG | 42.3 |
| w/o DB (仅向量检索) | 35.1 (-7.2) |
| w/o VS (仅SQL检索) | 30.8 (-11.5) |
| w/o 多视角(单一切分) | 37.6 (-4.7) |
| w/o ViewImage | 39.1 (-3.2) |

### 关键发现
- **混合检索显著优于单一范式**：去掉 DB 或 VS 都有明显下降，证明两者互补
- **多视角分块贡献 4.7%**：说明不同粒度的切分确实有用，不同查询需要不同视角
- **模型规模对 Agent 检索很重要**：GPT-4o > GPT-4o-mini > 开源模型，因为 Agent 需要更强的规划和 SQL 生成能力
- **表格和公式类问题提升最大**：这正是传统向量检索最弱的地方，符号检索带来了精确查询能力

## 亮点与洞察
- **DB schema 作为 VS 的"组织骨架"的设计非常优雅**：每个向量通过 (table, column, pk) 三元组映射回 DB，实现了两套系统的无缝桥接。可迁移到任何结构化文档的 RAG 场景（法律文书、财务报表等）
- **多视角切分是对"one-size-fits-all chunking"的有效改进**：同一文档的页面级、章节级、定长切分各服务不同查询类型，加上表格/图片级切分，覆盖全面
- **Agent 的自适应检索策略**：让 LLM 自己决定何时用 SQL、何时用向量搜索，比硬编码的混合检索更灵活

## 局限性 / 可改进方向
- **依赖 PDF 解析质量**：OCR、表格提取的错误会直接传播到检索结果
- **AirQA-Real 仅标注了 553 个样本**：数据量偏小，统计意义有限
- **Agent 交互轮数增加推理成本**：每轮 ReAct 都需要调用 LLM，多轮交互延迟较高
- **未与最新的 long-context LLM 直接对比**：如 GPT-4o-128k 直接读完整 PDF

## 相关工作与启发
- **vs Classic RAG**：仅向量检索，无法处理精确查询，本文 +17.3%
- **vs GraphRAG (Edge et al., 2024)**：用知识图谱组织信息，但缺少向量语义匹配，适合全局摘要但不擅长细节查询
- **vs HybridRAG (Sarmah et al., 2024)**：简单合并图谱和向量检索，缺少 Agent 的自适应选择机制
- **vs TAG (Biswal et al., 2024)**：纯 Text-to-SQL 方法，在语义模糊匹配上弱

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将神经和符号检索统一到 Agent 交互框架，DB schema 桥接设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 3 个数据集+消融+新标注数据集，但 AirQA-Real 规模偏小
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，动作空间定义详细
- 价值: ⭐⭐⭐⭐ 对学术论文 QA 和结构化文档 RAG 有实用价值
