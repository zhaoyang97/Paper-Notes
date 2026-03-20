# SwingArena: Adversarial Programming Arena for Long-context GitHub Issue Solving

**会议**: ICLR 2026 / **arXiv**: [2505.23932](https://arxiv.org/abs/2505.23932)  
**代码**: 提供Project Page、Code、Dataset链接  
**领域**: 代码评测 / LLM Benchmark / 软件工程  
**关键词**: 对抗性评测, CI流水线, Submitter-Reviewer, 检索增强代码生成(RACG), 多语言代码  

## 一句话总结
提出SwingArena对抗性评测框架，让LLM交替扮演提交者(生成补丁)和审查者(编写测试)，通过真实CI流水线验证，覆盖C++/Python/Rust/Go四种语言的400个GitHub issue，揭示不同模型在补丁生成vs验证方面的行为差异。

## 研究背景与动机
1. **现有代码基准过于静态**：HumanEval/MBPP仅评估短代码片段，SWE-Bench限于Python且只用单元测试，都无法捕捉真实软件开发的迭代协作特性
2. **忽视完整CI流水线**：真实PR需通过编译、linting、风格检查、回归测试等完整流程，现有基准仅问"代码能否通过单元测试"
3. **缺乏对抗性交互**：真实开发中reviewer会主动构造corner case挑战补丁质量，单agent静态评测无法暴露这类问题
4. **长上下文代码检索挑战**：大型代码库中相关信息分散在千行代码和多文件中，需要有效的检索策略
5. **多语言支持不足**：大多数基准仅支持Python，忽略了C++、Rust、Go等主流语言

## 方法详解

### 对抗性Battle协议
- **角色设计**：两个LLM分别扮演Submitter（生成补丁修复issue）和Reviewer（编写测试用例挑战补丁）
- **评分机制**：Submitter补丁通过所有测试(含Reviewer的)得+1，否则-1；Reviewer的测试若能暴露补丁缺陷得+1，若自身测试不能通过golden patch则-1
- **角色交换**：共10轮battle，每个agent各做5轮submitter和5轮reviewer
- **质量门控**：Reviewer测试必须通过golden patch、不能修改生产代码、限制行数、禁止非确定性、符合linting

### RACG：检索增强代码生成
- **FileRetriever**：BM25稀疏检索，从问题描述到文件级粗排，取top-k相关文件
- **CodeChunker**：语法感知分块，将代码分解为函数/类/块等语义单元，支持C++/Python/Rust/Go
- **CodeReranker**：CodeBERT稠密向量计算cosine相似度，加入语言感知打分、邻近性偏置、去重
- **Token Budget管理**：动态token预算分配，根据剩余窗口自适应选择粗/细粒度chunk

### 数据集构建
- GitHub API挖掘高star仓库 → 提取PR+Issue对 → CI测试过滤 → LLM-as-Judge评估清晰度/难度 → 人工专家校验
- 最终：2300个(issue, PR)对，400个评测实例(每语言100个)，100个消融子集

### 验证与复现
- 每个仓库的CI环境在隔离Docker容器中复现
- temperature=0保证确定性输出，固定随机种子

## 实验

### 主实验：对抗性Battle结果（闭源模型）
| 对战 | RPR | SPR | Win Rate |
|------|-----|-----|----------|
| GPT-4o vs GPT-4o | 0.71 | 0.68 | 0.97 |
| Claude vs Claude | 0.62 | 0.62 | 1.00 |
| Gemini vs Gemini | 0.72 | 0.63 | 0.91 |
| DeepSeek vs DeepSeek | 0.70 | 0.66 | 0.96 |
| GPT-4o vs Claude(审) | 0.65 | 0.55 | 0.90 |

### 多语言Best@3
| 模型 | 平均 | C++ | Go | Rust | Python |
|------|------|-----|-----|------|--------|
| DeepSeek | 0.59 | 0.64 | 0.61 | 0.58 | 0.52 |
| Gemini | 0.57 | 0.64 | 0.58 | 0.51 | 0.57 |
| GPT-4o | 0.57 | 0.63 | 0.53 | 0.56 | 0.54 |
| Claude | 0.55 | 0.63 | 0.55 | 0.52 | 0.50 |

### RACG消融
- RACG在C++上将Best@3从0.38提升到0.42，Win Rate从0.77到0.84
- Top-20检索+重排序为最强baseline(Best@3=0.43, Win Rate=0.73)

### 关键发现
- **GPT-4o擅长激进补丁生成**（Win Rate≥0.90），但整体正确性(SPR)偏低
- **DeepSeek和Gemini侧重正确性和CI稳定性**（SPR/RPR更高）
- 所有模型在C++上表现最好，Rust和Python相对较弱
- 对抗性battle的结果存在轻微非对称性，审查者模型会微妙影响结果

## 亮点
- 首个将完整CI流水线+对抗性角色交换引入LLM代码评测的框架
- 多语言支持(C++/Python/Rust/Go)填补了重要空白
- 数据构建流程严谨：GitHub挖掘→CI过滤→LLM评估→人工校验
- 揭示出模型在"攻击性补丁生成"vs"防御性质量保证"上的有趣行为差异

## 局限性
- RACG的Top-5文件检索限制可能成为复杂issue的瓶颈
- 评测成本高（每对battle需多轮CI执行）
- 仅评估了4种语言，未覆盖Java/TypeScript等主流语言
- 400个评测实例规模可能不足以消除统计噪声
- 开源模型评测仅做了补充(Qwen2.5-Coder-7B)，未深入分析

## 相关工作
- **代码基准**：HumanEval、MBPP、SWE-Bench(Python-only)、多语言扩展(Docker手动配置)
- **代码评测**：函数级正确性评测 → 本文推到repo级CI评测
- **RAG for Code**：BM25检索+AST解析 → 本文RACG跨语言+token预算管理
- 本文独特贡献是对抗性交互+真实CI验证的结合

## 评分 ⭐⭐⭐⭐
- **新颖性**: 5/5 — 对抗性CI评测框架具有开创性
- **实验充分度**: 4/5 — 覆盖多模型多语言，但开源模型评测不够深入
- **写作质量**: 4/5 — 框架描述清晰，实验呈现良好
- **价值**: 4/5 — 为代码LLM评测提供更接近真实场景的范式
