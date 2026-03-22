# AMS-IO-Bench and AMS-IO-Agent: Benchmarking and Structured Reasoning for Analog and Mixed-Signal Integrated Circuit Input/Output Design

**会议**: AAAI 2026  
**arXiv**: [2512.21613v1](https://arxiv.org/abs/2512.21613v1)  
**代码**: [https://github.com/Arcadia-1/AMS-IO-Agent](https://github.com/Arcadia-1/AMS-IO-Agent) / [https://github.com/Arcadia-1/AMS-IO-Bench](https://github.com/Arcadia-1/AMS-IO-Bench)  
**领域**: LLM Agent / EDA自动化 / 模拟与混合信号IC设计  
**关键词**: LLM Agent, AMS IC, I/O Ring, EDA自动化, 结构化推理  

## 一句话总结
提出AMS-IO-Agent，一个基于LLM的领域专用智能体，通过结构化意图图(Intent Graph)和领域知识库将自然语言设计意图转化为可生产的模拟混合信号IC I/O环设计，配套提出首个AMS I/O环自动化基准AMS-IO-Bench，在28nm CMOS流片中验证了智能体生成的I/O环可直接用于实际芯片制造。

## 背景与动机
模拟与混合信号(AMS) IC的输入/输出(I/O)子系统负责信号接口、供电和ESD保护，是芯片的基础组件。与数字I/O可通过脚本自动化不同，AMS I/O设计因涉及多种信号类型、多电源域、功率完整性约束、敏感模拟信号布线等复杂需求，至今仍高度依赖人工。一个新手工程师可能需要花1-2天来手动组装和验证I/O的摆放与连接，而迭代改引脚更是工作量巨大。

现有LLM在IC设计方面的应用（RTL代码生成、网表合成等）没有覆盖AMS I/O这一领域，原因有三：(1) 领域知识碎片化，分散在团队内部文档中；(2) 缺乏标准化任务接口，设计师通过GUI或领域专用语言交互，LLM难以直接介入；(3) 缺乏公开基准。

## 核心问题
如何让LLM理解非结构化的引脚规划描述（自然语言、表格等），并自动生成满足DRC(设计规则检查)和LVS(版图与原理图一致性检查)的、可直接用于流片的AMS I/O环设计？核心挑战在于：LLM无法直接生成质量可靠的EDA脚本（如Cadence SKILL），简单的端到端代码生成在这种约束密集、领域专用的任务上完全失败。

## 方法详解

### 整体框架
AMS-IO-Agent采用三层分层架构：用户提供引脚规划文本 → LLM将其结构化为Intent Graph(意图图) → Intent Graph Adaptor(适配器)将意图图解析为可执行的EDA脚本(SKILL/csh) → 在Cadence Virtuoso中生成原理图和版图 → 调用Calibre进行DRC/LVS验证。整个过程将高层意图推理与底层实现分离，LLM专注于理解设计意图，确定性模块负责约束求解和脚本生成。

### 关键设计
1. **设计意图结构化(Design Intent Structuring)**: 将非结构化的引脚规划（自然语言描述、引脚列表、半结构化表格）转换为标准化的Intent Graph——一种JSON格式的图表示。每个节点代表一个pad或corner cell，包含名称、器件类型、空间位置、方向、引脚连接等属性。构建过程分两步：**显式补全**——根据引脚名命名规范（如DCLK=数字时钟、VCM=共模电压）通过知识库推断信号类型、器件类型、默认方向等属性；**隐式推理**——自动插入设计规范中未提及但必需的元素如corner cell。Intent Graph不同于网表：网表只描述电路连接，而Intent Graph还捕获空间关系、语义上下文和领域知识，可同时被人和LLM理解，也能被代码高效解析。

2. **意图图适配器(Intent Graph Adaptor)**: 作为LLM与商业EDA工具之间的中间件层。为什么不让LLM直接生成SKILL脚本？因为SKILL训练数据极度匮乏，LLM无法可靠生成，同时SKILL本身也不适合复杂数据操作。适配器用Python实现确定性处理：结构化数据解析、约束求解、几何计算（如根据I/O设计规则精确计算每个cell的坐标）、SKILL脚本生成（用于Cadence Virtuoso创建原理图和版图）、csh脚本生成（用于调用Calibre验证工具）。这些工具被封装为可复用的工具库，供Agent按需调用，而非每次由LLM重新生成。

3. **领域专用知识库(Domain-Specific Knowledge Base)**: 从一个10+人的专业AMS IC设计团队的培训材料和设计实践中提炼而来，经过50+次流片验证。涵盖器件选型实践、版图惯例、电源域规则、ESD保护要求、命名规范等。整个知识库仅约6k tokens，足够紧凑到可以直接作为LLM的上下文输入，无需RAG检索或模型微调。这些知识和培训新工程师用的材料相同，本科背景的工程师1-3天即可掌握。

### 训练策略
本文方法不需要训练或微调LLM，而是通过prompt engineering将知识库注入上下文，结合结构化推理流程实现零样本设计自动化。使用smolagents框架，通过API调用GPT-4o/Claude-3.7/DeepSeek-V3等骨干模型。

## 实验关键数据

### AMS-IO-Bench基准
从10个真实流片项目中构建30个测试用例，分三个难度级别：
- **简单**(10例)：单信号域，小规模
- **中等**(10例)：多电源域(数字+模拟)，标准MPW芯片~1mm×1mm，需要跨域推理
- **困难**(10例)：双排/交错pad、放大chip outline、定制I/O cell、专用ESD供电等

### 五项评估指标
Intent Graph正确率 → Shape Score(VLM视觉评估) → DRC通过率 → LVS通过率 → DRC+LVS通过率

| 方法 | IG(%) | Shape(%) | DRC(%) | LVS(%) | DRC+LVS(%) | 时间(min) | Token(k) |
|------|-------|----------|--------|--------|------------|-----------|----------|
| 人工设计 | 100 | 100 | 100 | 100 | 100 | ~480 | – |
| 直接LLM(GPT-4o) | 0 | 0 | 0 | 0 | 0 | 0.2 | 1k |
| AMS-IO-Agent(GPT-4o) | 100 | 100 | 76.67 | 66.67 | 63.33 | 4.1 | 160k |
| AMS-IO-Agent(Claude-3.7) | 100 | 100 | 93.33 | 76.67 | 76.67 | 4.2 | 96k |
| AMS-IO-Agent(DeepSeek-V3) | 100 | 100 | 93.33 | 76.67 | 76.67 | 5.1 | 105k |

### 按难度级别DRC+LVS通过情况
| 模型 | 简单 | 中等 | 困难 |
|------|------|------|------|
| GPT-4o | 10/10 | 7/10 | 2/10 |
| Claude-3.7 | 10/10 | 9/10 | 4/10 |
| DeepSeek-V3 | 10/10 | 10/10 | 3/10 |

### 消融实验要点
- 仅用知识库(KB)不用Intent Graph和Adaptor → 0% DRC+LVS，LLM直接生成SKILL完全失败
- KB + Adaptor但不用Intent Graph(LLM直接生成Python代码) → Shape 100%但DRC仅20%，LVS 0%
- KB + Intent Graph但不用Adaptor → Intent Graph 100%但无法生成可用版图
- **三个组件缺一不可**：完整配置(KB+IG+Adaptor)达到93.33% DRC、76.67% DRC+LVS
- 直接LLM生成代码和结构化意图推理的差距极大，证明了中间表示的必要性

## 亮点
- **首次LLM Agent在真实芯片流片中发挥实质作用**：28nm CMOS流片验证，Agent生成的I/O环直接用于硅片制造并通过功能验证，这是首个LLM Agent完成非平凡AMS IC设计子任务并直接用于流片的报道
- **结构化中间表示的精妙设计**：Intent Graph同时对人和机器友好——人能看懂、LLM能生成、代码能解析，解决了LLM不擅长直接生成EDA DSL的根本问题
- **工程效率提升巨大**：从人工设计约480分钟/案例降至~5分钟，token消耗控制在100k量级
- **可迁移的Agent架构思路**：将高层意图理解(LLM擅长)与底层确定性执行(工具擅长)分离的架构，可迁移到其他需要LLM与专业工具协作的领域

## 局限性 / 可改进方向
- 仅覆盖wirebond封装的AMS I/O环，未涉及flip-chip等其他封装方式
- 知识库绑定特定设计惯例和工艺节点，迁移到其他foundry规则需要替换知识库
- 最高76.67%的DRC+LVS通过率说明仍有约1/4的设计需要人工修正，尤其是Hard级别案例
- 30个测试用例的基准规模偏小（但作者论证了AMS IC领域的数据量限制是合理的）
- 未探索更复杂的AMS设计任务（如完整版图布线、模拟电路优化等）

## 与相关工作的对比
- **vs. 现有EDA自动化工具**（如Yosys padring工具）：现有工具需要设计师编写详细配置表，自动化程度低，且基本面向数字SoC不支持AMS需求。本文通过自然语言接口和领域知识推理，真正实现了"理解设计意图"级别的自动化
- **vs. LLM for IC Design的其他工作**（如ChipNeMo、RTL代码生成等）：现有工作集中在数字设计（RTL生成、网表合成），未触及AMS I/O这一高度手工化的环节。本文首次将LLM Agent引入AMS IC领域并实现流片级验证
- **vs. 直接LLM代码生成**：消融实验表明直接让LLM生成SKILL代码完全失败(0% pass rate)，证明了结构化意图中间表示的必要性

## 启发与关联
- "LLM做高层推理 + 确定性工具做底层执行"的Agent架构模式值得关注，适用于任何需要将自然语言意图转化为精确技术产出的场景
- Intent Graph作为人机共读的中间表示的设计思路，可迁移到其他工程自动化领域
- 知识库的compact设计（~6k tokens直接in-context）避免了RAG的复杂性，在专业领域知识规模有限时是更实用的方案

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将LLM Agent引入AMS IC I/O设计并完成流片验证，问题定义新颖但Agent架构本身并不特别创新
- 实验充分度: ⭐⭐⭐⭐ 消融实验设计合理，有真实28nm流片case study验证，但基准规模偏小(30例)
- 写作质量: ⭐⭐⭐⭐ 问题动机、方法设计和实验分析都讲得清楚，图表质量高
- 价值: ⭐⭐⭐⭐ 对AMS IC设计自动化有重要推动，但应用范围较窄，主要面向IC设计从业者
