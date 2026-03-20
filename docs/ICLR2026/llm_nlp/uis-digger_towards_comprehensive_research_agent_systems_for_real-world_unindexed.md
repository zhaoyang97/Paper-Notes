# UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking

**会议**: ICLR2026  
**arXiv**: [2603.08117](https://arxiv.org/abs/2603.08117)  
**代码**: [HuggingFace](https://huggingface.co/datasets/UIS-Digger/UIS-QA)  
**领域**: llm_agent / information seeking  
**关键词**: 未索引信息, 深度浏览, 多Agent框架, 双模式浏览器, SFT+RFT  

## 一句话总结
识别并形式化"未索引信息检索"(UIS) 问题，提出首个 UIS 基准 UIS-QA (110 题) 和多 Agent 框架 UIS-Digger，用 ~30B 模型超越集成 O3/GPT-4.1 的系统。

## 背景与动机
1. 现有信息检索 Agent 严重依赖搜索引擎索引知识，但互联网上大量关键信息未被索引
2. 未索引信息包括：深层网页、动态交互内容、嵌入文件（PDF/XLSX）、需多步导航才能到达的页面
3. 现有基准 (GAIA, BrowseComp) 未区分索引与未索引信息，高分不代表真实能力
4. SOTA Agent 在 GAIA 上 70.90%，BrowseComp-zh 上 46.70%，但 UIS-QA 上骤降至 24.55%
5. 失败原因：(1) 动作空间不足——搜索型 Agent 无法执行浏览交互；(2) 基础模型能力限制
6. UIS 是信息检索 Agent 评估的根本盲区，需要专门的基准和方法

## 方法详解
**UIS-QA 基准**：
- 110 个专家标注 QA 对，跨政府公告、产品介绍、代码仓库、游戏、年报等领域
- 严格筛选：3 名标注员 Google 搜索验证 + z.ai 自动验证 + 离线 LLM 过滤
- 要求客观性、权威性、时间稳定性、可验证性、无需登录

**UIS-Digger 多 Agent 系统**（4 个 Agent）：
- **Planner**：任务分解 + 协调下属 Agent + 输出最终答案
- **Web Searcher**：搜索引擎 + 爬虫获取索引信息，可委派子任务
- **Web Surfer**：双模式浏览器（文本模式 + 视觉截图模式），共享记忆和浏览器状态
  - 动作空间：点击、滚动、输入、选择、导航、提交、下载、截图等
- **File Reader**：处理 PDF/XLSX/DOCX，超长文件分块读取

**训练策略**（两阶段）：
- 数据构造：100+ 真实网站 + 虚拟交互网站（模拟日期选择器、筛选器等）
- SFT：教师模型生成轨迹 + reject sampling（正确 + 非平凡）
- RFT：SFT 模型自我采样 (temp=0.4, group=4) + 难度加权 reject sampling

## 实验关键数据
| 系统 | UIS-QA | GAIA | BrowseComp-zh |
|------|--------|------|---------------|
| GPT-5 直接推理 | 0.9% | - | - |
| WebSailor (32B) | 7.3% | 53.2% | 25.5% |
| OWL (GPT-4.1) | 25.45% | 70.90% | 46.70% |
| **UIS-Digger (Qwen3-32B)** | **27.27%** | - | - |

- SOTA Agent 在 UIS-QA 上剧烈性能下降（GAIA 70% → UIS-QA 25%）
- ~30B 参数模型通过 SFT+RFT 超越使用 O3/GPT-4.1 的系统
- 视觉模式 + 文件读取是解决 UIS 的关键工具差异

## 亮点
- **首次形式化 UIS 问题**：区分索引 vs 未索引信息，揭示现有评估的根本盲区
- 双模式浏览策略（文本+视觉）共享记忆，兼顾效率和功能完备性
- 小模型 (~30B) + 专项训练 > 大模型 (O3/GPT-4.1) 通用能力，说明 UIS 需要专门优化
- UIS-QA 基准设计严谨（三重过滤 + 稳定性 + 可验证性）

## 局限性 / 可改进方向
- UIS-QA 仅 110 题，规模偏小
- 84/110 题为中文，英文覆盖有限
- 绝对准确率仍仅 27.27%，UIS 问题远未解决
- 未考虑需要登录或 CAPTCHA 验证的场景

## 与相关工作的对比
- vs GAIA/BrowseComp：这些基准不区分 UIS，高分可能只反映搜索能力
- vs WebArena/Mind2Web：这些聚焦浏览器操作但在受控环境中，UIS-QA 在真实网络中评估
- vs ReAct Agent：UIS-Digger 的多 Agent 架构提供更丰富的动作空间

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ UIS 问题的识别和形式化是重要贡献
- 实验充分度: ⭐⭐⭐⭐ 多种基线对比，但基准规模略小
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，形式化完整
- 价值: ⭐⭐⭐⭐⭐ 揭示信息检索 Agent 的关键盲区
