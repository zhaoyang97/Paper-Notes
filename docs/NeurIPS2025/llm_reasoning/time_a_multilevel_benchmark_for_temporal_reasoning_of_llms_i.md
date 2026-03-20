# TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios

## 基本信息
- **arXiv**: 2505.12891
- **会议**: NeurIPS 2025 Spotlight
- **作者**: Shaohang Wei, Wei Li, Feifan Song, Wen Luo, Tianyi Zhuang, Haochen Tan, Zhijiang Guo, Houfeng Wang
- **代码**: https://github.com/sylvain-wei/TIME
- **数据**: https://huggingface.co/datasets/SylvainWei/TIME
- **项目页**: https://sylvain-wei.github.io/TIME/

## 一句话总结
TIME 提出一个面向真实世界时序推理的多层级 benchmark，覆盖 38,522 个 QA、3 个子数据集与 11 个细粒度子任务，系统刻画 LLM 在高密度时间信息、快速事件变化和复杂社会时序依赖下的推理能力，并分析了 test-time scaling 对 temporal reasoning 的实际影响。

## 背景与动机
时间推理是语言模型理解现实世界的关键能力，但现有工作往往存在三类脱节：
- 任务过于理想化，缺少真实世界事件复杂度；
- temporal information density 不足；
- 少有系统分析不同 temporal reasoning 子能力。

作者认为，真正有价值的 benchmark 必须面对现实世界中的：
- 密集时间线；
- 高频动态变化；
- 社交互动中的复杂依赖关系。

## 核心问题
如何构建一个覆盖不同难度层级、不同现实语境和不同时间推理能力维度的 benchmark，用于系统测量 LLM temporal reasoning 的真实能力边界？

## 方法详解

### 1. 多层级设计
TIME 共包含 3 个 level 和 11 个细粒度子任务，强调从基础时间识别到复杂关系推断的层次化难度。

### 2. 三个真实子数据集
- **TIME-Wiki**：百科/知识类时序信息；
- **TIME-News**：新闻中高速变化事件的时间推理；
- **TIME-Dial**：对话和社会交互中的时间依赖。

这三类数据共同覆盖静态知识、动态事件、社会时序三种关键现实场景。

### 3. 大规模评测与 TIME-Lite
- 总计 38,522 QA；
- 提供人工标注子集 TIME-Lite；
- 系统评测 reasoning models 与 non-reasoning models。

### 4. Test-time scaling 分析
论文不仅给 benchmark，还专门分析了 test-time scaling 对 temporal reasoning 的影响，这是很有价值的附加贡献：
- 不是所有 temporal 任务都能从更长推理链受益；
- 不同任务类型对推理预算的敏感性不同。

## 实验结论
根据摘要：
- 当前 LLM 在真实场景 temporal reasoning 上仍存在明显短板；
- 不同层级与任务间性能差异显著；
- test-time scaling 对 temporal reasoning 的帮助是有条件的，而非普适增强。

## 亮点
1. **Benchmark 设计扎实**：多层级、多数据源、多子任务。
2. **面向现实世界**：不再停留在玩具时序推理。
3. **分析维度全面**：不仅给榜单，还研究推理预算效应。
4. **对 reasoning 研究很有价值**：可直接用于比较 R1 类模型和普通模型的真实差距。

## 局限性
1. 仍主要是文本时序推理，和视频/多模态时间理解之间有 gap。
2. benchmark 本身不解决能力提升，只提供测量工具。
3. 数据构造方式可能引入语料域偏差。

## 与相关工作的对比
- 相比一般时间 QA 数据集：TIME 更强调真实世界复杂情境和任务层级。
- 相比单一新闻/知识时序 benchmark：TIME 融合 Wiki、News、Dialogue 三种来源。
- 相比只报告平均分的评测：TIME 更关注能力分解和 scaling 规律。

## 启发
- 可将 TIME 作为 agent 规划中的时间一致性评测基准。
- 可结合 Video-LLM 和多模态事件数据，扩展为跨模态 TIME。
- 对长链推理方法的收益判断，TIME 提供了更细粒度诊断维度。

## 评分
- 新颖性：★★★★☆
- 基准价值：★★★★★
- 分析深度：★★★★☆
- 实用价值：★★★★★