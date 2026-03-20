# AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2410.24024](https://arxiv.org/abs/2410.24024)  
**代码**: [https://github.com/THUDM/Android-Lab](https://github.com/THUDM/Android-Lab)  
**领域**: Agent / 多模态VLM  
**关键词**: Android Agent, GUI自动化, 手机Agent, Benchmark, 指令微调  

## 一句话总结
提出AndroidLab——首个统一训练和评估Android Agent的系统性框架，包含9个App上的138个可复现任务，同时支持纯文本（XML模式）和多模态（SoM模式）模型，并构建Android Instruct数据集（94.3k步骤），将开源LLM的成功率从4.59%提升至21.50%。

## 背景与动机
Android Agent研究面临三大问题：(1) 评估不系统——已有benchmark大多只测闭源模型，且环境不可复现；(2) 缺乏统一框架——LLM和LMM使用不同的动作空间和观察模式，无法公平比较；(3) 开源模型差距大——开源模型在手机操作任务上success rate极低（~5%），缺乏高质量训练数据。

## 核心问题
如何建立一个系统性框架，使得Agent的训练数据收集、模型评估和跨模态比较在统一标准下进行？

## 方法详解

### 整体框架
AndroidLab = 统一操作环境 + 可复现Benchmark + Android Instruct训练数据集。

### 关键设计

1. **统一操作模式**: 
   - **XML模式（纯文本LLM）**: Agent接收UI树的XML描述，输出动作
   - **SoM模式（多模态LMM）**: Agent接收带Set-of-Mark标注的屏幕截图，输出动作
   - 两种模式共享**完全相同的动作空间**（Tap/Swipe/Type/Long Press/Press Key/Finish），确保公平比较
   - 每种模式都可用ReAct或SeeAct推理框架

2. **可复现Benchmark**: 
   - 9个预安装App（Clock, Settings, Contacts, Calendar, Bluecoins, Pi Music, MAPS.ME, Zoom, Cantook）
   - 138个任务，每个任务拆分为多个子目标（sub-goals），通过UI树结构匹配验证完成度
   - 使用Android虚拟设备+离线数据，消除网络和时间依赖

3. **评估指标**: 
   - **Success Rate**: 整体任务完成率
   - **Sub-Goal Success Rate**: 子目标完成率（更细粒度）
   - **Reversed Redundancy**: 衡量操作效率（冗余步骤比例）
   - **Reasonable Operation**: 合理操作占比

4. **Android Instruct数据集**: 在线标注工具收集10.5k条轨迹、94.3k步操作。用于SFT的子集包含726条轨迹、6208步操作，同时包含XML和SoM格式。

### 损失函数 / 训练策略
- 标准自回归SFT损失
- 训练6个开源模型：Llama-3.1-8B, GLM-4-9B, Qwen2-7B（LLM）+ Qwen2-VL-7B, CogVLM2, Llama-3.2-11B-Vision（LMM）

## 实验关键数据

**闭源模型成绩**:

| 模型 | 模式 | Success Rate |
|------|------|-------------|
| GPT-4o | XML+ReAct | 31.16% |
| GPT-4o | SoM+SeeAct | 31.16% |
| Claude-3.5-Sonnet | SoM+SeeAct | ~27% |
| Gemini-1.5-Pro | XML+ReAct | ~22% |

**开源模型SFT前后对比**:

| 模型 | SFT前 | SFT后 | 提升 |
|------|-------|-------|------|
| LLM平均 | 4.59% | **21.50%** | +16.91% |
| LMM平均 | 1.93% | **13.28%** | +11.35% |
| Llama-3.1-8B | 3.2% | **24.6%** | +21.4% |
| Qwen2-VL-7B | 4.3% | **17.4%** | +13.1% |

SFT后的开源7B模型接近闭源GPT-4o约2/3的成功率！

### 消融实验要点
- **XML vs SoM**: XML模式整体略优于SoM（文本信息更精确），但SoM模式更适合视觉密集场景
- **ReAct vs SeeAct**: ReAct框架整体更好（显式推理链有帮助）
- **SFT效果**: 不仅提升成功率，还显著降低操作冗余度、提高合理操作比例
- **跨App泛化**: SFT后在未见过的App上也有一定泛化能力
- **LLM vs LMM**: LLM SFT后成绩更高（21.5% > 13.3%），因为XML信息更精确

## 亮点
- **公平统一**: 首次在完全相同的动作空间下比较LLM和LMM的Agent能力
- **可复现**: 虚拟设备+离线数据，任何人都可以复现实验
- **数据+模型+评估全开源**: 94.3k步训练数据、6个微调模型、评估框架全部公开
- **SFT效果显著**: 小量高质量数据（6208步）就能让7B模型接近GPT-4o的66%性能

## 局限性 / 可改进方向
- 仅9个App、138个任务，覆盖的应用场景有限
- 最高成功率仅31.16%（GPT-4o），距离实用水平（>80%）差距很大
- 未引入强化学习（RL）——在线交互反馈可能进一步提升性能
- SoM模式下LMM表现不如LLM的XML模式，说明视觉定位仍是瓶颈
- 多步骤任务的错误传播问题未深入探讨

## 与相关工作的对比
- **vs AndroidWorld**: AndroidWorld有116个任务但不支持训练数据构建；AndroidLab统一了训练和评估
- **vs AITW**: AITW是静态数据集（无在线交互），AndroidLab支持真实在线交互
- **vs B-MOCA**: B-MOCA标准化了虚拟设备但任务多样性有限

## 启发与关联
- Agent SFT数据的质量（而非数量）是关键——6208步就有显著提升
- LMM在GUI定位（grounding）上的弱势与VLM幻觉问题相关——Visual Evidence Prompting可能也对Agent有帮助
- 与OS Agents Survey互为参考：Survey提供全景视角，AndroidLab提供具体训练/评估框架

## 评分
- 新颖性: ⭐⭐⭐⭐ 统一框架和公平比较的设计思路有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 17个模型（6开源+11闭源）、4种操作模式、详细消融
- 写作质量: ⭐⭐⭐⭐ 图表信息密度高，图1+图2就能理解全文要点
- 价值: ⭐⭐⭐⭐⭐ 全套开源（环境+数据+模型+评估），对Agent社区贡献极大
