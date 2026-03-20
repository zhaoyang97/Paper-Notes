# OS Agents: A Survey on MLLM-based Agents for Computer, Phone and Browser Use

**会议**: ACL 2025 (Long Paper)  
**arXiv**: 见ACL Anthology  
**网站**: https://os-agent-survey.github.io/  
**领域**: Agent / 多模态VLM  
**关键词**: OS Agent, GUI Agent, MLLM, 自主操作, 计算机使用, 手机控制, 浏览器自动化  

## 一句话总结
首个系统性综述基于（多模态）大语言模型的操作系统智能体（OS Agents），覆盖基础概念、构建方法（基础模型+Agent框架）、评估基准和商业产品，全面梳理了从CogAgent到Anthropic Computer Use等50+工作的技术演进。

## 背景与动机
OS Agent是指能像人类一样使用电脑、手机和浏览器的AI智能体——通过GUI或CLI操作完成用户指定的任务。随着GPT-4V、Claude、Gemini等MLLM的涌现，OS Agent从概念变为现实：OpenAI Operator、Anthropic Computer Use、Apple Intelligence、Google Project Mariner等商业产品相继发布。但学术研究高度分散，缺乏统一的综述和分类框架。

## 核心框架

### OS Agent基本组成

1. **环境（Environment）**: 计算机、手机、浏览器——三大操作平台
2. **观察空间（Observation Space）**: 屏幕截图（视觉）、HTML代码（结构化文本）、可访问性树（语义化描述）
3. **动作空间（Action Space）**: 点击、输入、滑动、长按、导航等操作

### 三大核心能力

1. **理解（Understanding）**: 理解复杂的GUI界面，识别小图标、密集文本、多层嵌套的界面元素
2. **规划（Planning）**: 将复杂任务分解为子任务序列，根据环境反馈动态调整计划
3. **定位（Grounding）**: 将文本指令映射到具体的屏幕元素和可执行动作（坐标、参数）

### 构建方法

#### 基础模型
- **现有LLM/MLLM**: 直接用GPT-4V、Claude等作为backbone
- **MLLM + 额外视觉模块**: CogAgent（高低分辨率双编码器）、Ferret-UI等
- **定制架构**: SeeClick、OS-Atlas等专门针对GUI设计的模型
- **训练策略**: 预训练（大规模GUI数据）、SFT（GUI任务指令数据）、RL（在线交互反馈）

#### Agent框架
- **观察处理**: Set-of-Mark提示、HTML解析、A11y Tree、OCR辅助
- **记忆机制**: 短期工作记忆（动作历史）、长期经验记忆（知识库）
- **规划策略**: 目标分解、反思（Reflexion）、任务图规划
- **动作定位**: 坐标预测、元素ID匹配、函数调用

### 评估基准

| 平台 | 代表性Benchmark | 特点 |
|------|----------------|------|
| 手机 | AndroidWorld, AITW | 真实手机环境/模拟器 |
| 电脑 | OSWorld, WindowsAgentArena | 跨平台、真实OS |
| 浏览器 | Mind2Web, WebArena, WebVoyager | 真实网页交互 |
| 跨平台 | AssistantBench | 复杂跨应用任务 |

### 商业产品
- **OpenAI Operator**: 任务自动化服务
- **Anthropic Computer Use**: Claude直接操作用户电脑
- **Apple Intelligence**: 集成Siri+设备操作
- **Google Project Mariner**: Chrome扩展形式的Agent

## 关键挑战与未来方向

1. **安全与隐私**: Agent直接操作用户设备，面临prompt injection、对抗攻击、数据泄露等风险
2. **个性化与自进化**: Agent需要记住用户偏好、从交互中持续学习，但多模态记忆管理是个难题
3. **GUI理解瓶颈**: 高分辨率屏幕中的小元素检测仍然困难，尤其是复杂布局和动态内容
4. **长步骤推理**: 复杂任务可能需要10-50步操作，错误累积导致成功率急剧下降
5. **泛化性**: 在一个App上训练的Agent难以迁移到其他App

## 亮点
- **全景式覆盖**: 从基础概念到商业产品，涵盖了OS Agent领域的方方面面
- **分类框架清晰**: 按平台（手机/电脑/浏览器）、按方法（基础模型/框架）、按能力（理解/规划/定位）三维度组织
- **时间线完整**: 从2023年早期工作到2025年最新进展，包括50+学术工作和4大商业产品
- **挑战分析深入**: 安全、个性化、自进化等方向的分析有前瞻性

## 局限性 / 可改进方向
- 作为综述，缺乏统一的实验对比（不同benchmark的结果难以横向比较）
- 对"如何评估OS Agent的实际可用性"讨论不够深入
- 未深入讨论Agent的计算成本和延迟问题（每步都需要MLLM推理）
- 多Agent协作和复杂工作流的讨论篇幅有限

## 启发与关联
- GUI理解本质上是一种特殊的文档理解——与mPLUG-DocOwl2的技术可以互通
- OS Agent的"定位"能力可以受益于VLM的grounding能力提升（如visual evidence prompting）
- Agent的长步骤推理问题可以用tree search/MCTS方法缓解
- 安全问题是商业化的核心障碍——prompt injection防御是一个重要研究方向

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统性OS Agent综述，分类框架有价值
- 实验充分度: ⭐⭐⭐ 综述性质，无原创实验
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，覆盖全面，图表信息密度高
- 价值: ⭐⭐⭐⭐⭐ 对OS Agent领域的入门和全局把握极有帮助
