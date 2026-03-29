# OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use

**会议**: ACL 2025
**arXiv**: [2508.04482](https://arxiv.org/abs/2508.04482)
**代码**: https://os-agent-survey.github.io/
**领域**: LLM Agent / 操作系统代理
**关键词**: OS Agents, GUI Grounding, MLLM, Agent Framework, Computing Device Automation

## 一句话总结
首篇全面综述 MLLM 驱动的操作系统代理（OS Agents），系统梳理其基础组件、构建方法、评估基准和未来方向。

## 研究背景与动机
1. **领域现状**：随着 Claude Computer Use、Apple Intelligence、AutoGLM 等产品发布，基于 (M)LLM 的 OS Agents 正快速发展。
2. **现有痛点**：缺乏对 OS Agents 领域的系统化综述，研究分散在 web、mobile、desktop 等不同平台。
3. **核心矛盾**：Siri/Cortana 等传统助手受限于上下文理解能力，无法实现真正的通用设备自动化。
4. **本文要解决什么？** 提供 OS Agents 研究的全景图，整合基础、方法、评估和未来方向。
5. **切入角度**：从环境（Environment）、观察空间（Observation Space）、动作空间（Action Space）三个核心组件出发。
6. **核心idea一句话**：构建 OS Agents 需要理解（Understanding）、规划（Planning）、执行落地（Grounding）三大核心能力。

## 方法详解

### 整体框架
综述从四个维度展开：(1) 基础定义和核心组件；(2) 基础模型构建（架构+训练策略）；(3) 代理框架设计；(4) 评估协议和基准。

### 关键设计
1. **基础模型构建**:
   - 做什么：总结 30+ 个 OS Agent 基础模型的架构和训练策略
   - 核心思路：四类架构（现有 LLM / 现有 MLLM / 拼接 MLLM / 改进 MLLM）+ 三阶段训练（预训练 / SFT / RL）
   - 设计动机：不同平台的 GUI 特性要求不同的架构适配方案

2. **代理框架设计**:
   - 做什么：梳理感知（Perception）、规划（Planning）、记忆（Memory）、动作（Action）四大模块
   - 核心思路：文本描述 vs GUI 截图感知、全局 vs 迭代规划、自动探索 vs 经验增强记忆
   - 设计动机：构建灵活的非微调代理框架

3. **评估体系**:
   - 做什么：整理桌面/移动/Web 三大平台的评估基准
   - 核心思路：按平台和任务类型分类，涵盖在线/离线评估
   - 设计动机：指导研究者选择合适的评测场景

### 核心发现

| 维度 | 关键总结 |
|------|---------|
| 架构选择 | MLLM（如 InternVL, Qwen-VL）是主流，视觉理解能力关键 |
| 训练策略 | SFT 最广泛；RL 主要用于 AutoGLM 等自进化场景 |
| 数据构建 | 跨平台泛化需统一动作空间；合成数据是主要来源 |
| 关键挑战 | 高分辨率 GUI 理解、动态环境适应、安全隐私 |

## 实验关键数据

### 基础模型对比

| 模型 | 架构 | 预训练 | SFT | RL | 日期 |
|------|------|-------|-----|-----|------|
| OS-Atlas | 现有 MLLM | ✓ | ✓ | - | 10/2024 |
| AutoGLM | 现有 LLM | ✓ | ✓ | ✓ | 10/2024 |
| ShowUI | 现有 MLLM | ✓ | ✓ | - | 10/2024 |
| CogAgent | 改进 MLLM | ✓ | ✓ | - | 12/2023 |
| Ferret-UI | 现有 MLLM | - | ✓ | - | 04/2024 |

### 消融实验
综述性论文无典型消融实验，但讨论了：
- 视觉 vs 文本观察方式对 grounding 性能的影响
- 不同规划策略（全局 vs 迭代）的优劣权衡

### 关键发现
- MLLM 视觉输入比纯 HTML 输入在 GUI grounding 上表现更好
- 跨平台泛化是核心难题，需要统一动作空间
- RL 自进化方法（如 AutoGLM）展现出强大的错误恢复能力

## 亮点与洞察
- 提供了极为完整的文献库和分类体系
- 明确指出 grounding（动作落地）而非理解是当前最大瓶颈
- 安全隐私+个性化自进化是两个最重要的未来方向

## 局限性 / 可改进方向
- 作为 ACL 9 页版本，部分内容（如评估基准细节）较简略
- 缺少定量的跨方法对比实验
- 未充分讨论多代理协作场景

## 相关工作与启发
- **vs GUI Agent Survey (Zhang et al. 2024)**: Zhang 聚焦 GUI grounding，本综述覆盖更广的 OS Agent 全栈
- **vs WebAgent (Gur et al. 2023)**: 早期 web 代理工作，本综述补充了移动端和桌面端进展


## 补充细节
- 综述涵盖了约 30 个基础模型和 20+ 个代理框架
- 深入讨论了四种架构类型：现有 LLM、现有 MLLM、拼接 MLLM、改进 MLLM
- 数据构建策略包括：跨平台 A11y 树模拟、网页渲染、智能手机交互图
- 未来方向：安全隐私保护、个性化与自进化、跨平台泛化
- 维护了一个持续更新的 GitHub 资源库
- 代理框架核心四模块：感知(Perception)、规划(Planning)、记忆(Memory)、动作(Action)
- 代表性产品：Anthropic Computer Use、Apple Intelligence、AutoGLM、Project Mariner
- 高分辨率 GUI 理解是核心技术挑战，CogAgent 引入 1120x1120 高分辨率编码器

## 评分
- 新颖性: ⭐⭐⭐⭐ 首篇系统化 OS Agents 综述，分类框架完整
- 实验充分度: ⭐⭐⭐ 综述性工作，无自有实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但 9 页版本内容压缩明显
- 价值: ⭐⭐⭐⭐⭐ 对快速发展的 OS Agents 领域提供了极好的入门导引
