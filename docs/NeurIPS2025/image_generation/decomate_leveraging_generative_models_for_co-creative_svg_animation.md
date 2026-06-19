---
title: >-
  [论文解读] Decomate: Leveraging Generative Models for Co-Creative SVG Animation
description: >-
  [NeurIPS 2025 (GenProAI Workshop)][图像生成][SVG animation] 提出 Decomate 交互系统，利用多模态大语言模型 (MLLM) 将非结构化 SVG 图形自动分解为语义组件，设计师通过自然语言为各组件指定动画行为，系统生成可生产的 HTML/CSS/JS 动画代码，支持迭代协作创作。
tags:
  - "NeurIPS 2025 (GenProAI Workshop)"
  - "图像生成"
  - "SVG animation"
  - "semantic decomposition"
  - "MLLM"
  - "co-creative design"
  - "natural language interaction"
---

# Decomate: Leveraging Generative Models for Co-Creative SVG Animation

**会议**: NeurIPS 2025 (GenProAI Workshop)  
**arXiv**: [2511.06297](https://arxiv.org/abs/2511.06297)  
**代码**: [HuggingFace Demo](https://huggingface.co/spaces/PrompTartLAB/Decomate)  
**领域**: 人机协作 / 创意计算 / SVG 动画  
**关键词**: SVG animation, semantic decomposition, MLLM, co-creative design, natural language interaction  

## 一句话总结
提出 Decomate 交互系统，利用多模态大语言模型 (MLLM) 将非结构化 SVG 图形自动分解为语义组件，设计师通过自然语言为各组件指定动画行为，系统生成可生产的 HTML/CSS/JS 动画代码，支持迭代协作创作。

## 研究背景与动机

**领域现状**：SVG 动画是 UI/UX 设计中重要但门槛高的技能。设计师需在多个平台间切换、手动编辑代码或依赖开发者实现动效，工作流缺乏灵活性和精细控制。

**现有痛点**（来自对 11 位专业设计师的访谈）：
   - **(a) 时间限制**：动画常因排期紧张被降优先级，许多设计师没有机会探索和集成动效
   - **(b) 精细调控依赖开发者**：调整 timing、easing 等细节需开发者配合，创意意图与最终效果脱节
   - **(c) AI 工具缺席**：全部受访者对 AI 辅助动画有兴趣，但无人使用过——缺乏面向设计师的直觉化工具

**现有工作不足**：Keyframer 等工具虽然支持自然语言生成 SVG 动画，但假设输入 SVG 有良好的结构（explicit class labels 和 groupings）——而真实设计资产通常是扁平的、无组织的。

**核心矛盾**：设计师的创意意图（"让翅膀缓慢拍动"）与底层技术实现（如何拆分 SVG 路径、编写 keyframe 动画）之间存在巨大鸿沟。

## 方法详解

### 整体管线（6 步交互流程）

**Step 1: SVG 输入**
- 用户上传 SVG 文件或粘贴 SVG 代码
- 系统接受从完全扁平到有组织的各种结构，无需预处理
- 用户提供高层对象名（如"dog"），引导语义分解

**Step 2: 语义分解 + 动画建议**
- 使用 MLLM（Claude Sonnet 4）同时分析 SVG 代码和渲染图像
- 将 SVG 重构为语义有意义的组件组（如"ears"、"nose"、"legs"）
- 优先视觉解释而非句法层级
- 为每个组生成自然语言动画建议

**Step 3: 预览 + 结构调整**
- 可视化展示分组方案
- 若分组不符合创意意图，用户用自然语言反馈调整（如"把左脚和右脚分开"）
- 系统根据反馈迭代修正分组

**Step 4: Prompt 驱动的动画创作**
- 用户以系统建议为基础，用自然语言描述动画行为
- 如："make the wings flap slowly with elastic easing"
- 无需了解 CSS animation 语法

**Step 5: 代码生成与渲染**
- LLM（Claude Sonnet 4）将最终分组结构 + 动画 prompt 翻译为 HTML/CSS/JS 代码
- 输出即可部署的动画 SVG + 源代码

**Step 6: 交互预览与迭代**
- 实时预览动画效果
- 用户检查 timing、easing、空间行为
- 通过追加 prompt 进一步调整（如"increase the bounce on landing"）
- 每次修改触发代码重新生成 + 预览更新

### 技术核心

- **语义分解**：不依赖 SVG 原有的 `<g>` 标签或 class 名——MLLM 理解视觉语义来重新分组 SVG 元素
- **双模型架构**：MLLM (视觉+代码理解) 做分解 + LLM 做代码生成，分工明确
- **迭代共创循环**：人-AI 在语义结构和动画行为两个层面上交替迭代

## 实验关键数据

### 形成性访谈（11 位专业设计师）

| 受访者 | 经验 | 动画工具 | 对 AI 辅助动画的态度 |
|--------|------|---------|-------------------|
| P1-P4 | 1-10年 | Figma, After Effects, Code | 有兴趣但未使用 |
| P5-P8 | 3-10年 | Figma, Lottie, After Effects | 有兴趣但未使用 |
| P9-P11 | 1-10年 | Figma, Lottie | 有兴趣但未使用 |

100% 的受访者对 AI 辅助动画有兴趣，但 0% 有使用经验。

### 用户研究（6 位参与者的定性反馈）

| 维度 | 正面反馈 | 主要问题 |
|------|---------|---------|
| 语义分解 | 对扁平/无组织 SVG 效果好，最小人工干预 | 部分元素分组不完整（如鲸鱼眼睛脱离身体） |
| 自然语言动画 | 意图-动画对齐基本合理 | 用户不确定如何措辞 prompt，有学习曲线 |
| 视觉质量 | 方向正确、基本效果可接受 | 精细控制不足（如运动强度、层级 timing） |
| 交互迭代 | 自然语言迭代优化有效 | 部分动作线性化，不够自然（如 swing vs linear） |

### 代表性用户直接反馈

- P2："对象分解很好，但动画行为出乎意料（鲸鱼眼睛脱离了）"
- P5："从文本直接生成 SVG 动画很方便，但没有 AI 建议时动效会很生硬"
- P6："直接的 UI 控件（曲线、时长）比纯文本输入更好用"

## 亮点与洞察

- **解决了 Keyframer 等工具的核心假设缺陷**：不要求 SVG 有预定义结构，通过 MLLM 视觉理解自动分解
- **人-AI 共创的两层迭代**：在"语义结构"和"动画行为"两个层面都支持自然语言驱动的迭代——比单层迭代更灵活
- **需求来源充分**：11 人访谈 → 系统设计 → 6 人用户研究的闭环验证
- **实际可用**：HuggingFace 上有 live demo，输出是可部署的 HTML/CSS/JS

## 局限与展望

- **Prompt 表达困难**：用户不确定如何措辞动画 prompt——需要模板/示例/自动补全机制
- **精细控制不足**：纯自然语言难以精确描述微妙的运动参数（如 easing 曲线、具体时长）——需要结合 slider/toggle 等结构化控件
- **语义分解偶尔出错**：MLLM 可能错误分组（如身体部分应联动但被拆分），需要更强的全局一致性约束
- **仅 Workshop paper**，评估以定性反馈为主，缺少定量对比实验（如 vs Keyframer 的控制实验）
- **依赖特定 LLM (Claude Sonnet 4)**：模型替换后效果未知，prompt engineering 与模型绑定

## 相关工作对比

- **vs Keyframer**：Keyframer 假设 SVG 结构良好，Decomate 增加了语义分解步骤处理非结构化 SVG
- **vs Kolthoff et al. (零样本 GUI 原型)**：聚焦 GUI 生成而非 SVG 动画，无语义分解
- **vs Xing et al. (SVG 理解/生成)**：提供数据集增强 LLM 的 SVG 能力，但不针对动画创作流程
- **vs 传统工具 (Figma + Lottie + After Effects)**：传统工具需要手动操作和技术专长，Decomate 降低门槛但精细度不足

## 评分
- 新颖性: ⭐⭐⭐⭐ 语义分解 + prompt 动画的组合是增量式创新，核心依赖 LLM 能力
- 实验充分度: ⭐⭐⭐⭐ 有需求调研+用户研究，但仅定性评估、小样本(6人)、无定量对比
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰、流程图和界面截图详细、用户反馈原汁原味
- 价值: ⭐⭐⭐⭐ 作为 workshop demo 有展示意义，但离可靠的设计工具仍有距离

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Enhancing Creative Generation on Stable Diffusion-based Models](../../CVPR2025/image_generation/enhancing_creative_generation_on_stable_diffusion-based_models.md)
- [\[CVPR 2025\] Redefining <Creative> in Dictionary: Towards an Enhanced Semantic Understanding of Creative Generation](../../CVPR2025/image_generation/redefining_creative_in_dictionary_towards_an_enhanced_semantic_understanding_of_.md)
- [\[ICCV 2025\] Revelio: Interpreting and Leveraging Semantic Information in Diffusion Models](../../ICCV2025/image_generation/revelio_interpreting_and_leveraging_semantic_information_in_diffusion_models.md)
- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[CVPR 2025\] Consistent and Controllable Image Animation with Motion Diffusion Models](../../CVPR2025/image_generation/consistent_and_controllable_image_animation_with_motion_diffusion_models.md)

</div>

<!-- RELATED:END -->
