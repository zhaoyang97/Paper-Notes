# SpiritSight Agent: Advanced GUI Agent with One Look

**会议**: CVPR 2025  
**arXiv**: [2503.03196](https://arxiv.org/abs/2503.03196)  
**代码**: [https://hzhiyuan.github.io/SpiritSight-Agent](https://hzhiyuan.github.io/SpiritSight-Agent)  
**领域**: LLM Agent  
**关键词**: GUI Agent, VLM, 元素定位, 多平台, 动态分辨率, UBP

## 一句话总结
提出 SpiritSight，一个基于视觉的端到端 GUI agent，通过 573 万样本的多层级数据集 GUI-Lasagne 和 Universal Block Parsing (UBP) 方法解决动态高分辨率输入的定位歧义，SpiritSight-8B 在 Multimodal-Mind2Web 上非候选元素设置下 Step SR 达 52.7%，全面超越所有视觉/语言/混合方法。

## 研究背景与动机
1. **领域现状**：GUI Agent 的三类方法——语言型（HTML/XML 输入）、视觉-语言混合型（截图+HTML）、纯视觉型（仅截图）。语言型和混合型在准确率上领先，但受平台限制（HTML 仅 Web 可用）、安全风险（注入攻击）和延迟问题。
2. **现有痛点**：纯视觉方法虽然兼容所有平台且延迟低，但 **元素定位（element grounding）能力严重不足**——现有视觉 GUI agent 无法精确定位屏幕上的小按钮、文本框等元素。两阶段方法（如 MobileAgent）引入 OCR/图标识别工具辅助定位，但增加了系统复杂度和延迟。
3. **核心矛盾**：现代 VLM（如 InternVL2）采用动态高分辨率策略——将输入图片按最佳宽高比裁切为 448×448 的块再拼接输入。但在 GUI 场景下，**块的展平操作丢失了 2D 空间信息**，导致模型学到的 $f: \mathbf{p}'' \to \mathbf{p}$ 映射是多值函数（同一块内坐标对应不同全局坐标），产生定位歧义。
4. **本文要解决什么？** (1) 用大规模、高质量 GUI 数据增强 VLM 的 GUI 感知和定位能力；(2) 从算法层面解决动态分辨率的定位歧义。
5. **切入角度**：数据（GUI-Lasagne 覆盖认知→定位→导航的三层级训练）+ 方法（UBP 用块内坐标替代全局坐标，消除多值映射）。
6. **核心idea一句话**：用三级递进的 573 万样本 GUI 数据训练 VLM 的定位能力，用 UBP 将多值映射变为单射映射消除歧义。

## 方法详解

### 整体框架
SpiritSight 基于 InternVL2（2B/8B/26B），分两阶段训练：(1) **持续预训练**：在 GUI-Lasagne 全量数据上训练，解冻视觉编码器、解码器和 MLP 层；(2) **微调**：在具体下游 GUI 导航数据集上 LoRA 微调。推理时输入 GUI 截图 + 任务描述 + 历史动作，输出下一步操作代码。策略被分解为三个子策略：步骤推理 $\pi_s$（生成自然语言描述）、位置推理 $\pi_{pos}$（定位操作坐标）、属性推理 $\pi_{attr}$（确定动作类型）。

### 关键设计

1. **GUI-Lasagne 数据集（573 万样本）**:
   - **Level 1 - Visual-Text Alignment（300 万样本）**：训练模型识别和定位 GUI 中的文本/图标元素。三个任务——text2bbox（给文本找位置）、bbox2text（给位置识别文本）、bbox2dom（给区域生成 DOM 树）。Web 数据从 CommonCrawl + 网站排名收集 755K 网页截图及 DOM 数据，Mobile 数据来自 AitW。自研 InternVL-Icon（在 3 万阿里巴巴图标库上微调 InternVL1.5-26B）用于图标标注。多数据对打包到同一训练样本以充分利用上下文长度
   - **Level 2 - Visual-Function Alignment（150 万样本）**：训练模型理解 GUI 元素的功能语义。function2bbox 任务——给定功能描述定位元素。用 back-translation 方法：将截图分为 3×3 网格描述位置 + 画边界框高亮元素 → InternVL2-26B 生成功能描述 → InternLM2.5-20B 增强多样性。人工验证接受率 90.9%，成本极低
   - **Level 3 - Visual GUI Navigation（64 万样本）**：CoT 风格的导航训练。基于 AitW 数据集，用 GPT-4o 清洗（TPR 93.7%），从 148 万源样本清洗得到 63 万样本。GPT-4o 对每一步进行：描述当前截图 → 与下一步截图对比分析变化 → 推理当前操作合理性
   - 设计动机：三级数据形成能力递进——先学会"看到什么"→再学会"能做什么"→最后学会"如何导航"。Level 1&2 占 90% 数据且可免费/低成本收集，大幅降低数据构建成本

2. **Universal Block Parsing (UBP)**:
   - 做什么：解决动态高分辨率裁切后的位置歧义
   - 核心思路：传统方法训练模型预测全局坐标 $\mathbf{p}=[x,y]$，但块展平后映射 $f: \mathbf{p}'' \to \mathbf{p}$ 是多值的（如 $\mathbf{p}''=[1, 168, 245]$ 可映射到 $[168, 693]$ 或 $[616, 245]$ 取决于 $n_w$）。UBP 改为训练模型预测块内坐标 $\mathbf{p}'' = [b_i, x', y']$，使映射变为单射（injective）。推理时通过后处理恢复全局坐标：$x = x' + (b_i \bmod n_w) \cdot w_{block}$，$y = y' + \lfloor b_i / n_w \rfloor \cdot h_{block}$
   - 结合 2D Block-wise Position Embedding (2D-BPE)：为每个块添加行列位置嵌入，保留 2D 空间关系。所有坐标值归一化到 0-999 区间并取整
   - 设计动机：基于理论分析的严格解法——将多值映射转为单射映射，本质上解决歧义而非用额外缩略图"缓解"。实验证明 UBP 主要提升 Ele.Acc 而非 Op.F1，说明确实是在提升定位能力

### 训练策略
- 预训练：学习率 1e-4/1e-4/5e-5（2B/8B/26B），batch size 1024，全参数训练（视觉编码器+解码器+MLP 全部解冻）
- 微调：先全参数训练 1 epoch（Level 3 + 下游训练集），再 LoRA 微调 1 epoch（alpha 为视觉编码器 32 / LLM 解码器 64）

## 实验关键数据

### 主实验：Multimodal-Mind2Web（非候选元素设置）
| 方法 | 模型大小 | 输入 | Cross-Task SR | Cross-Website SR | Cross-Domain SR |
|------|---------|------|:---:|:---:|:---:|
| SeeAct | - | Text+Image | 40.2% | 32.4% | 36.8% |
| OmniParser | - | Image | 39.4% | 36.5% | 42.0% |
| SeeClick | 9.6B | Image | 25.5% | 16.4% | 20.8% |
| **SpiritSight-2B** | 2B | Image | 44.9% | 37.8% | 36.9% |
| **SpiritSight-8B** | 8B | Image | **52.7%** | **44.0%** | **44.4%** |
| **SpiritSight-26B** | 26B | Image | **54.7%** | **48.1%** | **49.2%** |

### 跨平台导航 & ScreenSpot 功能定位
| 基准 | 之前 SOTA | SpiritSight-8B |
|------|----------|:---:|
| GUI-Odyssey (AMS) | 74.3% | **75.8%** |
| AMEX (AMS) | 70.7% | **80.7%** |
| AndroidControl-High | 64.8% | **68.1%** |
| GUIAct-Multi (Step SR) | 45.4% | **49.3%** |
| ScreenSpot Web | 49.5% (CogAgent) | **68.3%** |
| ScreenSpot Mobile | 65.0% (SeeClick) | **68.4%** |
| ScreenSpot Desktop | 51.1% (SeeClick) | **62.9%** |

### 消融实验
| 配置 | Mind2Web Step SR | 说明 |
|------|:---:|------|
| Level 1 only | ~36% | 基础定位已超越 SeeClick |
| Level 1+2 | ~44% | 功能理解显著提升 |
| Level 1+2+3 | ~52% | 导航数据进一步增益 |
| w/o UBP (baseline) | 较低 Ele.Acc | 定位歧义导致元素选错 |
| UBP + 2D-BPE | 最佳 | 两者互补效果最优 |

### 关键发现
- SpiritSight-8B 纯视觉方法超越所有非候选元素方法（包括语言型和混合型），颠覆了"纯视觉不如结构化输入"的共识
- Level 1 数据（免费收集）贡献了最大的基础增量——仅 Level 1 训练即可超越 SeeClick
- UBP 主要通过提升 Ele.Acc 起作用（Op.F1 几乎不变），直接证明其解决的是定位问题
- SpiritSight-2B 仅用 1/8 预训练数据即可超越 SeeClick，说明 GUI-Lasagne 的数据质量优势
- 跨语言实验：仅用英文训练数据在中文测试上达到英中混合训练的 50% 性能，展现出零样本跨语言能力

## 亮点与洞察
- **UBP 的理论优雅性**：把定位歧义问题形式化为多值函数→单射函数的变换，解法简洁但效果显著。这一思路可推广到任何使用动态分辨率的 VLM 的空间定位任务
- **"免费数据"的力量**：Level 1&2 数据几乎无需人工标注（Web 自动抓取 DOM + InternVL 生成功能描述），占 90% 数据量，这种数据构建范式对其他垂直领域（文档理解、图表分析）有参考价值
- **三层级递进设计**：认知→理解→行动的训练层次完全对应人类学习新软件的过程

## 局限性 / 可改进方向
- 作为纯视觉方法，始终需要截图输入，存在隐私和安全风险（截图可能包含个人信息）
- Level 3 导航数据仅来自移动端（AitW），在 Web 和桌面端缺乏导航训练数据
- 功能描述的自动生成（Level 2）有约 9% 错误率，可能引入噪声
- 未探索多步骤任务的规划能力——当前仅优化单步准确率
- 可探索结合 Level 1-2 免费数据和 RL（如 DigiRL）进一步提升导航能力

## 相关工作与启发
- **vs SeeClick**: 同为视觉 GUI agent 的先驱，SpiritSight 通过更大规模的定位训练数据（573 万 vs ~百万级）和 UBP 方法全面超越。SpiritSight 用 InternVL-2B 仅 1/8 数据即超越 SeeClick
- **vs OmniParser**: OmniParser 用额外的 OCR/图标检测工具辅助定位，SpiritSight 端到端学习定位，更简洁且效果更好
- **vs CogAgent**: CogAgent 用 18B 参数的专用模型，SpiritSight-8B 以一半参数量大幅超越，说明数据质量和 UBP 比模型规模更重要

## 评分
- 新颖性: ⭐⭐⭐⭐ UBP 理论分析优雅，三层级数据设计系统全面，但整体是工程优化而非范式创新
- 实验充分度: ⭐⭐⭐⭐⭐ 6 个 benchmark、3 个模型规模、完善的消融和 scaling 分析、跨语言实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据质量验证充分（人工评估 + TPR/TNR）
- 价值: ⭐⭐⭐⭐⭐ 证明纯视觉 GUI agent 可全面超越结构化输入方法，对领域范式有重要影响
