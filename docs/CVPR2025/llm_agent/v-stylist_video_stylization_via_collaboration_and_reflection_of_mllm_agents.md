# V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents

**会议**: CVPR 2025  
**arXiv**: [2503.12077](https://arxiv.org/abs/2503.12077)  
**代码**: 待确认  
**领域**: LLM Agent / 视频生成  
**关键词**: 视频风格化, 多Agent协作, MLLM自反思, 风格树搜索, AnimateDiff+ControlNet

## 一句话总结
提出 V-Stylist，一个基于 MLLM 多 agent 协作和反思的视频风格化系统，通过 Video Parser（视频分镜）、Style Parser（风格树搜索）和 Style Artist（多轮自反思渲染）三个角色协作，在复杂转场视频和开放风格描述上实现 SOTA，整体指标超越 FRESCO 6.05%。

## 研究背景与动机
1. **领域现状**：视频风格化取得了显著进展，但现有方法难以处理包含复杂转场的视频，且无法基于开放描述指定风格。
2. **现有痛点**：(1) 复杂转场视频的风格一致性难保证 (2) 用户的风格描述通常模糊，难以精确匹配
3. **核心矛盾**：需要同时解决视频结构理解（转场分析）和开放风格理解（模糊描述匹配）。
4. **本文要解决什么？** 构建通用的视频风格化系统，自动处理复杂转场和模糊风格描述。
5. **切入角度**：模拟人类专业流程——先分镜、再选风格、最后调细节——每步由专门的 MLLM agent 负责。
6. **核心idea一句话**：三角色 MLLM agent（分镜/选风格/渲染）+ 风格树搜索 + 多轮自反思，模拟专业人员工作流。

## 方法详解

### 整体框架
输入为用户视频和风格描述，输出为风格化视频。三个 MLLM agent 角色分工协作：Video Parser 处理视频分镜，Style Parser 匹配风格模型，Style Artist 执行风格渲染。

### 关键设计

1. **Video Parser（视频解析 Agent）**:
   - 做什么：将视频分解为镜头序列，为每个镜头生成扩散模型兼容的 prompt
   - 核心思路：AutoShot 检测转场 → Qwen2-VL 生成镜头字幕 → Mistral8x7B 转译为扩散模型 prompt
   - 每个镜头提取 3 帧作为关键帧
   - 设计动机：逐镜头处理可有效应对复杂转场，而字幕→prompt 的转译确保扩散模型能正确理解内容

2. **Style Parser（风格解析 Agent）**:
   - 做什么：从模糊的用户风格描述中精确匹配最佳风格模型
   - 核心思路：**风格树 + Tree-of-Thought 搜索**。风格模型组织为 3 层树结构（根→2 大类[Artistic/Realistic]→25 个模型覆盖 17 种风格）。Mistral8x7B 先提取风格偏好 → 5 个风格专家 + 1 个主席投票式逐层下降搜索：$\mathcal{D}_{l+1} = LLM(\mathcal{D}_l | \mathcal{S}, \mathcal{T})$
   - 风格类型覆盖：油画、日本动漫、像素艺术、西方写实等 17 种
   - 设计动机：用户说"电影感"时，直接匹配会失败，但树搜索可先定位"写实类"→再细化到具体风格模型

3. **Style Artist（风格渲染 Agent，含自反思）**:
   - 做什么：用匹配的风格模型渲染视频镜头，通过多轮 MLLM 自评估迭代优化
   - 核心思路：SD v1.5 + AnimateDiff（时序一致性）+ 4 个 ControlNet（tile/depth/softedge/lineart）渲染。渲染后 MLLM 评分（0-100），评分 ≥60 则接受，否则 MLLM 生成新的 ControlNet 权重再次渲染，最多 3 轮
   - 公式：$\mathcal{Y}_t = \mathcal{M}_L(\mathcal{X}_t, \mathcal{P}_t | \mathcal{C}_{1:N} \cdot \mathcal{W}_{1:N})$
   - 设计动机：一次渲染往往不理想，自反思机制可自适应调整 ControlNet 权重以平衡内容保持和风格表达

## 实验关键数据

### TVSBench（50 视频，平均 30 秒 30FPS，17 种风格）
| 方法 | CLIP-T | Aesthetic-V | Distortion-V | **Overall** |
|------|:---:|:---:|:---:|:---:|
| ControlVideo | 0.263 | 0.587 | 0.587 | 0.541 |
| FRESCO | 0.239 | 0.527 | 0.556 | 0.541 |
| Rerender | 0.206 | 0.412 | 0.404 | 0.502 |
| FLATTEN | 0.243 | 0.528 | 0.557 | 0.487 |
| **V-Stylist** | **0.267** | **0.591** | **0.745** | **0.601** |

### 消融实验
| 配置 | CLIP-T | Distortion-V | 说明 |
|------|:---:|:---:|------|
| Baseline（无 agent） | 0.256 | 0.576 | SD+ControlNet 直接生成 |
| + Video Parser | 0.263 | 0.584 | 加入分镜和 prompt 转译 |
| + Style Parser | 0.266 | 0.575 | 加入风格树搜索 |
| **Full V-Stylist** | **0.266** | **0.590** | 加入自反思渲染 |

Style Artist 自反思贡献最大——**视频质量和风格对齐提升 25.16%**

### 关键发现
- V-Stylist 整体指标超 FRESCO 6.05%、超 ControlVideo 4.51%
- 视频级失真（Distortion-V）优势最大：0.745 vs 次优 0.587（+27%），说明自反思机制对时序一致性贡献显著
- 风格树搜索在处理模糊描述时比直接匹配更鲁棒
- 硬件：8×RTX A6000，参数 top-k=10, top-p=0.95, temp=0.7

## 亮点与洞察
- **模拟人类专业工作流**的思路优雅——分镜/选风格/调细节是视频专业人员的标准流程
- **风格树 + Tree-of-Thought 搜索**的组合为模糊需求的精确匹配提供了新思路
- 多轮自反思机制可迁移到其他视觉生成任务

## 局限性 / 可改进方向
- 依赖预构建的风格树和风格模型库
- 多 agent + 多轮反思的计算开销较大
- 风格一致性在长视频上可能仍有挑战

## 评分
- 新颖性: ⭐⭐⭐⭐ 三角色协作+风格树搜索设计新颖
- 实验充分度: ⭐⭐⭐⭐ 新benchmark+定量对比+用户研究
- 写作质量: ⭐⭐⭐⭐ 结构清晰
- 价值: ⭐⭐⭐⭐ 对多agent视觉生成有参考价值
