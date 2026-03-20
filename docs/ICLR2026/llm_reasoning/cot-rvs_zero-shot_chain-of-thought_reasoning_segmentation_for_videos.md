# CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos

**会议**: ICLR 2026  
**arXiv**: [2505.18561](https://arxiv.org/abs/2505.18561)  
**代码**: 未公开  
**领域**: segmentation / video understanding  
**关键词**: reasoning VOS, chain-of-thought, zero-shot, keyframe selection, MLLM, temporal reasoning  

## 一句话总结
提出 CoT-RVS，一种无训练的多智能体框架，利用 MLLM 的零样本 Chain-of-Thought 能力进行时间-语义推理以选择关键帧，实现对复杂隐式查询的推理视频分割，在多个 benchmark 上大幅超越已有方法。

## 背景与动机
1. 推理视频分割(Reasoning VOS)需要根据复杂隐式文本查询生成视频掩码序列
2. 现有方法微调 MLLM 生成分割 token，但在时间敏感查询下表现差
3. 例如"哪个球员投了三分球"需要时间推理而非简单物体检索
4. 已有推理分割方法在图像域成功但缺乏视频时间域的"思考"
5. 微调过程耗时且限制了与闭源 MLLM 的兼容性
6. 当前方法难以整合时间信息与空间/文本上下文

## 方法详解
**多智能体框架**: 三个模块协作——MLLM 关键帧选择器 $\mathcal{F}_{key}$ + 推理图像分割模型 $\mathcal{F}_{seg}$ + 视频处理器 $\mathcal{F}_{vid}$

**MLLM 关键帧选择器 (核心)**:
- 均匀采样 ~8 个关键帧候选
- 对每个候选帧自动合成一系列 CoT 问题：从通用语义 → 时间相关 → 细节
- 最终输出：目标实例列表 + 对应关键帧 + 帧内目标描述
- 支持 GPT-4o / Gemma3 / LLaVA1.5

**推理图像分割**: 使用 Seg-Zero 在选定关键帧上生成 key mask

**视频处理器**: 使用 SAM2 将 key mask 沿时间轴追踪生成完整掩码序列

**在线扩展**: 每 $\xi$ 帧周期性地用 CoT 更新关键帧，支持流式视频

## 实验关键数据
| 方法 | MeViS J&F | Refer-DAVIS J&F | ReasonVOS J&F |
|------|-----------|-----------------|---------------|
| VISA-13B | 44.5 | 70.4 | - |
| SAMWISE | 49.5 | 70.6 | - |
| VideoLISA (Po) | 44.4 | 68.8 | 47.5 |
| GLUS | 51.3 | - | 49.9 |
| **CoT-RVS-GPT-4o** | **52.2** | **79.1** | **65.5** |

- Refer-DAVIS-17 上 J&F 79.1，比 HyperSeg (71.2) 高出 +7.9
- ReasonVOS 上 J&F 65.5，比 GLUS (49.9) 高出 +15.6
- 开源版本 CoT-RVS-Gemma3-12B 也有竞争力（MeViS 44.2, Refer-DAVIS 74.6）
- 在时间敏感查询子集 T-ReasonVOS 上优势更显著

## 亮点
- **完全无训练**: 兼容闭源/开源 MLLM，无需微调任何模块
- **时间推理能力强**: CoT 过程让 MLLM 真正"思考"帧间时间语义关联
- **模块化可替换**: 分割模型(LISA/Seg-Zero)和视频处理器(SAM2/Cutie)可灵活替换
- **支持在线流式**: 少有的在线推理视频分割方案

## 局限性
- GPT-4o 版本推理成本高（API 调用），不适合大规模应用
- 开源版本(LLaVA/Gemma3)性能显著低于 GPT-4o
- 依赖均匀帧采样，可能错过关键时刻
- 多实例场景下实例间可能存在冲突需后处理

## 相关工作
- **VISA/VideoLISA**: 微调 MLLM 的推理 VOS 方法，是主要对比对象
- **Seg-Zero/ThinkFirst**: CoT 推理图像分割，本文扩展到视频域
- **SAM2**: 作为视频处理器的核心追踪模块
- **SAMWISE/GLUS**: 强 referring VOS 基线

## 评分
- 新颖性: ⭐⭐⭐⭐ (零样本 CoT 用于视频时间推理)
- 实验充分度: ⭐⭐⭐⭐ (4 个 benchmark + 消融 + 在线扩展)
- 写作质量: ⭐⭐⭐⭐ (表述清晰，示例生动)
- 价值: ⭐⭐⭐⭐ (无训练范式有实用意义，但依赖强MLLM)
