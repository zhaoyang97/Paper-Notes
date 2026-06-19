---
title: >-
  [论文解读] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors
description: >-
  [NeurIPS 2025][多模态VLM][MLLM] VG LLM提出将3D视觉几何编码器（VGGT）集成到多模态大语言模型中，仅从视频输入（无需显式3D数据）即可提取和融合3D几何先验，在3D场景理解和空间推理任务上显著提升MLLM性能，4B模型在VSI-Bench上超越Gemini-1.5-Pro。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "MLLM"
  - "3D视觉几何"
  - "空间推理"
  - "视频理解"
  - "VGGT"
---

# Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors

**会议**: NeurIPS 2025  
**arXiv**: [2505.24625](https://arxiv.org/abs/2505.24625)  
**代码**: [GitHub](https://lavi-lab.github.io/VG-LLM)  
**领域**: 3D视觉 / 多模态大模型  
**关键词**: MLLM, 3D视觉几何, 空间推理, 视频理解, VGGT

## 一句话总结

VG LLM提出将3D视觉几何编码器（VGGT）集成到多模态大语言模型中，仅从视频输入（无需显式3D数据）即可提取和融合3D几何先验，在3D场景理解和空间推理任务上显著提升MLLM性能，4B模型在VSI-Bench上超越Gemini-1.5-Pro。

## 研究背景与动机

多模态大语言模型(MLLM)在2D图像和视频理解上取得了显著进展，但在3D空间推理上仍面临困难。现有将MLLM应用于3D场景理解的方法存在一个共同限制：**依赖显式3D数据输入**。

核心问题链：
1. 方法如Video-3D LLM需要深度图/点云图，GPT4Scene需要从重建的3D点云渲染BEV图
2. 这类3D数据在很多真实场景中难以获取
3. 从图像直接估计3D属性会引入估计误差，降低性能
4. 传统视觉编码器将视频帧独立编码为token，丢失了帧间3D几何信息（如对应关系）

核心idea：**利用预训练的3D视觉几何模型（如VGGT）作为额外编码器，从视频中隐式提取3D几何先验，无需任何显式3D数据输入**。

## 方法详解

### 整体框架

VG LLM在标准MLLM基础上增加一个3D视觉几何编码器分支：
- 输入图像序列 → 2D视觉编码器提取语义特征 + 3D几何编码器提取几何特征
- 两路特征在patch级别融合 → 送入MLLM主干生成回答

### 关键设计

1. **3D视觉几何编码器**:

    - 采用VGGT-1B作为几何编码器，VGGT是在点云预测等3D任务上预训练的模型
    - VGGT包含三个部分：per-image编码器、跨帧融合解码器、任务预测头
    - 本文仅使用编码器和融合解码器（不用预测头），提取包含3D几何先验的特征
    - 关键能力：VGGT能捕获帧间对应关系，其隐特征能恢复3D场景结构

2. **视觉特征融合**:

    - 2D编码器输出 $T_i^{V'} \in \mathbb{R}^{\lfloor h/2p \rfloor \times \lfloor w/2p \rfloor \times c}$（经Qwen2.5-VL的2×2空间合并压缩）
    - 3D几何特征 $T_i^G$ 经相同的空间合并策略（拼接相邻2×2 patch + 两层MLP）变换为 $T_i^{G'}$
    - 最终特征：$T_i^S = T_i^{G'} + T_i^{V'}$，简单的加法融合
    - 融合后与文本嵌入拼接送入MLLM主干

3. **多任务3D场景理解**:

    - **3D视觉定位**：给定语言查询，输出帧索引+3D定向包围盒 $(x,y,z,w,h,d,\psi,\theta,\phi)$
    - **3D密集描述**：给定3D物体中心坐标，生成该物体的详细描述
    - **3D视频目标检测**：在统一坐标系下检测视频中所有出现的物体
    - 所有任务统一为文本生成，用下一token预测目标训练

4. **空间推理增强训练**:

    - 使用SPAR-7M数据集（仅采样3%即234K样本）进行空间推理指令微调
    - 混合LLaVA-Video-178K的LLaVA-Hound子集保持通用能力
    - 以第一帧坐标系作为基准坐标系，所有坐标转换到此系下

### 损失函数 / 训练策略

- 标准下一token预测损失（交叉熵）
- 训练时冻结：2D视觉编码器、3D几何编码器、多模态连接器
- 仅训练MLLM主干（LoRA或全参数微调）
- 基于Qwen2.5-VL的3B和7B版本 + VGGT-1B
- 8×H100 GPU，3D场景理解训练9-12h，空间推理训练7-9h

## 实验关键数据

### 主实验

| 任务/基准 | 指标 | VG LLM-8B | 对比模型 | 说明 |
|----------|------|-----------|---------|------|
| ScanRefer (3D定位) | Acc@0.25 | **41.6** (57.6*) | Video-3D LLM: 58.1 | *含proposal精化；无3D输入 |
| Scan2Cap (3D描述) | C@0.5 | **80.0** | Video-3D LLM: 80.0 | 持平SOTA，仅用RGB输入 |
| 3D视频检测 (4帧) | F1@0.25 | **41.2** | Qwen2.5-VL-7B: 32.5 | +8.7%绝对提升 |
| VSI-Bench (空间推理) | Avg | **50.7** | Gemini-1.5-Pro: 45.4 | 4B版本47.3即超越 |

### 消融实验

| 配置 | F1 (3D检测) | 说明 |
|------|-----------|------|
| Qwen2.5-VL-3B (无几何) | 30.0 | 基线 |
| + VGGT几何 (VG LLM-4B) | **38.2** | +8.2%，几何先验有效 |
| Qwen2.5-VL-7B (无几何) | 32.5 | 更大基线 |
| + VGGT几何 (VG LLM-8B) | **41.2** | +8.7%，一致性提升 |

| 融合策略 | VSI-Bench Avg | 说明 |
|---------|--------------|------|
| 仅2D视觉 | 基线 | 无3D先验 |
| 加法融合 (ours) | **50.7** | 简单有效 |

### 关键发现

- **无需显式3D输入**仍可达到甚至超越需要3D数据的方法，证明3D几何先验可从视频中隐式学习
- 3D几何增强对**自我中心-他中心转换**能力提升尤为显著（3D视频检测F1提升10.7%）
- VG LLM-4B在VSI-Bench上得分47.3%，超越最佳商用模型Gemini-1.5-Pro的45.4%
- 模型对帧数变化具有鲁棒性：4帧训练可直接在6帧上推理且性能不明显下降
- 仅使用SPAR-7M 3%的数据即可获得显著的空间推理提升

## 亮点与洞察

- **架构极简但高效**：只需在标准MLLM旁边加一个冻结的3D几何编码器，加法融合即可带来巨大收益
- **VGGT的隐式3D建模**：VGGT预训练的帧间对应关系建模能力是核心——它让MLLM不必自己从raw token推断3D结构
- **数据高效**：仅234K空间推理数据（SPAR-7M的3%）即可在VSI-Bench上达到SOTA
- **统一文本生成框架**：所有3D任务（定位、描述、检测）统一为文本生成，无需任务特定头

## 局限与展望

- VGGT-1B在推理时增加了计算开销，尤其对长视频序列
- 3D视觉定位的绝对精度仍低于使用真实3D数据的方法（41.6% vs 58.1%）
- 加法融合可能不是最优的特征融合方式，交叉注意力或门控机制可能进一步提升
- 坐标系以第一帧为基准，对大范围运动场景可能不理想

## 相关工作与启发

- **vs Video-3D LLM**: Video-3D LLM将3D坐标注入patch级视觉特征，需要显式深度输入；VG LLM纯视频输入，用隐式几何编码替代
- **vs GPT4Scene**: GPT4Scene需要3D重建生成BEV图，VG LLM无需任何重建步骤
- **vs SPAR**: SPAR通过合成数据增强空间推理能力但仅关注数据侧，VG LLM从模型架构层面引入几何能力

## 评分

- 新颖性: ⭐⭐⭐⭐ 将3D视觉几何模型作为MLLM的辅助编码器是自然但有效的想法，但融合方式比较简单
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖3D场景理解、空间推理、通用基准等多个维度，消融和分析详尽
- 写作质量: ⭐⭐⭐⭐ 动机清晰，实验展示全面，结构规范
- 价值: ⭐⭐⭐⭐⭐ 证明了无需显式3D数据即可增强MLLM的3D理解能力，应用前景广泛

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)
- [\[NeurIPS 2025\] BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models](bridgevla_input-output_alignment_for_efficient_3d_manipulation_learning_with_vis.md)
- [\[ICCV 2025\] AdvDreamer Unveils: Are Vision-Language Models Truly Ready for Real-World 3D Variations?](../../ICCV2025/multimodal_vlm/advdreamer_unveils_are_visionlanguage_models_truly_ready_for.md)
- [\[NeurIPS 2025\] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)
- [\[NeurIPS 2025\] Text to Robotic Assembly of Multi Component Objects using 3D Generative AI and Vision Language Models](text_to_robotic_assembly_of_multi_component_objects_using_3d_generative_ai_and_v.md)

</div>

<!-- RELATED:END -->
