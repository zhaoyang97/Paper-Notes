# SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design

**会议**: ICRA 2026 (arXiv标注为CVPR2026列表)  
**arXiv**: [2603.13098](https://arxiv.org/abs/2603.13098)  
**代码**: 匿名仓库(论文中提及)  
**领域**: 3D生成 / CAD建模 / 多模态数据集  
**关键词**: CAD generation, multimodal dataset, parametric modeling, SolidWorks, language-driven 3D design  

## 一句话总结
构建SldprtNet——含242K+工业CAD零件的大规模多模态数据集，每个样本包含.sldprt/.step模型、7视角合成图、参数化建模脚本(13种命令无损编解码)和Qwen2.5-VL生成的自然语言描述，baseline实验验证多模态输入(图+文)在CAD生成上优于纯文本输入。

## 背景与动机
语言驱动的CAD建模(Text-to-CAD)处于起步阶段。现有CAD数据集面临多重局限：(1) ShapeNet/ModelNet只有网格/点云，不保留参数化建模历史；(2) ABC数据集有B-Rep但无文本标注和建模序列；(3) Fusion 360 Gallery有建模历史但仅限sketch+extrude，无语言标注；(4) DeepCAD/Text2CAD只支持2种命令类型且文本为合成生成，与真实几何可能不对齐。核心问题：缺少一个集成了3D模型、多视角图像、参数化指令序列和自然语言描述的大规模多模态数据集来支撑Text-to-CAD研究。

## 核心问题
如何构建一个满足多模态、双向可表示、语义标注、可编辑和人类可读的CAD数据集？如何实现CAD模型与文本的无损双向转换以支持数据集扩展？多模态输入对CAD生成的效果提升有多大？

## 方法详解

### 整体框架
数据构建Pipeline：(1) 从GrabCAD、McMaster-Carr、FreeCAD收集~680K .sldprt文件 → (2) 过滤保留含13种特征类型的242K+模型 → (3) SolidWorks Macro渲染7视角图(6正交+1等轴)合成为单张PNG → (4) Encoder工具提取参数化文本(Encoder_txt) → (5) 转换为.step格式 → (6) 用Qwen2.5-VL-7B对图像+参数文本生成自然语言描述(Des_txt)。最终每个样本有5个对齐模态：.sldprt、.step、.png、Encoder_txt、Des_txt。

### 关键设计
1. **Encoder/Decoder工具(13种CAD命令)**: 基于SolidWorks COM API实现。Encoder遍历.sldprt的Feature Tree，按建模历史顺序提取特征类型、名称、父子关系和详细参数(尺寸、约束、草图实体等)，输出结构化人类可读文本。Decoder读取文本反向重建.sldprt，实现无损闭环。支持的13种命令：2D Sketch、Extrusion、Chamfer、Fillet、Linear Pattern、Mirror Pattern等，远超DeepCAD的2种(sketch+extrude)，覆盖真实工业设计的大部分操作。

2. **7视角合成图**: 6个正交视图(前后左右上下) + 1个等轴视图合并为单张图，充分捕获3D几何信息。关键优势：相比7张独立图片，单张合成图在VLM推理时大幅减少输入token数，加速inference。

3. **Qwen2.5-VL-7B生成自然语言描述**: 输入合成图+参数化文本，让VLM生成描述零件外观和功能的自然语言。用12块A100 GPU跑368 GPU-hours完成242K+样本的描述生成。视觉编码器帮助模型捕获孔型图案、轮廓线、长宽比等细节，使描述与几何高度对齐。人工校验确保质量。

### 损失函数 / 训练策略
Baseline使用Qwen2.5-7B和Qwen2.5-7B-VL在50K子集上微调。评价指标包括：Exact Match Score、BLEU Score、Command-Level F1、Tolerance Accuracy、Partial Match Rate。

## 实验关键数据
| 指标 | Qwen2.5-7B (纯文本) | Qwen2.5-7B-VL (图+文) |
|------|---------------------|----------------------|
| Exact Match Score | 0.0058 | **0.0099** |
| BLEU Score | 97.18 | **97.93** |
| Command-Level F1 | 0.3247 | **0.3670** |
| Partial Match Rate | 0.5554 | **0.6162** |
| Tolerance Accuracy | **0.5016** | 0.4630 |

### 消融实验要点
- **多模态 > 纯文本**: Exact Match提升71%，Command F1提升13%，Partial Match提升11%，验证了视觉模态对理解几何语义和建模逻辑的增益
- **Tolerance Accuracy纯文本略优**: 可能是纯文本模型对数值参数过拟合，而缺乏结构语义理解
- **数据集复杂度分布合理**: Level 1(1-5特征) 93K，Level 2(6-10) 79K，Level 3(11-100) 69K，Level 4(100+) 1.2K，支持课程学习

### 数据集与现有工作对比
| 特性 | SldprtNet | ABC | ShapeNet | DeepCAD | Text2CAD |
|------|-----------|-----|----------|---------|----------|
| 规模 | 242K | 1M+ | 3M+ | 170K | 170K |
| 参数化 | ✓ | ✓ | × | ✓ | ✓ |
| 多视图 | ✓ | × | ✓ | × | × |
| 可重建 | ✓ | × | × | × | × |
| 自然语言描述 | ✓ | × | × | × | ✓(合成) |
| 命令种类 | 13 | - | - | 2 | 2 |

## 亮点 / 我学到了什么
- **Encoder/Decoder闭环是关键贡献**: 实现CAD↔文本的无损双向转换，使数据集可扩展且支持生成结果的验证。这种"code as intermediate representation"的思路在Text-to-3D领域会越来越重要
- **7视角合成图减少token的工程技巧**: 简单但有效——把多张图拼成一张，在VLM推理时避免多图token爆炸
- **Exact Match极低(0.006~0.01)说明CAD生成仍极具挑战**: 即使在结构化、确定性的CAD命令生成任务上，当前LLM的精确匹配率依然很低，说明这个领域还有很大空间
- **工业级真实零件 vs 合成数据**: 相比Omni-CAD等使用合成数据的数据集，SldprtNet来自GrabCAD等真实工程部件，更能反映实际设计需求

## 局限性 / 可改进方向
- **评估指标不充分**: 没有3D几何层面的评估(如Chamfer Distance、IoU)，仅从文本匹配角度评估CAD生成，无法真正衡量生成模型的质量
- **仅测试了50K子集**: 242K数据集的全量训练效果未知
- **依赖SolidWorks**: Encoder/Decoder通过SolidWorks COM API实现，绑定商业软件，不利于开源社区复现和扩展
- **自然语言描述由VLM自动生成**: 虽然声称人工校验，但242K的人工校验工作量存疑
- **论文实际被ICRA 2026接收**: 标注了ICRA，但出现在CVPR2026论文列表中，可能是分类错误

## 与相关工作的对比
- **vs DeepCAD/Text2CAD**: 核心差距在命令种类(13 vs 2)和多模态对齐。DeepCAD只有sketch+extrude，SldprtNet覆盖fillet/chamfer/pattern等工业常用操作，更接近真实设计
- **vs CAD-GPT/CAD-MLLM/CAD-Coder**: 这些是模型方案，SldprtNet是数据集方案，互为补充。SldprtNet的编解码器可为这些模型提供更丰富的训练数据
- **vs ABC**: ABC有1M+ B-Rep模型但无文本标注和建模序列，无法用于语言驱动生成

## 与我的研究方向的关联
- Text-to-CAD是3D生成的重要分支，但与多模态VLM和视觉理解的主流方向距离较远
- "参数化表示 + LLM生成 + 解码器重建"的范式可能对其他结构化生成任务(电路设计、分子生成等)有启发
- 数据集构建的方法论（多模态对齐、自动标注+人工校验）可参考

## 评分
- 新颖性: ⭐⭐⭐ 核心是数据集和工具，技术创新有限，但填补了重要空白
- 实验充分度: ⭐⭐⭐ 只有baseline对比(纯文本 vs 多模态)，缺少与其他CAD生成方法的对比和3D评估
- 写作质量: ⭐⭐⭐⭐ 数据集描述全面，分析系统，但引用的一些论文与CAD数据集关系不大(NeRF, SwinTransformer等)
- 对我的价值: ⭐⭐ CAD生成非核心关注方向，但多模态数据集构建思路有参考价值
