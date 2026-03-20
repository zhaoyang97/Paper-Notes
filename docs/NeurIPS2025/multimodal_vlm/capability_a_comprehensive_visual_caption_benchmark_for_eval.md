# CAPability: A Comprehensive Visual Caption Benchmark for Evaluating Both Correctness and Thoroughness

**会议**: NeurIPS 2025  
**arXiv**: [2502.14914](https://arxiv.org/abs/2502.14914)  
**代码**: 有  
**领域**: 多模态VLM / Benchmark / 图像描述  
**关键词**: visual captioning, benchmark, correctness, thoroughness, precision, hit, know-but-cannot-tell  

## 一句话总结
提出CAPability，一个全面的多视角视觉描述benchmark，跨6个关键视角12个维度评估MLLM生成caption的正确性（precision）和全面性（hit），用近11K人工标注的图像视频，并引入"知道但说不出"（K/T̄）指标揭示模型在QA和caption之间的显著能力差距。

## 背景与动机
随着MLLM的进步，传统的图像描述benchmark（基于简短GT句子和BLEU/CIDEr等指标）已经过时——无法评估模型生成的详细长描述的质量。近期benchmark尝试通过关键词提取或物体中心评估来改进，但仍限于粗粒度分析和不完整的视觉元素覆盖。缺乏一个同时评估"说得对不对"（正确性/precision）和"说得全不全"（全面性/hit/recall）的综合benchmark。

## 核心问题
如何全面、稳定、多维度地评估MLLM的视觉描述能力？正确性和全面性应该如何分别衡量？

## 方法详解

### 整体框架
CAPability设计了12个评估维度（跨6个关键视角），用近11K张人工标注的图像和视频构建视觉元素注释。评估时将模型生成的caption与注释对比，同时计算precision（正确性——说的对不对）和hit（全面性——有没有漏说）。

### 关键设计
1. **12维度6视角的全面评估**：不只看"有哪些物体"（object-centric），而是从6个视角（如物体属性、空间关系、动作交互、场景氛围、文字信息、时间动态等）评估12个维度。这提供了对MLLM描述能力的360度诊断。

2. **Precision + Hit双指标**：Precision衡量"caption中描述的内容有多大比例是正确的"——检测幻觉和错误描述；Hit衡量"标注的视觉元素有多大比例被caption提到了"——检测遗漏。两个指标互补——只看precision可能高分但遗漏很多，只看hit可能全面但错误也多。

3. **"Know but Cannot Tell"（K/T̄）指标**：将标注转换为QA对，发现模型在QA格式下能正确回答的内容在caption生成时却常常遗漏——即"知道但说不出"。这个差距（K/T̄）量化了模型的描述能力瓶颈——不是感知不到，而是生成时不主动描述。

### 损失函数 / 训练策略
纯评估benchmark，无训练。

## 实验关键数据
- **近11K人工标注**的图像和视频
- **12维度**的全面评估
- K/T̄指标揭示QA能力与caption能力之间的**显著差距**
- 识别了各MLLM在不同维度上的具体优势和弱点

## 亮点
- **Precision + Hit的双指标设计**是对单一F1类指标的重要改进——分别诊断幻觉和遗漏
- **"知道但说不出"（K/T̄）是非常有价值的发现**——揭示了MLLM的描述生成瓶颈不在于理解而在于主动表达
- **12维度6视角**提供了前所未有的细粒度诊断——帮助研究者精确定位模型弱点
- 覆盖图像和视频双模态
- 近11K高质量人工标注

## 局限性 / 可改进方向
- 人工标注成本高，难以大规模扩展
- 12维度的评估可能增加分析复杂度
- 标注者偏好可能影响评估结果

## 与相关工作的对比
- **vs. CHAIR**：CHAIR只衡量物体幻觉（precision侧）；CAPability同时衡量正确性和全面性
- **vs. ChartMuseum**：ChartMuseum专注图表的视觉推理；CAPability专注通用视觉描述的全面性
- **vs. DetailCaps**：DetailCaps关注描述的详细程度；CAPability从更多维度评估

## 启发与关联
- K/T̄发现暗示需要在训练中鼓励模型"主动描述更多内容"——这可能通过RL奖励全面性来实现

## 评分
- 新颖性: ⭐⭐⭐⭐ precision+hit双指标和K/T̄发现有创新
- 实验充分度: ⭐⭐⭐⭐ 11K标注，多维度分析
- 写作质量: ⭐⭐⭐⭐ 设计理念清晰
- 价值: ⭐⭐⭐⭐ 为视觉描述评估提供了更全面的工具
