# CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection

**会议**: ICLR2026  
**arXiv**: [2602.22621](https://arxiv.org/abs/2602.22621)  
**代码**: [GitHub](https://github.com/Michael-McQueen/CGSA)  
**领域**: 目标检测 / 无源域自适应  
**关键词**: source-free domain adaptation, object-centric learning, slot attention, DETR, contrastive learning  

## 一句话总结
首次将 Object-Centric Learning（Slot Attention）引入无源域自适应目标检测（SF-DAOD），通过分层 Slot 感知模块提取结构先验，并用类引导对比学习驱动域不变表征。

## 背景与动机
1. 目标检测器部署时面临域偏移（天气/摄像头/场景变化），性能大幅下降
2. 无源域自适应（SF-DAOD）：仅有源域预训练模型和无标注目标域数据，无法访问源数据
3. 现有方法主要关注伪标签阈值调优或 teacher-student 框架改进，忽略了跨域的目标级结构共性
4. Slot Attention 可将场景分解为独立的目标表征，天然适合提取域不变结构先验
5. DETR 已使用 object queries，嵌入 slot 先验是自然但未探索的方向

## 方法详解
**框架**: 源域预训练 + 目标域自适应（Teacher-Student）

**HSA（分层 Slot 感知）模块**:
- 两阶段分解：第一阶段提取 n=5 个粗粒度 slot，第二阶段细化为 n²=25 个细粒度 slot
- Slot Attention 迭代聚合 + 空间广播 MLP 解码重建，softmax 竞争确保 slot 绑定不同区域
- 投影后与 object queries 相加，形成 slot-aware queries 输入 decoder

**CGSC（类引导 Slot 对比）模块**:
- 维护 EMA 更新的全局类原型 P_c
- 用注意力 mask 加权聚合 slot → weighted slot → 匈牙利匹配分配伪标签
- InfoNCE 对比损失拉近同类 slot-原型、推远异类

**总损失**: L_total = L_unsup（伪标签检测）+ λ_con·L_con + λ_rec·L_rec

## 实验关键数据
**Cityscapes → BDD100K**:
| 方法 | SF | mAP |
|------|-----|-----|
| DATR (DAOD) | ✗ | 43.3 |
| TITAN (SF-DAOD) | ✓ | 38.3 |
| **CGSA (Ours)** | ✓ | **53.0** |

**Cityscapes → Foggy-Cityscapes**: CGSA mAP 显著优于所有 SF-DAOD 方法
- 基于 RT-DETR 检测器，4×A100 训练
- 在多个跨域场景（正常→雾天，真实→合成等）均取得 SOTA

## 亮点
- 首次将 OCL/Slot Attention 引入 SF-DAOD，开创新范式
- 分层 slot 设计突破传统 slot 数量限制（≤10 → 25）
- 提供理论泛化分析（目标域风险下降界）
- 在有源和无源设置下都超越现有方法

## 局限性 / 可改进方向
- 仅在驾驶场景数据集验证，更多领域（医疗/航拍等）待测试
- Slot 数量 n=5 是手动设定，可考虑自适应机制
- 匈牙利匹配依赖检测器预测质量，early stage 可能不稳定
- 计算开销：HSA 的两阶段 Slot Attention + 重建目标增加训练成本

## 与相关工作的对比
- **SFOD/PETS/A2SFOD**: 聚焦伪标签过滤，忽略目标级结构；CGSA 利用 slot 结构先验
- **DATR/MRT** (有源 DAOD): 需要源数据，CGSA 在无源设置下仍超越
- **Slot Attention/SAVi**: 用于分割/视频预测，首次用于域自适应检测

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (OCL + SF-DAOD 的首创结合)
- 实验充分度: ⭐⭐⭐⭐ (5个数据集，完整消融)
- 写作质量: ⭐⭐⭐⭐ (动机清晰，理论+实验双支撑)
- 价值: ⭐⭐⭐⭐ (为 SF-DAOD 开辟新方向)
