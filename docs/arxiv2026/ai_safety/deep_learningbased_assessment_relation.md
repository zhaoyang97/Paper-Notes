# Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning

**会议**: arXiv 2026  
**arXiv**: [2603.11850](https://arxiv.org/abs/2603.11850)  
**作者**: Johan Andreas Balle Rubak, Sara Haghighat, Sanyam Jain, Mostafa Aldesoki, Akhilanand Chaurasia
**代码**: 待确认  
**领域**: 隐私/安全/公平  
**关键词**: deep, learning-based, assessment, relation, between  

## 一句话总结
靠近下颌管的下颌第三磨牙的阻生会增加下牙槽神经损伤的风险。
## 背景与动机
Impaction of the mandibular third molar in proximity to the mandibular canal increases the risk of inferior alveolar nerve injury.. Panoramic radiography is routinely used to assess this relationship.

## 核心问题
靠近下颌管的下颌第三磨牙的阻生会增加下牙槽神经损伤的风险。全景X光检查通常用于评估这种关系。
## 方法详解

### 整体框架
- Panoramic radiography is routinely used to assess this relationship.
- Automated classification of molar-canal overlap could support clinical triage and reduce unnecessary CBCT referrals, while federated learning (FL) enables multi-center collaboration without sharing patient data.
- We compared Local Learning (LL), FL, and Centralized Learning (CL) for binary overlap/no-overlap classification on cropped panoramic radiographs partitioned across eight independent labelers.

### 关键设计
1. **关键组件1**: Impaction of the mandibular third molar in proximity to the mandibular canal increases the risk of inferior alveolar nerve injury.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 在测试集上，CL 实现了最高性能（AUC 0.831；准确度 = 0.782），FL 显示了中等性能（AUC 0.757；准确度 = 0.703），LL 在客户端之间的泛化能力较差（AUC 范围 = 0.619-0.734；平均值 = 0.672）。
- 总体而言，集中训练提供了最强的性能，而 FL 提供了优于 LL 的隐私保护替代方案。
## 局限性 / 可改进方向
- 待深读后补充

## 与相关工作的对比
待深读后补充。

## 与我的研究方向的关联
- 待分析

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐
