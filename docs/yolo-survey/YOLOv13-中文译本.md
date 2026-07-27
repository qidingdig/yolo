# YOLOv13：超图增强自适应视觉感知的实时检测（中文译本）

> 原文：*YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception*  
> 作者：Mengqi Lei, Siqi Li, Yihong Wu, Han Hu, You Zhou, Xinhu Zheng, Guiguang Ding, Shaoyi Du, Zongze Wu, Yue Gao 等（清华等）  
> 来源：[arXiv:2506.17733](https://arxiv.org/abs/2506.17733)（2025）  
> 代码：https://github.com/iMoonLab/yolov13  
> 说明：论文式中文整理译本；公式以图片嵌入。

**生产提示（Ultralytics）：** 官方曾提示 YOLO13 相对 YOLO11 增益有限、更大更慢，复现性需关注。本译本保留学术要点供精读。

---

## 摘要

YOLO11 及更早以卷积局部聚合为主；YOLOv12 的区域自注意力仍偏**成对（pairwise）**相关，难以刻画全局 **多对多高阶相关**。本文提出 **YOLOv13**：

1. **HyperACE**：自适应超图相关增强，突破手工超边阈值；  
2. **FullPAD**：全管线聚合—分发，把增强特征送回骨干/颈/头；  
3. **深度可分离高效块**：替换大核普通卷积，降参降算。

COCO 上 YOLOv13-N 相对 YOLO11-N / YOLOv12-N 约 **+3.0 / +1.5 mAP**。

---

## 1. 引言

卷积感受野受核与深度限制；区域注意力为控成本而局部化，且本质是全连接语义图上的二元相关。超图可用超边连接多顶点表达高阶相关，但既有视觉方法常靠**手工特征距离阈值**建边，鲁棒性差。

YOLOv13 目标：自适应挖掘跨位置、跨尺度高阶相关，并在整网高效分发。

---

## 2. 方法

### 2.1 总体：HyperACE + FullPAD

<p align="center">
  <img src="figures/yolov13/eq_arch.png" alt="YOLOv13 arch" width="720" />
</p>

流程概要：

1. Backbone（含 **DS-C3k2** 等轻量块）提多尺度特征 \(B_1\ldots B_5\)；  
2. 将 \(B_3,B_4,B_5\) 送入 **HyperACE** 做跨尺度高阶相关建模与增强；  
3. **FullPAD** 经多条隧道把增强特征分发到：骨干—颈连接、颈内部、颈—头连接；  
4. 多尺度检测头输出。

### 2.2 HyperACE

<p align="center">
  <img src="figures/yolov13/eq_hyperace.png" alt="HyperACE" width="680" />
</p>

- 顶点：多尺度特征图像素（特征）；  
- **自适应超边**：学习每个顶点对每条超边的连续参与度 \(A\in[0,1]^{N\times M}\)，而非 \(\{0,1\}\) 手工阈值；  
- **超图消息传递**：线性复杂度聚合，引导跨位置/跨尺度融合；  
- 同时保留基于 DS-C3k 的**低阶局部分支**，高低阶互补。

超边数 \(M\)：N/S/L/X 典型取 4 / 8 / 8 / 12。

### 2.3 FullPAD

<p align="center">
  <img src="figures/yolov13/eq_fullpad.png" alt="FullPAD" width="680" />
</p>

突破传统「骨干→颈→头」单向信息流：相关增强特征被分发到全管线，改善表示协同与梯度传播。消融显示：去掉 HyperACE 或只通一条隧道都会掉点。

### 2.4 轻量 DS 块

用深度可分离系列块替换大核普通卷积：N/S 上 AP 几乎不掉，FLOPs/参数明显下降（论文报告 N：约 -1.1G / -0.6M）。

---

## 3. 实验（摘要）

训练约 600 epoch、640、与 YOLO11/v12 对齐设定；T4 TensorRT FP16 测延迟。

| 模型 | FLOPs | 参数 | AP<sup>val</sup> | Latency (ms) |
|------|-------|------|------------------|--------------|
| YOLO11-N | 6.5 | 2.6 | 38.6* | 1.53 |
| YOLOv12-N | — | — | ~40.1 量级 | — |
| **YOLOv13-N** | **6.4** | **2.5** | **41.6** | **1.97** |
| **YOLOv13-S** | **20.8** | **9.0** | **48.0** | **2.98** |

\*表中 YOLO11-N 数字以 YOLOv13 论文对照表为准，或与 Ultralytics 文档略有出入。

相对 YOLOv12：N/S/L/X 约 **+1.5 / +0.9 / +0.4 / +0.4** AP。

---

## 4. 结论

YOLOv13 把 YOLO 的相关建模从「局部 / 成对」推进到 **自适应超图高阶相关**，并用 **FullPAD** 打通全管线信息流，在保持轻量的同时提升复杂场景检测。

---

## 译本说明

以 arXiv:2506.17733 为准；生产选型请结合 Ultralytics 建议与自行复现。
