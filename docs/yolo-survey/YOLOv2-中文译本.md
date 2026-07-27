# YOLO9000：更好、更快、更强

**原文：** YOLO9000: Better, Faster, Stronger  
**作者：** Joseph Redmon\*†, Ali Farhadi\*†  
**单位：** University of Washington\*，Allen Institute for AI†  
**项目主页：** http://pjreddie.com/yolo9000/  
**原文链接：** https://arxiv.org/abs/1612.08242  

> 说明：本文为便于调研阅读的中文译本。  
> **公式以论文风格图片呈现**（见 `figures/yolov2/`）。  
> 文中 **YOLOv2** 指检测器改进版；**YOLO9000** 指在 YOLOv2 基础上做检测+分类联合训练、可检测 9000+ 类的系统。

---

## 摘要

我们提出 YOLO9000，一个可实时检测超过 9000 类物体的先进目标检测系统。首先，我们对 YOLO 检测方法提出多项改进（既有新方法，也吸收已有工作）。改进后的模型 **YOLOv2** 在 Pascal VOC、COCO 等标准检测任务上达到当时先进水平。借助新颖的多尺度训练，同一套 YOLOv2 模型可在不同输入尺寸下运行，从而在速度与精度间平滑权衡：67 FPS 时 VOC 2007 上达到 76.8 mAP；40 FPS 时达到 78.6 mAP，超过带 ResNet 的 Faster R-CNN 与 SSD，同时明显更快。

最后，我们提出一种在检测数据与分类数据上联合训练的方法，据此在 COCO 检测集与 ImageNet 分类集上同时训练 YOLO9000。联合训练使 YOLO9000 能对**没有检测标注**的类别也能给出检测框。在 ImageNet 检测验证集上，尽管 200 类中仅 44 类有检测标注，YOLO9000 仍取得 19.7 mAP；在与 COCO 不重叠的 156 类上取得 16.0 mAP。此外它还能检测远超 200 类、超过 9000 个不同类别，并保持实时。

**图 1：YOLO9000。** YOLO9000 能实时检测非常多样的物体类别。

---

## 1 引言

通用目标检测应当快速、准确，并能识别广泛物体。神经网络引入后，检测框架越来越快、越来越准，但多数方法仍受限于较小的类别集合。

相较分类、打标签等任务，当前检测数据集规模有限：常见检测集通常是数千到数十万张图、几十到几百类；分类集则有数百万图、数万到数十万类。

我们希望检测也能扩展到接近分类的规模。但检测标注远比分类/打标签昂贵，短期内很难出现与分类同规模的检测集。

因此我们提出：利用已有的大规模分类数据，扩展现有检测系统的能力。方法包括：
1. 用层次化分类视角合并不同数据集；
2. 设计联合训练算法，使检测器可同时使用检测标注与分类标注——用检测图学习精确定位，用分类图扩大词汇量并增强鲁棒性。

据此训练 YOLO9000：先改进基础 YOLO 得到先进实时检测器 YOLOv2，再用数据集合并与联合训练，在 ImageNet 的 9000+ 类与 COCO 检测数据上训练。

代码与预训练模型：http://pjreddie.com/yolo9000/

---

## 2 Better（更好）

相对当时先进检测系统，YOLO（v1）有明显短板。与 Fast R-CNN 的错误分析表明：YOLO **定位误差多**，且相对基于区域提议的方法 **召回率偏低**。因此本节重点提升召回与定位，同时尽量保持分类精度。

计算机视觉常趋向更大、更深网络，或靠集成提升性能。但 YOLOv2 希望“更准且仍快”：**不靠把网络做大，而是简化网络、让表征更容易学习**。我们汇集已有技巧并提出新想法，结果汇总见表 2。

### Batch Normalization

对所有卷积层加入 BN，收敛更好，并可去掉部分其他正则。YOLO 全卷积层加 BN 后 mAP 提升超过 **2%**；有 BN 后可去掉 dropout 且不过拟合。

### High Resolution Classifier（高分辨率分类器预训练）

先进检测器通常先在 ImageNet 上预训练分类器。自 AlexNet 以来，多数分类器输入小于 256×256。原 YOLO 在 224×224 上训分类器，检测时再升到 448，网络必须同时适应“检测任务 + 新分辨率”。

YOLOv2 先把分类网络在完整 **448×448** 上微调 10 个 epoch，让滤波器适应高分辨率，再去做检测微调。这一步带来接近 **4%** 的 mAP 提升。

### Convolutional With Anchor Boxes（卷积 + Anchor）

YOLO v1 用全连接层直接回归框坐标。Faster R-CNN 则用手工先验（anchor）：RPN 在特征图每个位置预测相对 anchor 的偏移与置信度。预测偏移通常比直接预测绝对坐标更容易学。

YOLOv2：
- 去掉全连接层，改用 **anchor boxes** 预测框；
- 去掉一层 pooling，提高卷积特征分辨率；
- 输入改为 **416×416**（不再是 448），使下采样 32 倍后特征图为 **13×13**（奇数尺寸，图像中心有单一中心格，利于大物体居中预测）。

引入 anchor 后，还将**类别预测与空间位置解耦**：每个 anchor 各自预测类别与 objectness。objectness 仍预测与 GT 的 IOU；类别预测仍是“有物体条件下的条件概率”。

效果：精度略降，但召回明显升——无 anchor 中间模型约 69.5 mAP / 81% recall；有 anchor 约 69.2 mAP / **88%** recall。召回上升意味着后续改进空间更大。每图预测框从约 98 个增加到 **1000+**。

### Dimension Clusters（维度聚类）

Anchor 有两个问题。第一：框尺寸常手工挑选。网络虽能学着调整，但更好的先验会让学习更容易。

我们不对宽高用欧氏距离做普通 k-means（大会框误差主导），而用与尺寸无关、直接服务 IOU 的距离：

![k-means 距离](figures/yolov2/eq_kmeans_dist.png)

对多个 k 画“与最近质心的平均 IOU”（图 2），取 **k=5** 作为复杂度与召回的折中。聚类得到的先验与手工 anchor 明显不同：更少“矮胖框”，更多“高瘦框”。

**表 1：VOC 2007 上框到最近先验的平均 IOU**

| 生成方式 | 数量 | Avg IOU |
|---|---:|---:|
| Cluster SSE | 5 | 58.7 |
| Cluster IOU | 5 | 61.0 |
| Anchor Boxes（手工） | 9 | 60.9 |
| Cluster IOU | 9 | 67.2 |

仅 5 个聚类先验就与 9 个手工 anchor 相当（61.0 vs 60.9）；用 9 个聚类先验则到 67.2。说明 k-means 先验能给模型更好起点。

**图 2：** 在 VOC/COCO 上对框尺寸聚类；左图为不同 k 的平均 IOU，右图为相对质心形状。

### Direct location prediction（直接位置预测）

第二个问题：早期训练不稳定，主要来自预测 `(x,y)`。RPN 一类写法预测 `t_x,t_y`，中心可写为：

![RPN 偏移示意](figures/yolov2/eq_rpn_offset.png)

例如 `t_x=1` 表示向右平移一个 anchor 宽度。该形式**无约束**，任意位置的预测都可能把框挪到图像任意处，随机初始化时很难尽快稳定。

YOLOv2 改回 YOLO 思路：相对**网格单元**预测中心，使真值落在 0–1，并用 logistic（sigmoid）把预测约束到该范围。

输出特征图每个格子预测 5 个框；每个框预测 5 个数：`t_x, t_y, t_w, t_h, t_o`。若格子相对图像左上角偏移为 `(c_x,c_y)`，先验宽高为 `p_w,p_h`，则：

![直接位置预测](figures/yolov2/eq_direct_loc.png)

约束中心位置后参数化更容易学、网络更稳。维度聚类 + 直接中心预测，相对“仅用 anchor”的版本大约再涨 **5%**。

**图 3：** 用维度先验与位置预测得到的边界框。宽高相对聚类质心；中心相对滤波器作用位置用 sigmoid 预测。

### Fine-Grained Features（细粒度特征 / passthrough）

此时检测在 13×13 特征图上进行，对大物体够用，小物体仍可能受益于更细特征。Faster R-CNN / SSD 会在多层特征上跑检测；YOLOv2 采用更简单做法：加 **passthrough**，把更早的 26×26 特征引入。

passthrough 把相邻特征堆到通道维（类似 ResNet identity 的拼接思想），将 26×26×512 变为 13×13×2048，再与原特征拼接。检测头作用在扩展后的特征上，约有 **1%** 提升。

### Multi-Scale Training（多尺度训练）

原 YOLO 固定 448；加 anchor 后常用 416。但模型只有卷积与池化，可动态改输入尺寸。为使模型对多种尺寸鲁棒：

- 每 10 个 batch 随机选新输入边长；
- 因下采样因子为 32，从 `{320,352,...,608}` 中选；
- 最小 320×320，最大 608×608，改尺寸后继续训练。

同一套权重可在不同分辨率推理，形成速度–精度滑杆：小尺寸更快更省；大尺寸更准。例如约 288×288 可超过 90 FPS，mAP 接近 Fast R-CNN；高分辨率在 VOC 2007 可达 **78.6 mAP** 且仍实时以上（见表 3、图 4）。

### 从 YOLO 到 YOLOv2 的路径

**表 2：多数设计都会显著提升 mAP。** 例外：换成全卷积+anchor 时 mAP 几乎不变但召回上升；换新网络则计算量约减 33%。

| 技术 | VOC2007 mAP（逐步） |
|---|---:|
| YOLO（基线） | 63.4 |
| + batch norm | 65.8 |
| + hi-res classifier | 69.5 |
| + convolutional | 69.2 |
| + anchor boxes | 69.2 |
| + new network | 69.6 |
| + dimension priors | 74.4 |
| + location prediction | 75.4 |
| + passthrough | 76.8 |
| + multi-scale / hi-res detector | **78.6** |

（表中勾选组合与原文 Table 2 一致；上表按论文叙述归纳关键节点。）

**表 3：Pascal VOC 2007（GTX Titan X）**

| 方法 | Train | mAP | FPS |
|---|---|---:|---:|
| Fast R-CNN | 07+12 | 70.0 | 0.5 |
| Faster R-CNN VGG-16 | 07+12 | 73.2 | 7 |
| Faster R-CNN ResNet | 07+12 | 76.4 | 5 |
| YOLO | 07+12 | 63.4 | 45 |
| SSD300 | 07+12 | 74.3 | 46 |
| SSD500 | 07+12 | 76.8 | 19 |
| YOLOv2 288 | 07+12 | 69.0 | 91 |
| YOLOv2 352 | 07+12 | 73.7 | 81 |
| YOLOv2 416 | 07+12 | 76.8 | 67 |
| YOLOv2 480 | 07+12 | 77.8 | 59 |
| YOLOv2 544 | 07+12 | **78.6** | 40 |

注意：表中各 YOLOv2 行是**同一套权重**在不同测试尺寸下的结果。

**表 4：VOC 2012 test**  
YOLOv2 544 取得 **73.4 mAP**，与 Faster R-CNN ResNet、SSD512 同级，速度却快约 2–10 倍。

**表 5：COCO test-dev2015**  
YOLOv2 在 AP@0.5 上 44.0，与 SSD/Faster R-CNN 可比；在更严的 AP@[.5:.95] 上 21.6，仍有差距（定位精细度仍是课题）。

---

## 3 Faster（更快）

检测既要准也要快。多数框架用 VGG-16 作骨干：强，但重——224×224 单次前向约 **30.69B** FLOPs。原 YOLO 定制 GoogLeNet 风格网络约 **8.52B** FLOPs，更快但 ImageNet top-5 略逊（88.0% vs VGG-16 的 90.0%）。

### Darknet-19

YOLOv2 新骨干 **Darknet-19**：
- 多为 3×3 卷积；每次 pooling 后通道加倍（类 VGG）；
- 用 1×1 压缩特征（类 NIN），全局平均池化做分类；
- 全面使用 BN。

结构：19 个卷积层 + 5 个 maxpool（见表 6）。约 **5.58B** 运算，ImageNet top-1 **72.9%**，top-5 **91.2%**。

**表 6：Darknet-19（分类版摘要）**

| Type | Filters | Size/Stride | Output |
|---|---:|---|---|
| Conv | 32 | 3×3 | 224×224 |
| Maxpool | | 2×2/2 | 112×112 |
| Conv | 64 | 3×3 | 112×112 |
| Maxpool | | 2×2/2 | 56×56 |
| Conv | 128 / 64 / 128 | 3×3, 1×1, 3×3 | 56×56 |
| Maxpool | | 2×2/2 | 28×28 |
| Conv | 256 / 128 / 256 | 3×3, 1×1, 3×3 | 28×28 |
| Maxpool | | 2×2/2 | 14×14 |
| Conv | 512 与 256 交替 | 3×3 / 1×1 | 14×14 |
| Maxpool | | 2×2/2 | 7×7 |
| Conv | 1024 与 512 交替 | 3×3 / 1×1 | 7×7 |
| Conv | 1000 | 1×1 | 7×7 |
| Avgpool + Softmax | | Global | 1000 |

### 训练细节

**分类：** ImageNet 1000 类，160 epoch，SGD，初始 lr=0.1，多项式衰减 power=4，weight decay=0.0005，momentum=0.9；标准增强。再在 448 分辨率微调 10 epoch（lr=1e-3），top-1/top-5 到 76.5% / 93.3%。

**检测：** 去掉最后分类卷积，加三个 3×3×1024 卷积，再接 1×1 输出检测所需通道。VOC 上每位置 5 个框、每框 5 坐标 + 20 类：

![VOC 输出通道](figures/yolov2/eq_output_filters.png)

并从最后 3×3×512 层拉 passthrough 到倒数第二层。检测训练 160 epoch，lr 从 1e-3 起，在 60/90 epoch 除以 10。

---

## 4 Stronger（更强：YOLO9000）

目标：同时用检测标注与分类标注训练。混合两类图像：
- 见到检测图：按完整 YOLOv2 损失反传；
- 见到分类图：只反传分类相关部分。

难点：检测标签粗（dog/boat），分类标签细（各种梗犬）。若对全部类做单一 softmax，会假设互斥——“Norfolk terrier”与“dog”不能简单并进同一扁平 softmax。

### Hierarchical classification 与 WordTree

ImageNet 标签来自 WordNet。例如 Norfolk terrier ⊂ terrier ⊂ hunting dog ⊂ dog ⊂ …  
我们从 ImageNet 视觉名词构建层次树 **WordTree**（把 WordNet 有向图简化为树）：先加入仅有单路径到根（physical object）的概念，再迭代加入“使树增长最少”的路径。

分类时，在每个节点预测其下位词的**条件概率**，例如在 terrier 节点：

![WordTree 条件概率](figures/yolov2/eq_wordtree_cond.png)

某节点绝对概率 = 沿根到该节点的条件概率连乘：

![WordTree 绝对概率](figures/yolov2/eq_wordtree_abs.png)

分类时假设图像含物体：`Pr(physical object)=1`。

用 WordTree1k（1000 类扩到含中间节点共 1369）训练 Darknet-19：top-1/top-5 约 71.9% / 90.4%，只略降。好处包括：对未知细类可优雅退化——不确定犬种时仍可高置信输出 “dog”。

检测时不再假设图中必有物体，而用 YOLOv2 的 objectness 作为 `Pr(physical object)`；再沿树取高置信分支向下走，直到阈值，得到类别。

**图 5：** ImageNet 大 softmax vs WordTree 多组同级 softmax。  
**图 6：** 用 WordTree 合并 ImageNet 与 COCO 标签。

### 联合训练 YOLO9000

合并 COCO 检测 + ImageNet 全量中 top 9000 类（并补上 ImageNet 检测挑战中缺失类），WordTree 约 **9418** 类。因 ImageNet 更大，对 COCO 过采样，使 ImageNet:COCO 大约 4:1。

结构基于 YOLOv2，但只用 **3 个先验**以控制输出规模。

- 检测图：正常反传；分类损失只在标签对应层级及以上计算（标成 “dog” 时，不强迫区分 Shepherd vs Retriever）。  
- 分类图：只反传分类损失——找到对该类预测概率最高的框，只在其上计算；并假设该框与“想象中的 GT”至少 0.3 IOU，据此反传 objectness。

结果：用 COCO 学“找物体/定位”，用 ImageNet 学“认更多类”。

### ImageNet 检测结果

YOLO9000 总体 **19.7 mAP**；对从未见过检测框的 156 类 **16.0 mAP**（弱监督色彩浓）。同时还能检测其余数千类，并保持实时。

**表 7：** 156 个弱监督类中最好/最差。动物类往往不错（objectness 易从 COCO 动物泛化）；服装/器械等差（COCO 几乎只有 person，没有衣物框）。

| 较差类（AP≈0） | 较好类（AP） |
|---|---|
| diaper, sunglasses, swimming trunks… | armadillo 61.7, tiger 61.0, koala 54.3… |

---

## 5 结论

我们提出 YOLOv2 与 YOLO9000。YOLOv2 在多个检测集上又快又强，并可多分辨率权衡速度与精度。YOLO9000 通过 WordTree 合并数据、联合优化检测与分类，朝缩小检测/分类数据规模鸿沟迈出一步。

多尺度训练、层次化标签等技术也可迁移到其他视觉任务。未来工作包括弱监督分割、更好的弱标签匹配等。

---

## 附录：核心公式速查

**聚类距离**

![k-means](figures/yolov2/eq_kmeans_dist.png)

**直接位置预测（YOLOv2）**

![direct loc](figures/yolov2/eq_direct_loc.png)

**WordTree 绝对概率**

![wordtree](figures/yolov2/eq_wordtree_abs.png)

**VOC 检测头通道数**

![filters](figures/yolov2/eq_output_filters.png)

---

## 参考文献

[1] S. Bell et al. Inside-outside net. arXiv:1512.04143, 2015.  
[2] J. Deng et al. ImageNet. CVPR, 2009.  
[3] M. Everingham et al. The PASCAL VOC Challenge. IJCV, 2010.  
[4] P. F. Felzenszwalb et al. DPM release 4.  
[5] R. B. Girshick. Fast R-CNN. 2015.  
[6] K. He et al. Deep residual learning. 2015.  
[7] S. Ioffe, C. Szegedy. Batch normalization. 2015.  
[8] A. Krizhevsky et al. ImageNet classification with deep CNNs. 2012.  
[9] M. Lin et al. Network in network. 2013.  
[10] T.-Y. Lin et al. Microsoft COCO. ECCV, 2014.  
[11] W. Liu et al. SSD. 2015.  
[12] G. A. Miller et al. WordNet. 1990.  
[13] J. Redmon. Darknet. 2013–2016.  
[14] J. Redmon et al. You only look once. 2015.  
[15] S. Ren et al. Faster R-CNN. 2015.  
[16] O. Russakovsky et al. ILSVRC. IJCV, 2015.  
[17] K. Simonyan, A. Zisserman. VGG. 2014.  
[18] C. Szegedy et al. Inception-v4. 2016.  
[19] C. Szegedy et al. Going deeper with convolutions. 2014.  
[20] B. Thomee et al. YFCC100M. CACM, 2016.
