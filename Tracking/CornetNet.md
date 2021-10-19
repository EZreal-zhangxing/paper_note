# CornerNet

## 1.Introduction

传统追踪使用基于Anchor的方法，来检测目标。用Anchor来截取出ROI(Region of interesting) 然后送入检测网络进行识别分类。但是基于Anchor的方法有两个缺点

1. 需要大量的Anchor。因为要训练检测器对Anchor进行分类是否和Groundtruth重叠。所以需要大量的数据。但是如果有大量的数据的话，势必导致正样本很少。正负样本的数量不平衡也会导致训练很慢
2. Anchor的引入会引入许多的超参数，例如锚点的尺寸，数量，长宽比等。在多尺度情况下会变得比较复杂

## 2.Corner pooling

两个featuremaps

1. 在每个像素点，对于每个通道，最大池化该像素点右侧的所有特征向量
2. 在每个像素点，对于每个通道，最大池化该像素点下侧的所有特征向量

根据上面求出的最大值做一个叠加，最后生成一个同大小的池化层 Featuremap $(N \times C \times Height \times Width)$

**优缺点**：

1. 比Anchor中心定位好，因为中心定位依赖四条边，而角点的定位只依赖两条边，因此相对而言会更简单
2. 角点池只需要$O(w\times h)$的角来代表$O(w^2 \times h^2)$的Anchor

### 1.Two-stage 算法

两阶段检测器（R-CNN,Fast-RCNN,Faster-RCNN），先生成一组ROI（Regions of Interest）区域，然后通过网路对每个ROI进行分类检测。早期的两阶段检测器（RCNN）会产生大量的冗余计算，再后来提出了Fast-RCNN对冗余计算进行了优化，在又来Faster-RCNN引入了区域提案网络（RPN）见[CS231n.md,Object Detection部分](../CS231n/CS231n.md)

### 2.One-stage 算法

一阶段检测器（例如，YOLO，SSD）消除了ROI，相比性能上能够和两阶段保持竞争力，同时有更高的计算效率

SSD密集的将锚框放置在多尺度的feature map上，直接对每个锚框进行分类。

YOLO则是直接通过图像预测边界框的坐标。然后转换成锚框。



