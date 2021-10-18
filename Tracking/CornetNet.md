# CornerNet

## 1.Introduction

传统追踪使用基于Anchor的方法，来检测目标。用Anchor来截取出ROI(Region of interesting) 然后送入检测网络进行识别分类。但是基于Anchor的方法有两个缺点

1. 需要大量的Anchor。因为要训练检测器对Anchor进行分类是否和Groundtruth重叠。所以需要大量的数据。但是如果有大量的数据的话，势必导致正样本很少。正负样本的数量不平衡也会导致训练很慢
2. 

