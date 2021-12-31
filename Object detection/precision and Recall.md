# mAP计算 ReCall 计算

- True Positive(真正, TP)：将正类预测为正类数. **正确归类的正类** **（真阳性）**
- True Negative(真负 , TN)：将负类预测为负类数.**正确归类的负类** **（真阴性）**
- False Positive(假正, FP)：将负类预测为正类数 $\rightarrow$ **误报** (Type I error). **错误归类的负类，即负类归到正类 （假阳性）** 
- False Negative(假负 , FN)：将正类预测为负类数 $\rightarrow$ **漏报** (Type II error),**错误归类的正类，即正类归到负类（假阴性）**

P/N，表示正类和负类，T/F表示结果正确还是错误

```html
假设我们手上有60个正样本，40个负样本，我们要找出所有的正样本，系统查找出50个，其中只有40个是真正的正样本，计算上述各指标。

TP: 将正类预测为正类数  40 真阳性
FN: 将正类预测为负类数  20 --> 60-40 假阴性
FP: 将负类预测为正类数  10 --> 50-40 假阳性
TN: 将负类预测为负类数  30 --> 40-10 真阴性
```

精确率和召回率又被称为**查准率**和**查全率**，

查准率＝检索出的相关信息量 / 检索出的信息总量
查全率＝检索出的相关信息量 / 系统中的相关信息总量

Precision（精确率）: ${TP \over (TP+FP)} = {TP \over {all_{detection}}}$ ，查准率  = $检索出的相关信息量 \over 检索出的信息总量$

Accuracy (准确率) : $TP+TN \over TP+TN+FP+FN$

Recall: ${TP \over (TP+FN) }={TP \over all_{groundtruth}} $ ，查全率 = $检索出的相关信息量 \over 系统中的相关信息总量$

mAP: mean Average Precision, 即各类别AP的平均值

AP: PR曲线下面积，其实是在0～1之间所有recall值的precision的平均值。

PR曲线: Precision-Recall曲线