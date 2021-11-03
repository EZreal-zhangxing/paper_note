# 非极大值抑制（non maximum suppression）

算法思想很简单，按照分类概率排序，概率最高的框作为候选框，其它所有与它的IOU高于一个阈值（这是人工指定的超参）的框其概率被置为0。然后在剩余的框里寻找概率第二大的框，其它所有与它的IOU高于一个阈值（这是人工指定的超参）的框其概率被置为0。依次类推。

```python
import numpy as np
def py_cpu_nms(bboxs, thresh):
    """Pure Python NMS baseline."""
    dets = []
    for item in bboxs:
        bbox = item['bbox']
        score = item['score']
        bbox.append(score)
        dets.append(bbox)
    dets = np.array(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 对参数从小到大排序，然后用切片逆序 变成从大到小的
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        '''
        用得分最大的左上，和右下的坐标作为基准
        和其他框的左上，右下点进行比较，
        对于左上角的点，得到坐标最大的
        对于右下角的点，得到坐标最小的
        '''
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 得到最小的长和宽
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        # 计算面积
        inter = w * h
        # 优化后的面积 / 用得分最大的框的面积+其他框的面积-优化后的面积
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 阈值比较 一直到所有的面积比大于阈值
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

bboxs = [{'score': 0.53439176, 'class': 15, 'ct':[429.  , 341.25], 'tracking':[-0.85791016,  3.130371  ], 'bbox':[421.658  , 340.77545, 439.34854, 348.69495]},
         {'score': 0.47248393, 'class': 15, 'ct':[572.    , 296.5625], 'tracking':[0.03570557, 8.581055  ], 'bbox':[565.36475, 299.62994, 583.12244, 307.41077]},
         {'score': 0.40402642, 'class': 15, 'ct':[732.875 , 296.5625], 'tracking':[1.8614502, 2.1117554], 'bbox':[726.1142 , 292.06567, 746.0963 , 304.9978 ]},
         {'score': 0.23502168, 'class': 34, 'ct':[732.875 , 296.5625], 'tracking':[1.8614502, 2.1117554], 'bbox':[726.1142 , 292.06567, 746.0963 , 304.9978 ]},
         {'score': 0.22896464, 'class': 34, 'ct':[563.0625, 305.5   ], 'tracking':[8.669067 , 1.3942261], 'bbox':[561.2625 , 300.20123, 579.99493, 314.1026 ]},
         {'score': 0.22578397, 'class': 5, 'ct':[563.0625, 305.5   ], 'tracking':[8.669067 , 1.3942261], 'bbox':[561.2625 , 300.20123, 579.99493, 314.1026 ]}, {'score': 0.22443116, 'class': 34, 'ct':[429.  , 341.25], 'tracking':[-0.85791016,  3.130371  ], 'bbox':[421.658  , 340.77545, 439.34854, 348.69495]}, {'score': 0.19851686, 'class': 5, 'ct':[732.875 , 296.5625], 'tracking':[1.8614502, 2.1117554], 'bbox':[726.1142 , 292.06567, 746.0963 , 304.9978 ]}, {'score': 0.18461587, 'class': 5, 'ct':[429.  , 341.25], 'tracking':[-0.85791016,  3.130371  ], 'bbox':[421.658  , 340.77545, 439.34854, 348.69495]}, {'score': 0.12083422, 'class': 9, 'ct':[563.0625, 305.5   ], 'tracking':[8.669067 , 1.3942261], 'bbox':[561.2625 , 300.20123, 579.99493, 314.1026 ]}, {'score': 0.11675542, 'class': 1, 'ct':[429.  , 341.25], 'tracking':[-0.85791016,  3.130371  ], 'bbox':[421.658  , 340.77545, 439.34854, 348.69495]}, {'score': 0.116644554, 'class': 1, 'ct':[572.    , 296.5625], 'tracking':[0.03570557, 8.581055  ], 'bbox':[565.36475, 299.62994, 583.12244, 307.41077]}, {'score': 0.111599885, 'class': 30, 'ct':[429.  , 341.25], 'tracking':[-0.85791016,  3.130371  ], 'bbox':[421.658  , 340.77545, 439.34854, 348.69495]}]
print(len(bboxs))
print(py_cpu_nms(bboxs,0.53))
```

## Reference

[目标检测中的检测框合并策略：NMS和Soft-NMS](https://blog.csdn.net/yuanlulu/article/details/89762861)

