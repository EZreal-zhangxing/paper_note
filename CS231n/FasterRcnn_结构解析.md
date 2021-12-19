# Faster RCNN

## RPN

```python
# B*C*W*H --> B*d*(C*W/d)*H
@staticmethod
def reshape_layer(x, d):
    input_shape = x.size()
    # x = x.permute(0, 3, 1, 2)
    # b c w h
    x = x.view(
        input_shape[0],
        int(d),
        int(float(input_shape[1] * input_shape[2]) / float(d)),
        input_shape[3]
    )
    # x = x.permute(0, 2, 3, 1)
    return x
```

```python
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class RPN(nn.Module):
    _feat_stride = [16, ]
    anchor_scales = [8, 16, 32]

    def __init__(self):
        super(RPN, self).__init__()
		
        self.features = VGG16(bn=False) # backbone
        # 输入512,输出512卷积 + BatchNormalization/Relu
        # same_padding = True 输出尺寸不变使用 (kernel_size-1)/2 进行填充
        # same_padding = False 输出尺度会进行变化 in_channel*w*h --> out_channel * ((in_size+padding*2-F)/stride + 1)
        self.conv1 = Conv2d(512, 512, 3, same_padding=True) 
        # 输入512,输出 Anchor * 3 * 2(有目标的概率和没有目标的概率)
        self.score_conv = Conv2d(512, len(self.anchor_scales) * 3 * 2, 1, relu=False, same_padding=False) 
        # 输入512,输出 Anchor * 3 * 4(四个坐标值)
        self.bbox_conv = Conv2d(512, len(self.anchor_scales) * 3 * 4, 1, relu=False, same_padding=False)

        # loss
        self.cross_entropy = None
        self.los_box = None

    def forward(self, im_data, im_info, gt_boxes=None, gt_ishard=None, dontcare_areas=None):
        # 图像转换成pytorch变量
        im_data = network.np_to_variable(im_data, is_cuda=True)
        # 交换维度 变成 N*C*W*H
        im_data = im_data.permute(0, 3, 1, 2)
        # 送入 backbone 例 3*1024*1024 --> 512*64*64
        # VGG下采样16倍
        features = self.features(im_data)
	    # 512*64*64 --> 512*64*64
        rpn_conv1 = self.conv1(features)

        # rpn score
        rpn_cls_score = self.score_conv(rpn_conv1) # 512*64*64 -->(Anchor*3*2) * 64 * 64
        # 把通道对半分，分别用softmax计算每个类别的概率
        rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score, 2) # (Anchor*3*2) * 64 * 64 --> 2* ((Anchor*3*2*64)/2) * 64
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape) # 得到类别的概率分布 1*2*576*64
        # reshape 概率分布 2* ((Anchor*3*2*64)/2) * 64 --> (Anchor*3*2) * 64 * 64
        rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob, len(self.anchor_scales)*3*2) 

        # rpn boxes
        rpn_bbox_pred = self.bbox_conv(rpn_conv1) # 512*64*64 -->(Anchor*3*4) * 64 * 64

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info,
                                   cfg_key, self._feat_stride, self.anchor_scales)

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None
            rpn_data = self.anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas,
                                                im_info, self._feat_stride, self.anchor_scales)
            self.cross_entropy, self.loss_box = self.build_loss(rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)

        return features, rois
```

```python
# rpn_cls_prob_reshape 概率分布的reshape  (Anchor*3*2) * 64 * 64
# rpn_bbox_pred bbox的特征图 (Anchor*3*4) * 64 * 64
# im_info, cfg_key, _feat_stride, anchor_scales 相关参数 图像信息?，‘TRAIN?TEST’,特征stride,Anchor的大小
def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales):
    rpn_cls_prob_reshape = rpn_cls_prob_reshape.data.cpu().numpy()
    rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()
    x = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales)
    x = network.np_to_variable(x, is_cuda=True)
    return x.view(-1, 5)
```

```python
def proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride=[16, ],
                   anchor_scales=[8, 16, 32]):
    """
    Parameters
    ----------
    rpn_cls_prob_reshape: (1 , H , W , Ax2) outputs of RPN, prob of bg or fg
                         NOTICE: the old version is ordered by (1, H, W, 2, A) !!!!
    rpn_bbox_pred: (1 , H , W , Ax4), rgs boxes output of RPN
    im_info: a list of [image_height, image_width, scale_ratios]
    cfg_key: 'TRAIN' or 'TEST'
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_rois : (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)
    #layer_params = yaml.load(self.param_str_)

    """
    # 更具Anchor_scales 生成Anchor
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]
    # rpn_cls_prob_reshape = np.transpose(rpn_cls_prob_reshape,[0,3,1,2]) #-> (1 , 2xA, H , W)
    # rpn_bbox_pred = np.transpose(rpn_bbox_pred,[0,3,1,2])              # -> (1 , Ax4, H , W)

    # rpn_cls_prob_reshape = np.transpose(np.reshape(rpn_cls_prob_reshape,[1,rpn_cls_prob_reshape.shape[0],rpn_cls_prob_reshape.shape[1],rpn_cls_prob_reshape.shape[2]]),[0,3,2,1])
    # rpn_bbox_pred = np.transpose(rpn_bbox_pred,[0,3,2,1])
    im_info = im_info[0]

    assert rpn_cls_prob_reshape.shape[0] == 1, \
        'Only single item batches are supported'
    # cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
    # cfg_key = 'TEST'
    pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N # nms 处理之前的Anchor的个数 12000
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N # nms 处理之后的Anchor的个数 2000
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH # nms 阈值 0.7
    min_size = cfg[cfg_key].RPN_MIN_SIZE # box的阈值 16

    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    scores = rpn_cls_prob_reshape[:, _num_anchors:, :, :]
    bbox_deltas = rpn_bbox_pred
    # im_info = bottom[2].data[0, :]

    if DEBUG:
        print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
        print 'scale: {}'.format(im_info[2])

    # 1. Generate proposals from bbox deltas and shifted anchors
    height, width = scores.shape[-2:]

    if DEBUG:
        print 'score map size: {}'.format(scores.shape)

    # Enumerate all shifts
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    anchors = _anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

    # Same story for the scores:
    #
    # scores are (1, A, H, W) format
    # transpose to (1, H, W, A)
    # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

    # Convert anchors into proposals via bbox transformations
    proposals = bbox_transform_inv(anchors, bbox_deltas)

    # 2. clip predicted boxes to image
    proposals = clip_boxes(proposals, im_info[:2])

    # 3. remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    keep = _filter_boxes(proposals, min_size * im_info[2])
    proposals = proposals[keep, :]
    scores = scores[keep]

    # # remove irregular boxes, too fat too tall
    # keep = _filter_irregular_boxes(proposals)
    # proposals = proposals[keep, :]
    # scores = scores[keep]

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    keep = nms(np.hstack((proposals, scores)), nms_thresh)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]
    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
    return blob
    # top[0].reshape(*(blob.shape))
    # top[0].data[...] = blob

    # [Optional] output scores blob
    # if len(top) > 1:
    #    top[1].reshape(*(scores.shape))
    #    top[1].data[...] = scores
```



```python
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
	
    base_anchor = np.array([1, 1, base_size, base_size]) - 1 #Anchor [x_start,y_start,x_end,y_end]
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    # 进行Scale规模扩大
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) 
                         for i in xrange(ratio_anchors.shape[0])]) 
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    返回按照宽高比计算出来的Anchor列表
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor) 
    size = w * h
    size_ratios = size / ratios # 按比例缩放后的面积大小
    ws = np.round(np.sqrt(size_ratios)) # 按比例缩放后的 宽
    hs = np.round(ws * ratios) # 对应的高
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr) #返回左上角和右下角的bbox
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
   	返回[宽，高，中点x坐标，中点y坐标]
    """
    w = anchor[2] - anchor[0] + 1 # Anchor 的宽
    h = anchor[3] - anchor[1] + 1 # Anchor 的高
    x_ctr = anchor[0] + 0.5 * (w - 1) # 中点x
    y_ctr = anchor[1] + 0.5 * (h - 1) # 中点y
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    返回Anchor的左上角点，和右下角点坐标[左上角x,左上角y，右下角x，右下角y]
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    # 返回左上角和右下角的点
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
	对每个Anchor进行scales的扩张返回Anchor的左上角和右下角的坐标
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
```

