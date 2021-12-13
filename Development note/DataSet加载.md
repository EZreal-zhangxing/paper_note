# 数据集的加载相关知识

## Dataset

主要用于加载数据集，做数据集的处理，返回一个图片还有对于的标签

```python
class selfDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(selfDataset,self).__init__()
        '''
        your code
        '''
      	pass
   	def __item__(self):
        '''
        your code
        返回图片，还有标签数据
        '''
        pass
   	def __len__(self):
        '''
        返回数据集的长度
        '''
        pass
```

## Sample

采样器，主要是定义从数据集中的采样规则

```python
class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group #这里是创建一个group的生成器

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
```



## Dataloader

数据集加载类，用于加载数据集

其中一个参数为 **collate_fn**  该参数是传入一个函数，将一个batch打包成一个大的Tensor。

对于coco数据集，因为每个图片的大小不统一，标签bbox个数不统一，因此可以用这个参数。

定义一个函数对于采样得到的一批数据进行对齐

```python
def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()
	# 取该批次图片中宽高最大的值创建一个图片
    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
		# 对图片进行复制
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:
		# 对齐标签
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        # 对齐标签
        annot_padded = torch.ones((len(annots), 1, 5)) * -1
	# 更改维度为 B*C*H*W
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}
```

其中collate_fn的函数输入参数是dataset中getitem返回的一个批次的数据的集合形式，

而定义成函数的时候，要传入一些其他参数有两种解决方案

### 使用lamda表达式

```python
info = args.info	# info是已经定义过的
loader = Dataloader(collate_fn=lambda x: collate_fn(x, info))
```

### 创建成一个可被调用的类

```python
class collater():
	def __init__(self, *params):
		self. params = params
	
	def __call__(self, data):
		'''在这里重写collate_fn函数'''
```

