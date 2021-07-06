## SRC-sparse representation-based classifier

## 测试样本与训练样本的表示关系

我们对于第$$ i $$类测试样本取n个样本，每个样本有m个特征

给出如下的表达式：$$ A_i = [v_{i,1},v_{i,2},v_{i,3},\ldots v_{i,n}] \in R^{ m*n} \rightarrow \begin{pmatrix}v_{1,1}&v_{2,1}&\cdots &v_{n,1} \\ v_{1,2}&v_{2,2}&\cdots &v_{n,2}\\ \vdots & \vdots&\vdots &\vdots\\ v_{1,m}&v_{2,m}&\cdots &v_{n,m}  \end{pmatrix}$$ 这些样本是出于同一个线性子空间，对于给定的测试样本$$x$$

如果样本属于这个类别那么这个样本是可以被这个子空间线性表示的,
$$
x = \alpha_{i,1}*v_{i,1}+\alpha_{i,2}*v_{i,2}+\cdots+\alpha_{i,n}*v_{i,n}
$$
那么对于有K类的所有的训练样本可以有如下的定义：
$$
A = [A_1,A_2,\ldots,A_K] \sub R^{m*N}
$$
则对所有的测试样例可以有如下的表示：$$ y = Ax_0 \rightarrow x_o = [0,0,0\ldots,0;\alpha_1,\alpha_2,\alpha_3,\ldots,\alpha_n;0,0,0,\ldots,0] $$

$$x_0$$ 中除了属于的类是有参数的 其他的都是0

所以 分类问题等于求解 $y=Ax_0$ 

显然当$ m > N$时 如果存在解那么一般都是唯一的  但是当$m<N$时 方程可能有无数解

考虑后者情况使用最小化$l^2$范数来求解 $\mathop {argmin}||x||_2$

观察：

```ejs
A valid test sample y can be sufficiently represented using only the training samples from the same class. This representation is naturally sparse if the number of object classes k is reasonably large.
```

所以$x_0$越稀疏x的分类越准确，所以我们使用$l^0$范数来考虑 $\hat x = \mathop {argmin}||x_0||_0$

```
For Most Large Underdetermined Systems of Linear Equations the Minimal L1-norm Solution Is Also the Sparsest Solution DAVID L. DONOHO
```

如果$x_0$足够系数，$l^0$优化是等同于$l^1$优化的

## 稀疏表示的分类 

同样对于一个测试样本，在通过上述的求解后得到的系数向量，会有一大部分是属于本身类的系数，只会有少部分是属于其他类的系数，我们可以通过系数属于哪个类投票选择该测试样本是属于哪个类，也可以使用如下方法：

对每一个类$i$， $\delta_i$ 是选择该类的相关系数的函数，$\delta_i(x)$函数式用来求$x$系数中属于$i$类的非零值 并转换成一个系数向量

通过 下面这个式子来衡量 
$$
\min r_i(y) = ||y- A\delta_i(\hat x_1)||_2
$$
$y$ 是测试样本

![image-20210401140141198](.\image-20210401140141198.png)

可以看图片残差小的集中在某一类

![image-20210401140628460](.\image-20210401140628460.png)

### $\mathop {argmin}l^1,\mathop{argmin} l^2$的对比

![image-20210401140920760](.\image-20210401140920760.png)

可见$l^2$范式求出来的系数解并没有$l^1$求出来的稀疏

### 无关图片的求解特性

![image-20210401141316057](.\image-20210401141316057.png)

可以看到 有效的测试样本的系数应该大部分分布在一个类别上，而无关的测试样本会分散在大多数类上

为了衡量这个指标定义了一个SCI（sparsity concentration index）
$$
SCI(x) = {k \cdot max_i||\delta_i(x)||_1/||x||_1 -1 \over k-1} \in [0,1]
$$
$SCI$ 越大表明系数越集中在一个类别上，所以我们可以定义一个阈值 $\tau$

如果 $SCI(x) \ge \tau$ 即输出这个类别

