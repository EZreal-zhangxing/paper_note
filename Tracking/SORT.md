# SORT

使用卡里曼滤波器和匈牙利算法的组合达到了SOTA(state-of-the-art)的跟踪器效果

### 1、卡里曼滤波

利用不确定性的动态信息来对系统下一步的走向做出有根据的预测。

### 例子

对于一个物体 $\hat x_k$ 我们使用速度 $v$ 和位置 $p$ 来描述他的状态 $p_k$ 为他的协方差矩阵
$$
\hat x_k = 
\begin{bmatrix}
position \\
velocity
\end{bmatrix},
p_k = \begin{bmatrix}
\sum_{pp} & \sum_{pv} \\
\sum_{vp} & \sum_{vv}
\end{bmatrix}
$$
使用基本的运动学公式来表示下一个点的状态
$$
P_k = P_{k-1}+\Delta t v_{k-1} \\
v_k = v_{k-1} 
$$
将上式化成矩阵表示形式：
$$
\hat x_k = \begin{bmatrix}
position \\
velocity
\end{bmatrix} 
= \begin{bmatrix}
1 & \Delta t \\
0 & 1
\end{bmatrix} \cdot
\begin{bmatrix}
P_{k-1} \\
v_{k-1}
\end{bmatrix} = F_k \hat x_{k-1} \tag{3}
$$
这个就是目标的预测矩阵，通过上一时刻(k-1)的状态得到下一时刻的状态。

同时更新协方差矩阵 我们对每个状态点乘以矩阵$A$ 能够得到如下:
$$
Cov(x,x) = Var(x) =  \sum \\
Cov(Ax,Ax) = A\sum A^T \tag{4}
$$
结合方程$(4)$ 和方程$(3)$ 可以得到：
$$
\hat x_k = F_k \hat x_{k-1} \\
P_k =Cov(F_k X_{k-1}) =  F_k P_{k-1}F_k^T  \tag{5}
$$
但由于有外部因素对系统产生干扰，我们可以假设一个加速度$a$
$$
p_k = p_{k-1}+\Delta t v_{k-1} +{1 \over 2} a \Delta t^2 \\
v_k = v_{k-1} + a \Delta t \\
$$
表示成矩阵的话
$$
\hat x_k = F_k \hat x_{k-1} + \begin{bmatrix}
\Delta t^2 \over 2 \\
\Delta t
\end{bmatrix} a \\
= F_k \hat x_{k-1}+B_k \vec {u_k}   \tag{6}
$$
其中$B_k$ 称为控制矩阵，$\vec {u_k}$称为控制向量

原始估计中的每个状态变量 更新到新的状态后，任然服从高斯分布。我们可以说$X_{k-1}$的每个状态移动到了一个新的服从高斯分布的区域，协方差为$Q_k$ 所以产生了不同的协方差矩阵 我们可以将$(5)$更新为如下：
$$
\hat x_k = F_k \hat x_{k-1} + B_k \vec {u_k} \\
P_k =Cov(F_k X_{k-1}) =  F_k P_{k-1}F_k^T +Q_k  \tag{7}
$$
可以看出来**新的最优估计**是更具**上一最优估计**预测得到的,并且加上了已知外部控制量的修正

我们用传感器得到的值是满足一个高斯分布 记为$Z_1$

用上一最优估计估计得到当前的预测值也满足一个高斯分布 记为$Z_2$

那么我们只需要将这两个高斯分布相乘既可以得到一个新的高斯分布这个分布就是两个分布最可能的值，也就是最优估计了

### 融合高斯分布

一维高斯分布：
$$
\N(x,\mu,\delta) = {1 \over \delta \sqrt {2 \pi}}e^{- {(x-\mu)^2 \over 2 \delta ^2}}
$$
两个服从高斯分布的函数相乘可以得到：
$$
N_1(x_1,\mu_1,\delta_1) \cdot \N_2(x_2,\mu_2,\delta_2)
= {1 \over \delta_1 \delta_2 2 \pi} e^{-({(x_1-\mu_1)^2 \over 2\delta_1^2}+{(x_2-\mu_2)^2 \over 2\delta_2^2})} 
$$
重新归一化可以得到：
$$
\mu = \mu_1 + {\delta_1^2(\mu_2 -\mu_1) \over \delta_1^2+\delta_2^2} \\
\delta^2 = \delta_1^2 - {\delta_1^4 \over \delta_1^2+\delta_2^2} \tag{11}
$$
令$K = {\delta_1^2 \over \delta_1^2+\delta_2^2}$ 那么就可以得到
$$
\mu' = \mu_1 + K(\mu_2-\mu_1) \\
\delta'^2 = \delta_1^2 - K\delta_1^2 \tag{13}
$$

$$
转换成矩阵形式：K = \sum_1(\sum_1 + \sum_2)^{-1} \tag{14}
$$


$$
\vec\mu = \vec \mu_1 +K(\vec \mu_2-\vec \mu_1) \\
\sum' = \sum_1 - K\sum_1 \tag{15}
$$


矩阵$K$被称为卡尔曼增益

我们得到预测部分
$$
(\mu_0,\sum_0) = (H_k\hat x_k,H_kP_kH_k^T)
$$
测量部分
$$
(\mu_1,\sum_1) = (\vec z_k,R_k)
$$
将上式带入$(15)$算出它们之间的重叠部分
$$
H_K \hat x_k' = H_k \hat x_k + K(\vec z_k - H_k \hat x_K) \\
H_kP'_kH_K^T = H_kP_kH_K^T - KH_kP_kH_k^T \tag{16}
$$
由式$(14)$可得卡尔曼增益为：
$$
K = H_kP_kH_k^T(H_kP_kH_k^T+R_k)^{-1} \tag{17}
$$
将$(17)$带入$(16)$ 左右两边同乘$H_k^{-1}$ 得：
$$
\hat x'_k =\hat x_k+K'(\vec z_k- H_k \hat x_k) \\
P'_k = P_k - K'H_kP_k \\
其中 K' = P_kH_k^T(H_kP_kH_k^T+R_k)^{-1}
$$
$x'_k$就是新的最优估计，将$x'_k$和$P'_k$放到下一个预测和更新方程中不断迭代

其中$H_k\hat x_k$ 为$k$时刻的预测坐标

$H_kP_kH_k^T$ 为预测值的协方差矩阵

$\vec z_k$ 为测量的坐标

##### 1.协方差矩阵

各个向量中的每个元素之间的协方差所构成的矩阵，设$X = (X_1,X_2,\dots,X_N)^T$ 则矩阵
$$
C = \begin{pmatrix}
c_{11} & c_{12} & \dots & c_{1n} \\
c_{21} & c_{22} & \dots & c_{2n} \\
\vdots & \vdots & \vdots & \vdots \\
c_{n1} & c_{n2} & \dots & c_{nn} \\
\end{pmatrix} 其中：
c_{ij} = Cov(X_i,X_j)
$$
协方差矩阵的性质
$$
Cov(X,Y) = Cov(Y,X)^T \\
Cov(AX+b,Y) = ACov(X,Y),其中A是矩阵，b是向量\\
Cov(X+Y,Z) = Cov(X,Z)+Cov(Y,Z) \\
Cov(AX) = ACov(X)A^T
$$


##### 2.协方差

衡量两个变量的总体误差，方差是协方差的一种特殊情况——两个变量相同的情况
$$
\begin{array}{rcl}
Cov(X,Y) & = & E[(X-E(X))(Y-E(Y))] \\
& = & E[XY] - 2E[Y]E[X]+E[X][Y] \\
& = & E[XY] - E[X]E[Y]
\end{array}
$$
如果两个变量的变化趋势一致，也就是说如果其中一个大于自身的期望值时另外一个也大于自身的期望值，那么两个变量之间的协方差就是正值；如果两个变量的变化趋势相反，即其中一个变量大于自身的期望值时另外一个却小于自身的期望值，那么两个变量之间的协方差就是负

如果*X*与*Y*是统计独立的，那么二者之间的协方差就是0，因为两个独立的随机变量满足*E*[*XY*]=*E*[*X*]*E*[*Y*]。

协方差的性质：
$$
Cov(X,Y) = Cov(Y,x) \\
Conv(aX,bY) = abCov(X,Y) \\ 
Cov(X_1+X_2,Y) = Cov(X_1,Y) + Cov(X_2,Y) \\
Cov(X+a,Y+b) = Cov(X,Y)
$$


##### 3.方差

衡量随机变量或一组数据的离散程度的度量，即度量随机变量和其数学期望之间的偏离程度
$$
D(x) = E[X^2]-E^2[X]
$$
方差越大变量变化越剧烈



### 2、匈牙利算法

利用回溯 对两个集合进行匹配的问题，参考 [匈牙利算法&KM算法](https://zhuanlan.zhihu.com/p/62981901)



传统的多目标检测（MOT）：多假设追踪（MHT）、联合概率数据关联过滤器（Joint Probabilistic Data Associ-
ation (JPDA) filters）

### Detection

使用了FrRCNN（Fast Region CNN）:第一阶段提取特征并为第二阶段提取区域。然后在提出的区域对对象进行分类，对输出概率大于50%的检测结果传递给跟踪框架

对比FrRCNN和ACF检测器时，检测质量对跟踪性能有显著的影响。

使用一个线性等速模型来近似每个物体的帧间位移，建模为：$x = [\mu,\nu,s,r,\dot \mu,\dot \nu,\dot s]$

$\mu$ 代表水平像素的位置，$\nu$代表垂直像素的位置，$s$ 目标边界框的面积  $r$边界框的横纵比 。

检测到的目标框用于更新目标状态，其中速度分量通过卡尔曼滤波器求解

### Creation and Deletion of Track

创建跟踪器：对于任何检测到的重叠框，如果 小于 $IOU_{\min}$  就被认为存在未被跟踪的目标

然后使用**速度设置为0**的 Bounding box初始化追踪器，因为这点上的速度不可观测，所以速度分量的**协方差**用**最大值**初始化。

新的跟踪器需要经过试用期，在试用期中需要与检测相关联已积累足够的证据，从而防止FP目标



如果在$T_{Loss}$个帧中没有探测到目标，检测将会被终止。这防止了在没有探测器校正的情况下，跟踪器数量无限制增长。我们一般将$T_{Loss}$设为1有两个愿意





