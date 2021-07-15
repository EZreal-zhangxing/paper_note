## 矩阵求导

对于矩阵$A$ 向量$\vec x$有如下
$$
Y= A \vec x = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \cdots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn} \\
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix} = 
\begin{bmatrix}
a_{11}x_1 + a_{12}x_2+\ldots+a_{1n}x_n \\
a_{21}x_1 + a_{22}x_2+\ldots+a_{2n}x_n \\
\vdots \\
a_{m1}x_1 + a_{m2}x_2+\ldots+a_{mn}x_n
\end{bmatrix} 记为：
\begin{bmatrix}
y_1\\
y_2\\
\vdots \\
y_m
\end{bmatrix}.M \ne N
$$
则用$Y$对$\vec x$求偏导有：将$Y$的每一个元素分别对$\vec x$的每一个元素求导
$$
{\part y \over \part \vec x} = \begin{pmatrix}
{\part y \over \vec x_1} \\
{\part y \over \vec x_2} \\
\vdots \\
{\part y \over \vec x_n} \\
\end{pmatrix} 记为 \begin{pmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{pmatrix}
\tag{1}
$$
而式子$(1)$中的每个元素对元素$x_n$求导有如下式子：
$$
y_n = \begin{pmatrix}
{\part y_1 \over \part x_1} \\
{\part y_2 \over \part x_1} \\
\vdots \\
{\part y_m \over \part x_1} \\
\end{pmatrix} \tag{2}
$$
将式子$(2)$带入计算得到求导结果
$$
{\part y \over \part \vec x} = \begin{pmatrix}
{\part y_1 \over \vec x_1} \\
{\part y_1 \over \vec x_2} \\
\vdots \\
{\part y_1 \over \vec x_n} \\
\ldots \\
{\part y_m \over \vec x_1} \\
{\part y_m \over \vec x_2} \\
\vdots \\
{\part y_m \over \vec x_n} \\
\end{pmatrix} \tag{3}
$$
**不过为了方便我们在实践中应用，通常情况下即使$y$向量是列向量也按照行向量来进行求导**

式子$(3)$化简成
$$
{\part y \over \part \vec x} = \begin{pmatrix}
{\part y_1 \over \vec x_1} & {\part y_2 \over \vec x_1} & \cdots & {\part y_m \over \vec x_1} \\
{\part y_1 \over \vec x_2} & {\part y_2 \over \vec x_2} & \cdots & {\part y_m \over \vec x_2} \\
\vdots & \vdots & \vdots & \vdots\\
{\part y_1 \over \vec x_n} & {\part y_2 \over \vec x_n} & \cdots & {\part y_m \over \vec x_n} \\
\end{pmatrix} = \begin{pmatrix}
a_{11} & a_{21} \cdots & a_{n1} \\
a_{12} & a_{22} \cdots & a_{n2} \\
\vdots & \vdots & \vdots\\
a_{1m} & a_{2m} \cdots & a_{nm} \\
\end{pmatrix} = A^T
$$
得到几个公式推广：
$$
{\part A \vec x \over \part \vec x} = A^T \\
{\part A \vec x \over \part \vec x^T} = A \\
{\part \vec x A  \over \part \vec x} = A
$$
矩阵求导可以参考连接: [矩阵求导](https://zhuanlan.zhihu.com/p/273729929)

