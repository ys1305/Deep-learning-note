

我们用下面的记号来描述一个前馈神经网络：
层数一般只考虑隐藏层和输出层。

- $L$：表示神经网络的层数；

-  $m^{(l)}​$：表示第 $l ​$层神经元的个数；

- $f_l(·)：​$表示 $l ​$层神经元的激活函数；

- $ W^{(l)} ∈ R^{m^{(l)}×m^{(l-1)}}：$表示 $l - 1$层到第 $l $层的权重矩阵；

-  $b^{(l)} ∈ R^{m^{(l)}}：​$表示$ l - 1​$层到第 $l ​$层的偏置；

-  $z^{(l)}∈ R^{m^{(l)}}​$：表示$ l ​$层神经元的净输入（净活性值）；

- $a^{(l)} ∈ R^{m^(l)}$：表示 ​$l $层神经元的输出（活性值）。 

- $W^{l}_{jk}$表示从网络第$(l−1) $层中第$k$个神经元指向第$l$ 层中第$j $个神经元
     个神经元的连接权重
    --------------------- 
    作者：痴澳超 
    来源：CSDN 
    原文：https://blog.csdn.net/u014303046/article/details/78200010 
    版权声明：本文为博主原创文章，转载请附上博文链接！

前馈神经网络通过下面公式进行信息传播， 
$$
\begin{aligned} \mathbf{z}^{(l)} &=W^{(l)} \cdot \mathbf{a}^{(l-1)}+\mathbf{b}^{(l)} \\ \mathbf{a}^{(l)} &=f_{l}\left(\mathbf{z}^{(l)}\right)\\
\mathbf{z}^{(l)}&=W^{(l)} \cdot f_{l-1}\left(\mathbf{z}^{(l-1)}\right)+\mathbf{b}^{(l)}\\
\mathbf{a}^{(l)}&=f_{l}\left(W^{(l)} \cdot \mathbf{a}^{(l-1)}+\mathbf{b}^{(l)}\right)
\end{aligned}
$$
前馈神经网络可以通过逐层的信息传递，得到网络最后的输出 
$$
\mathbf{x}=\mathbf{a}^{(0)} \rightarrow \mathbf{z}^{(1)} \rightarrow \mathbf{a}^{(1)} \rightarrow \mathbf{z}^{(2)} \rightarrow \cdots \rightarrow \mathbf{a}^{(L-1)} \rightarrow \mathbf{z}^{(L)} \rightarrow \mathbf{a}^{(L)}=\varphi(\mathbf{x} ; W, \mathbf{b}) )
$$
交叉熵损失函数 
$$
\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})=-\mathbf{y}^{\mathrm{T}} \log \hat{\mathbf{y}}
$$

## 反向传播算法

### 为什么定义误差?

$$
\delta^{(l)}=\frac{\partial \mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})}{\partial \mathbf{z}^{(l)}} \in \mathbb{R}^{m^(l)}
$$

因为需要更新的参数$W^{(l)},b^{(l)}$都只和关键节点$\mathbf{z}^{(l)}$相关
$$
\begin{aligned} \frac{\partial \mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})}{\partial W_{i j}^{(l)}} &=\left(\frac{\partial \mathbf{z}^{(l)}}{\partial W_{i j}^{(l)}}\right)^{\mathrm{T}} \frac{\partial \mathcal{L}(\mathbf{y}, \hat{\boldsymbol{y}})}{\partial \mathbf{z}^{(l)}} \\ \frac{\partial \mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})}{\partial \mathbf{b}^{(l)}} &=\left(\frac{\partial \mathbf{z}^{(l)}}{\partial \mathbf{b}^{(l)}}\right)^{\mathrm{T}} \frac{\partial \mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})}{\partial \mathbf{z}^{(l)}} \end{aligned}
$$
1 
$$
\frac{\partial \mathbf{z}^{(l)}}{\partial W_{i j}^{(l)}}=\frac{\partial\left(W^{(l)} \mathbf{a}^{(l-1)}+\mathbf{b}^{(l)}\right)}{\partial W_{i j}^{(l)}}\\
=\left[ \begin{array}{c}
{\frac{\partial\left(W^{(l)}_{1:} \mathbf{a}^{(l-1)}+\mathbf{b}^{(l)}\right)}{\partial W_{i j}^{(l)}}} 
\\ {\vdots} 
\\ {\frac{\partial\left(W^{(l)}_{i:} \mathbf{a}^{(l-1)}+\mathbf{b}^{(l)}\right)}{\partial W_{i j}^{(l)}}} 
\\ {\vdots}
\\ {\frac{\partial\left(W^{(l)}_{m^{l}} \mathbf{a}^{(l-1)}+\mathbf{b}^{(l)}\right)}{\partial W_{i j}^{(l)}}}
\end{array}\right]
=\left[ \begin{array}{c}{0} \\ {\vdots} \\ {a_{j}^{(l-1)}} \\ {\vdots} \\ {0}\\
\end{array}\right] \triangleq \mathbb{I}_{i}\left(a_{j}^{(l-1)}\right)
$$
$W^{(l)}_{i:}$为矩阵$W^{(l)}$的第$i$行

2
$$
\frac{\partial \mathbf{z}^{(l)}}{\partial \mathbf{b}^{(l)}}=\mathbf{I}_{m^{(l)}}\in R^{m^{l}\times m^{l}}
$$
3
$$
由\mathbf{z}^{(l+1)}=W^{(l+1)} \mathbf{a}^{(l)}+\mathbf{b}^{(l+1)}可得\\
\frac{\partial \mathbf{z}^{(l+1)}}{\partial \mathbf{a}^{(l)}}=\left(W^{(l+1)}\right)^{\mathrm{T}}
$$

$$
\begin{aligned} \frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{z}^{(l)}} &=\frac{\partial f_{l}\left(\mathbf{z}^{(l)}\right)}{\partial \mathbf{z}^{(l)}} \\ &=\operatorname{diag}\left(f_{l}^{\prime}\left(\mathbf{z}^{(l)}\right)\right) \end{aligned}
$$

$$
\begin{aligned} 
\delta^{(l)}&=\frac{\partial \mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})}{\partial \mathbf{z}^{(l)}} \\
&=\frac{\partial \mathbf{a}^{(l)}}{\partial \mathbf{z}^{(l)}} \cdot \frac{\partial \mathbf{z}^{(l+1)}}{\partial \mathbf{a}^{(l)}} \cdot \frac{\partial \mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})}{\partial \mathbf{z}^{(l+1)}}
\\&=\operatorname{diag}\left(f_{l}^{\prime}\left(\mathbf{z}^{(l)}\right)\right) \cdot\left(W^{(l+1)}\right)^{T} \cdot \delta^{(l+1)}\\
矩阵维度的对应:&[m^l \times m^l][m^{l+1} \times m^l]^T [m^{l+1} \times 1]=[m^{l} \times 1]\\
&=f_{l}^{\prime}\left(\mathbf{z}^{(l)}\right) \odot\left(\left(W^{(l+1)}\right)^{\mathrm{T}} \delta^{(l+1)}\right)\\
矩阵维度的对应:&[m^l \times 1]\odot[m^{l+1} \times m^l]^T [m^{l+1} \times 1]=[m^{l} \times 1]\\
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial \mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})}{\partial W_{i j}^{(l)}}&=\mathbb{I}_{i}\left(a_{j}^{(l-1)}\right)^{\mathrm{T}} \delta^{(l)}=\delta_{i}^{(l)} a_{j}^{(l-1)}\\向量化后为:
\frac{\partial \mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})}{\partial W^{(l)}}&=\delta^{(l)}\left(\mathbf{a}^{(l-1)}\right)^{\mathrm{T}}\\
矩阵维度的对应:[m^{l} \times m^{l-1}]&=[m^{l} \times 1][m^{l-1} \times 1]^T\end{aligned}
$$

最后一层的误差不能按照公式计算
$$
\begin{array}{c}{\delta_{j}^{[L]}=\frac{\partial J}{\partial a_{j}^{L}} f_l^{\prime}\left(z_{j}^{[L]}\right)} \\ {\delta^{[L]}=\nabla_{a^L} J \odot f_l^{\prime}\left(z^{[L]}\right)}\end{array}
$$

$$
\frac{\partial \mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})}{\partial \mathbf{b}^{(l)}}=\delta^{(l)}
$$