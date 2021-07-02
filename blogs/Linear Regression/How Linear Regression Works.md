#            **How Linear Regression Works?**



### Overview

This is one of the simple algorithm. In this article, we will learn how actually linear regression works, how do they come up with coefficients. Those who don't know what is linear regression. Please go through [this one.](https://en.wikipedia.org/wiki/Linear_regression) 
In short, Linear regression is one of the linear modelling approach which map independent variable to dependent variable where dependent variable is continuous in nature.
Below figure is the illustration.

![Linear Regression Image](E:\Github\rahulrajaero.github.io\blogs\Linear Regression\lr1.JPG)

Linear Regression algorithm help us to find the equation of that trend-line (Red line). Let's equation be 
$$
y = mx + b
$$
We have to find the value of m and c. In data Science terminology, m and b are called the weight and bias respectively. Above is simple linear regression algorithm. If we have more than one independent variable (aka feature, column, attribute) the equation be like 


$$
y = m_1x_1 + m_2x_2 + ... + m_nx_n + b \\
Matrix,
Y = \beta*X, where \beta = (b, m_1, ..., m_n)
$$


Task is to find out the weights and bias matrix. 



There are two ways to find the weight: - 

1.  **Analytical Formula,** 


$$
   Y_{M} = 
   \begin{pmatrix}
   1 & x_1^1 & x_2^1 & \cdots & x_n^1 \\
   1 & x_1^2 & x_2^2 & \cdots & x_n^2 \\
   \vdots  & \vdots  & \ddots & \vdots  \\
   1 & x_1^m & x_2^m & \cdots & x_n^m
   \end{pmatrix}_{M,N+1}*
   \begin{pmatrix}
   \theta_0 \\ \theta_1 \\ \theta_2 \\ \cdots \\ \theta_n
   \end{pmatrix}_{N+1,1}
   \\~\\~\\
   Y = X\theta^T 
   \\~\\
   J(\theta) = 1/2 (X\theta^T - Y)^T(X\theta^T - Y)
$$



After minimizing this cost function, we arrive at following estimate
$$
\theta^T = (X^\intercal X)^{-1}X^\intercal Y
$$



2. **Using Gradient Descent Algorithms**

   There is a cost function which is minimized using gradient descent algorithms. 

   steps:

   * Define cost function:- 
     $$
     L = \sum (Y - Y^\hat~ )^2 \\
     Y^\hat~ = \beta^\hat~ * X + \epsilon
     $$

   

   * Update the weight matrix with the help of gradient descent
     $$
     \beta^\hat~ = \beta^\hat~ - \alpha * \nabla L
     $$
     ​		where $\alpha$ is called learning parameter, usually between 0.0 and 1.0.

     ​      and $\nabla L$ is the derivative of loss function w.r.t. to weight.



### Implementation in Python:

[Here you will find the python notebook.](ab.com)





For any comments you can reach out to me via LinkedIn.

Ref:-

* [StackExchange](#https://stats.stackexchange.com/questions/278755/why-use-gradient-descent-for-linear-regression-when-a-closed-form-math-solution)

  