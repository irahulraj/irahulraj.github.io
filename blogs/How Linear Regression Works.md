#            **How Linear Regression Works?**



### Overview

This is one of the simple algorithm. In this article, we will learn how actually linear regression works, how do they come up with coefficients. Those who don't know what is linear regression. Please go through [this one.](https://en.wikipedia.org/wiki/Linear_regression) 
In short, Linear regression is one of the linear modelling approach which map independent variable to dependent variable where dependent variable is continuous in nature.
Below figure is the illustration.

![Image](C:\Users\RAHUL\Desktop\TheAnalyticsConsultant\Quick intro to plotly\lr1)



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


Task is to find out the weights and bias matrix. Under it through a simple dataset.

Here is the snapshot of dataset with shape = (28007, 62)

| target | Jan-19 | Apr-17 | Jul-19 | Nov-16 | May-16 | Dec-18 | Oct-19 | Feb-19 | Jun-17 | Jan-20 | Sep-18 | Nov-18 | Mar-18 | Oct-16 | Mar-16 | Jul-17 | Oct-18 | Aug-18 | Oct-20 | Jun-19 | Sep-19 | Dec-19 | Dec-17 | Jun-20 | Apr-18 | Jan-18 | Sep-20 | Nov-19 | Jan-16 | Apr-16 | Oct-17 | Aug-17 | Feb-17 | Jun-16 | Mar-19 | Nov-15 | Sep-16 | Mar-20 | Feb-16 | Apr-20 | Jul-20 | Nov-20 | Sep-17 | Jan-17 | Jul-16 | Dec-16 | Feb-20 | Aug-19 | Jul-18 | May-20 | May-17 | Dec-15 | Jun-18 | Aug-20 | Aug-16 | Apr-19 | Mar-17 | May-19 | Feb-18 | Nov-17 | May-18 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 385    | 0      | 0      | 440    | 0      | 0      | 345    | 515    | 155    | 0      | 210    | 135    | 55     | 0      | 0      | 0      | 0      | 85     | 95     | 280    | 185    | 660    | 505    | 0      | 390    | 3600   | 0      | 1320   | 292    | 0      | 0      | 0      | 0      | 0      | 0      | 55     | 0      | 0      | 260    | 0      | 610    | 660    | 770    | 0      | 0      | 0      | 0      | 120    | 483    | 65     | 230    | 0      | 0      | 350    | 1200   | 0      | 630    | 0      | 585    | 0      | 0      | 750    |
| 935    | 605    | 0      | 660    | 0      | 0      | 1100   | 825    | 550    | 0      | 639    | 0      | 440    | 0      | 0      | 0      | 0      | 0      | 385    | 655    | 880    | 440    | 605    | 0      | 495    | 2940   | 0      | 660    | 770    | 0      | 0      | 0      | 0      | 0      | 0      | 605    | 0      | 0      | 605    | 0      | 495    | 605    | 770    | 0      | 0      | 0      | 0      | 655    | 770    | 880    | 442    | 0      | 0      | 380    | 660    | 0      | 715    | 0      | 935    | 0      | 0      | 970    |
| 1200   | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 200    | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 1500   | 0      | 1350   | 250    | 0      | 0      | 0      | 0      | 0      | 2850   | 0      | 0      | 610    | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      |
| 530    | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 1300   | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 900    | 0      | 1100   | 1400   | 0      | 0      | 0      | 0      | 1420   | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 2200   | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 380    | 0      | 0      | 0      | 0      | 0      | 780    | 1180   | 600    |
| 330    | 0      | 0      | 80     | 0      | 0      | 80     | 80     | 80     | 0      | 0      | 0      | 160    | 330    | 0      | 0      | 0      | 40     | 660    | 0      | 0      | 40     | 0      | 280    | 0      | 200    | 200    | 0      | 520    | 0      | 0      | 910    | 0      | 0      | 0      | 40     | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 2640   | 0      | 0      | 0      | 0      | 40     | 80     | 0      | 0      | 0      | 289    | 0      | 0      | 0      | 0      | 0      | 180    | 480    | 370    |





There are two ways to find the weight: - 

1.  **Analytical Formula,** 
   $$
   \beta = (X^\intercal X)^{-1}X^\intercal Y
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
     ​		where $\alpha$ is called learning parameter, usually between -1 and 1.

     ​      and $\nabla L$ is the derivative of loss function w.r.t. to weight.



### Implementation in Python:



 













Ref:-

* [StackExchange](#https://stats.stackexchange.com/questions/278755/why-use-gradient-descent-for-linear-regression-when-a-closed-form-math-solution)
* 