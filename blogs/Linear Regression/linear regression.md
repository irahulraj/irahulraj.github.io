```python
# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
# create dataset
np.random.seed(100)
dataset = np.random.randint(100, 300, size = [1000000,10])
dataset
```




    array([[108, 124, 167, ..., 238, 194, 280],
           [198, 153, 166, ..., 207, 160, 158],
           [244, 237, 193, ..., 232, 259, 229],
           ...,
           [216, 169, 201, ..., 154, 163, 255],
           [226, 256, 144, ..., 100, 131, 170],
           [212, 196, 224, ..., 284, 218, 165]])




```python
# add target variable
np.random.seed(101)
target = np.random.randint(50, 150, size = [1000000,1])
target
```




    array([[145],
           [ 61],
           [131],
           ...,
           [ 57],
           [146],
           [ 63]])




```python
# convert to dataframe
df = pd.DataFrame(dataset)
df['target'] = target
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>108</td>
      <td>124</td>
      <td>167</td>
      <td>203</td>
      <td>187</td>
      <td>179</td>
      <td>276</td>
      <td>238</td>
      <td>194</td>
      <td>280</td>
      <td>145</td>
    </tr>
    <tr>
      <th>1</th>
      <td>198</td>
      <td>153</td>
      <td>166</td>
      <td>114</td>
      <td>134</td>
      <td>124</td>
      <td>243</td>
      <td>207</td>
      <td>160</td>
      <td>158</td>
      <td>61</td>
    </tr>
    <tr>
      <th>2</th>
      <td>244</td>
      <td>237</td>
      <td>193</td>
      <td>186</td>
      <td>230</td>
      <td>255</td>
      <td>208</td>
      <td>232</td>
      <td>259</td>
      <td>229</td>
      <td>131</td>
    </tr>
    <tr>
      <th>3</th>
      <td>241</td>
      <td>200</td>
      <td>104</td>
      <td>191</td>
      <td>287</td>
      <td>167</td>
      <td>235</td>
      <td>149</td>
      <td>275</td>
      <td>293</td>
      <td>120</td>
    </tr>
    <tr>
      <th>4</th>
      <td>161</td>
      <td>114</td>
      <td>283</td>
      <td>299</td>
      <td>180</td>
      <td>102</td>
      <td>221</td>
      <td>205</td>
      <td>247</td>
      <td>163</td>
      <td>113</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

### Using Gradient Descent

We will implement this method using scikit-learn library.


```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

class lr_gd:
    def __init__(self, df, target):
        self.df = df
        self.target = target
    
    def run(self):
        X = self.df.drop(self.target, axis=1)
        y = self.df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state=42)
        
        # fit and train the linear regression
        self.reg = LinearRegression()
        self.reg.fit(X_train, y_train)
        
        return self.reg     
    
    def calculate_error(self):
        y_pred = self.reg.predict(X_test)
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
```


```python
%%time
reg = lr_gd(df,'target')
r = reg.run()
```

    Wall time: 766 ms
    


```python
r.coef_
```




    array([ 0.00150014,  0.00111651,  0.00020152,  0.00027139,  0.00017927,
            0.00096132, -0.00059907, -0.0012823 , -0.00065084, -0.00048191])




```python
r.intercept_
```




    99.27882470006116



### Using Slope formula


```python
class lr_slope:
    def __init__(self, df, target):
        self.df = df
        self.target = target
    
    def run(self):
        X = self.df.drop(self.target, axis=1)
        X['one'] = 1
        y = self.df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state=42)
        
        # convert them into numpy array to perform matrix operations
        np_X_train = X_train.to_numpy()
        np_y_train = y_train.to_numpy()
        
        # calculte the slope from formula
        weight = np.linalg.inv(np_X_train.T.dot(np_X_train)).dot(np_X_train.T).dot(np_y_train)
        
        return weight     

```


```python
%%time
lr = lr_slope(df,'target')
w = lr.run()
```

    Wall time: 790 ms
    


```python
w
```




    array([ 1.50013603e-03,  1.11651459e-03,  2.01517949e-04,  2.71392883e-04,
            1.79273644e-04,  9.61320105e-04, -5.99071710e-04, -1.28230042e-03,
           -6.50843172e-04, -4.81914596e-04,  9.92788247e+01])




```python
def get_coef_diff(weight, reg):
    coef1 = list(reg.coef_)
    coef1.append(reg.intercept_)
    
    weight1 = list(w)
    temp = []
    for i in range(len(coef1)):
        temp.append(coef1[i]-weight1[i])
    print('Difference of two wights are: \n')
    print(temp)
    
```


```python
get_coef_diff(w, r)
```

    Difference of two wights are: 
    
    [1.4200880055215137e-15, -4.098284211995207e-15, 3.474325861729799e-15, -3.037880277195759e-15, -2.5719714986244258e-15, 5.125023669338979e-16, -1.0366056971133908e-15, 7.435458498905589e-16, 7.238133703513228e-16, 1.1046393834368562e-15, 8.526512829121202e-14]
    

We can see how close they are. Almost no difference. so why do we not use analytical formula to calculate the slope. This invite another level of discussion. But in simple, it's the computational complexity which dominates. More discussion can be found here: - [StackOverflow](#https://stats.stackexchange.com/questions/278755/why-use-gradient-descent-for-linear-regression-when-a-closed-form-math-solution)




```python

```

## END
