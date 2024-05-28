## Data_Frames(a)




```python
import numpy as np
```


```python
import pandas as pd
```


```python
from numpy.random import randn
```


```python
np.random.seed(101)
```


```python
df = pd.DataFrame(randn(5,6),['A','B','C','D','E'], ['W','X','Y','Z','A','B'])
```


```python
df
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
    </tr>
    <tr>
      <th>B</th>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>C</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
    </tr>
    <tr>
      <th>D</th>
      <td>2.605967</td>
      <td>0.683509</td>
      <td>0.302665</td>
      <td>1.693723</td>
      <td>-1.706086</td>
      <td>-1.159119</td>
    </tr>
    <tr>
      <th>E</th>
      <td>-0.134841</td>
      <td>0.390528</td>
      <td>0.166905</td>
      <td>0.184502</td>
      <td>0.807706</td>
      <td>0.072960</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Z']
```




    A    0.503826
    B    0.740122
    C    0.955057
    D    1.693723
    E    0.184502
    Name: Z, dtype: float64




```python
type(df['Y'])
```




    pandas.core.series.Series




```python
df.Y
```




    A    0.907969
    B   -2.018168
    C   -0.933237
    D    0.302665
    E    0.166905
    Name: Y, dtype: float64




```python
df['new'] = df['W'] + df['Y']
```


```python
df
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
      <th>A</th>
      <th>B</th>
      <th>new</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
      <td>3.614819</td>
    </tr>
    <tr>
      <th>B</th>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
      <td>-2.866245</td>
    </tr>
    <tr>
      <th>C</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
      <td>-0.744542</td>
    </tr>
    <tr>
      <th>D</th>
      <td>2.605967</td>
      <td>0.683509</td>
      <td>0.302665</td>
      <td>1.693723</td>
      <td>-1.706086</td>
      <td>-1.159119</td>
      <td>2.908633</td>
    </tr>
    <tr>
      <th>E</th>
      <td>-0.134841</td>
      <td>0.390528</td>
      <td>0.166905</td>
      <td>0.184502</td>
      <td>0.807706</td>
      <td>0.072960</td>
      <td>0.032064</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop('new', axis = 1, inplace=True)
```


```python
df
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
      <th>W</th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.706850</td>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
    </tr>
    <tr>
      <th>B</th>
      <td>-0.848077</td>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>C</th>
      <td>0.188695</td>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
    </tr>
    <tr>
      <th>D</th>
      <td>2.605967</td>
      <td>0.683509</td>
      <td>0.302665</td>
      <td>1.693723</td>
      <td>-1.706086</td>
      <td>-1.159119</td>
    </tr>
    <tr>
      <th>E</th>
      <td>-0.134841</td>
      <td>0.390528</td>
      <td>0.166905</td>
      <td>0.184502</td>
      <td>0.807706</td>
      <td>0.072960</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop('W', axis = 1, inplace=True)
```


```python
df
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
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.683509</td>
      <td>0.302665</td>
      <td>1.693723</td>
      <td>-1.706086</td>
      <td>-1.159119</td>
    </tr>
    <tr>
      <th>E</th>
      <td>0.390528</td>
      <td>0.166905</td>
      <td>0.184502</td>
      <td>0.807706</td>
      <td>0.072960</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop('E', axis = 0, inplace=True)
```


```python
df
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
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.683509</td>
      <td>0.302665</td>
      <td>1.693723</td>
      <td>-1.706086</td>
      <td>-1.159119</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop('D')
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
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (4, 5)




```python
df
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
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.683509</td>
      <td>0.302665</td>
      <td>1.693723</td>
      <td>-1.706086</td>
      <td>-1.159119</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['Y','X']]
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
      <th>Y</th>
      <th>X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0.907969</td>
      <td>0.628133</td>
    </tr>
    <tr>
      <th>B</th>
      <td>-2.018168</td>
      <td>0.605965</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-0.933237</td>
      <td>-0.758872</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.302665</td>
      <td>0.683509</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['A']
```




    X    0.628133
    Y    0.907969
    Z    0.503826
    A    0.651118
    B   -0.319318
    Name: A, dtype: float64




```python
df.iloc[2]
```




    X   -0.758872
    Y   -0.933237
    Z    0.955057
    A    0.190794
    B    1.978757
    Name: C, dtype: float64




```python
df.loc[['A','B']]
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
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0.628133</td>
      <td>0.907969</td>
      <td>0.503826</td>
      <td>0.651118</td>
      <td>-0.319318</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.605965</td>
      <td>-2.018168</td>
      <td>0.740122</td>
      <td>0.528813</td>
      <td>-0.589001</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[2,4]
```




    1.9787573241128278




```python
df.iloc[2:4]
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
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C</th>
      <td>-0.758872</td>
      <td>-0.933237</td>
      <td>0.955057</td>
      <td>0.190794</td>
      <td>1.978757</td>
    </tr>
    <tr>
      <th>D</th>
      <td>0.683509</td>
      <td>0.302665</td>
      <td>1.693723</td>
      <td>-1.706086</td>
      <td>-1.159119</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[['A','B'],['Z','Y']]
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
      <th>Z</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0.503826</td>
      <td>0.907969</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.740122</td>
      <td>-2.018168</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
