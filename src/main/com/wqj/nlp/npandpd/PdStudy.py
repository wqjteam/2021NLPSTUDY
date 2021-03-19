import pandas as pd
import numpy as np
from pandas import Series, DataFrame

obj = Series([1, 2, 3, 4, 5])
print(obj)
print(obj.values)
print(obj.index)

data = {'a': 1000, 'b': 20000, 'c': 30000}
obj2 = Series(data)
print(obj2)

data2 = {'a': None, 'b': 20000, 'c': 30000}
obj3 = Series(data2)
print(pd.isnull(obj3))

dates = pd.date_range('20190301', periods=6)
print(dates)
df = pd.DataFrame(np.random.rand(6, 4), index=dates, columns=list('ABCD'))
print(df)
print(df['20190301':'20190303'])
print(df.loc['20190301':'20190302',['A','B']])
print(df.at[dates[0],'A'])
