# Tiny

```py
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(12345)

x = np.arange(0, 100) + np.random.normal(0, 3, size=100)

y = np.concatenate((
    np.random.normal(1, 0.5, size=10),
    np.random.normal(1, 0.5, size=10),
    np.random.normal(3, 0.5, size=10),
    np.random.normal(5, 0.5, size=40),
    np.random.normal(3, 0.5, size=30)
))

# init weights randomly
init_ww = {
    'w_10': .05,
    'w_11': .03,
    'w_12': .01,
    'w_13': .02,
    'w_20': .10,
    'w_21': .30,
    'w_22': .20
}

# bias, always 1
b_00 = 1
b_10 = 1

# activation function
f = lambda x: 2.7 ** x

# deep learning network
def myDeep(ww, pt):
    x_11 = (b_00 * ww['w_10']) + (pt * ww['w_11'])
    x_12 = (b_00 * ww['w_12']) + (pt * ww['w_13'])
    a_11 = f(x_11)
    a_12 = f(x_12)
    return (b_10 * ww['w_20']) + (a_11 * ww['w_21']) + (a_12 * ww['w_22'])

fig, ax = plt.subplots()
plt.scatter(data[:, 0:1], data[:, 1:2]);
plt.scatter(data[:, 0:1], myDeep(init_ww, data[:, 0:1]));

init_ww['w_11'] =+ 0.02
plt.scatter(data[:, 0:1], myDeep(init_ww, data[:, 0:1]));
```




## References
- https://arxiv.org/pdf/1502.05767.pdf
- https://ipywidgets.readthedocs.io/en/stable
- https://matplotlib.org/stable/tutorials/index.html
