## float array

### Case

require

```
| x |    | x / z |
| y | -> | x / z |
| z |    |   1   |
```

best

```
X[:, 0] /= X[:, 2]
A[:, 1] /= A[:, 2]
A[:, 2] = 1
```

bad

```
X /= np.tile(A[:, 2], (3, 1)).T
```

### Case

require

```
X' = M * X
| x' |   | m0  m1  m2 | * | x |
| y' | = | m3  m4  m5 | * | y |
| z' |   | m6  m7  m8 | * | z |
```

best

```
X_ = X.dot(M.T)
```

### [Case](./source/test_230328_1.py)

```
(u, v) -> H * U^ -> s(u', v', 1) -> (u', v')
| s.u' |   | h0  h1  h2 | * | u |
| s.v' | = | h3  h4  h5 | * | v |
|   s  |   | h6  h7   1 | * | 1 |
```

### [Case](./source/test_230328_2.py)

```
u' = (u - cx) / fx
v' = (v - cy) / fy
```

### [Case](./source/test_230402_1.py)

```
create
| 0 0 .. 0 |
| 1 1 .. 1 |
| : :    : |
| n n .. n |
```