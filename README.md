# ICE Estimation

```
pip install ice-estimation
```

Supported Filters: 

* Sliding Innovation Filter
* Kalman Filter

```
from ice_estimation import filters

filters.sif(x, z, u, P, A, B, C, Q, R,delta)
filters.kf_filter(x, z, u, P, A, B, C, Q, R)
```
