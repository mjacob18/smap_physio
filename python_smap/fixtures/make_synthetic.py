import numpy as np, scipy.io as sio, sys
rng=np.random.default_rng(1); n=3000
x=np.zeros(n); y=np.zeros(n)
for i in range(2,n):
    x[i]=0.9*x[i-1]-0.2*x[i-2]+rng.normal()
    g=0.4*np.tanh(y[i-1])            # state-dependent gain
    y[i]=0.7*y[i-1]+g*x[i-3]+0.5*rng.normal()
X=np.column_stack([x,y,rng.normal(size=n).cumsum()*0.05+rng.normal(size=n)])
X[100:105,0]=np.nan
sio.savemat(sys.argv[1],dict(X=X))
