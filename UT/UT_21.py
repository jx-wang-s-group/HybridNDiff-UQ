import sys
 
# setting path
sys.path.append('../UnscentedTransform_UQpropagation')

import time
import jax
import jax.numpy as jnp
from jax import random
from UT.unsented_transform import get_unsented_transform_fu, get_2Dunsented_transform_fu

import matplotlib.pyplot as plt


##################################################################################################
if __name__ == "__main__":

    starttime = time.time()

    def fu(x, args):
        t = args
        # return jnp.array((0.5*x[0]**2+x[1]**2))
        return jnp.array((x[0]/x[1]))

    vmap_fu = jax.vmap(fu, in_axes=(0,None), out_axes=(0))

    key = random.PRNGKey(0)

    mean = jnp.array([100,250])
    cov  = jnp.array([[32,0],[0,40]])

    key, subkey = random.split(key)
    x = random.multivariate_normal(subkey, mean, cov, shape=[1000])

    y = vmap_fu(x,0)

    xmean, xcov = jnp.mean(x,0), jnp.cov(x.T)
    ymeant, ycovt = jnp.mean(y), jnp.var(y)
    
    UT_fu = get_unsented_transform_fu(vmap_fu, sigma_pts=True, cov_diag=False)
    (ymean, ycov), (x_sigma_pts, y_sigma_pts) = UT_fu(xmean, xcov, 0)
        
    print('True:', ymeant, ycovt)
    print('Pred:', ymean, ycov)

    plt.close('all')
    fig,ax = plt.subplots(2, 1, figsize=(1*5, 2*4))
    ax[0].scatter(x[:,0], x[:,1])
    ax[0].scatter(x_sigma_pts[:,0], x_sigma_pts[:,1], c='r', marker="o")
    ax[0].plot(xmean[0], xmean[1], '^r')
    
    ax[1].scatter(y, jnp.ones_like(y))
    ax[1].scatter(y_sigma_pts, jnp.ones_like(y_sigma_pts), c='r', marker="o")
    ax[1].plot(ymeant, 1, '^r')
    ax[1].plot(ymean, 1, 'ok', mfc='none', linewidth=3, markersize=12)
    plt.savefig('./UT/UT21_dist.png')

    endtime = time.time()
    print("run time = ",time.strftime("%H:%M:%S", time.gmtime(endtime - starttime)))

    # 2D --------------------

    mean = jnp.array([1,1])
    cov  = jnp.array([[1,0],[0,1]])

    key, subkey = random.split(key)
    x = random.multivariate_normal(subkey, mean, cov, shape=[1000])

    y = vmap_fu(x, 0)

    xmean,  xcov  = jnp.mean(x,0), jnp.cov(x.T)
    ymeant, ycovt = jnp.mean(y,0), jnp.cov(y.T)

    xmean = jnp.ones((2,10,10))
    xcov  = jnp.ones((2,10,10))

    UT_fu = get_2Dunsented_transform_fu(vmap_fu, (2,10,10), sigma_pts=True)
    (ymean, ycov), (x_sigma_pts, y_sigma_pts) = UT_fu(xmean, xcov, 0)
       
    plt.close('all')
    fig,ax = plt.subplots(2, 1, figsize=(1*5, 2*4))
    ax[0].scatter(x[:,0], x[:,1])
    ax[0].scatter(x_sigma_pts[0,:,0], x_sigma_pts[0,:,1], c='r', marker="o")
    ax[0].plot(xmean[0,0,0], xmean[1,0,0], '^r')
    
    ax[1].scatter(y, jnp.ones_like(y))
    ax[1].scatter(y_sigma_pts[0], jnp.ones_like(y_sigma_pts[0]), c='r', marker="o")
    ax[1].plot(ymeant, 1, '^r')
    ax[1].plot(ymean[0,0], 1, 'ok', mfc='none', linewidth=3, markersize=12)
    
    plt.savefig('./UT/UT21_dist2.png')
