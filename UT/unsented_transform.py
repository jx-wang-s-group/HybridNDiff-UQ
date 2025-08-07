import jax
import numpy as np
import jax.numpy as jnp
from jax import random

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

kappa=0
alpha=1
beta=2

def get_unsented_transform_fu(vmap_fu, kappa=kappa, alpha=alpha, beta=beta, sigma_pts=False, cov_diag=True):

    vmap_tensor_prod = jax.vmap(lambda a: a.T @ a, in_axes=(0), out_axes=(0))

    @jax.jit
    def UT_fu(xmean, xcov, args):
        n_dim = len(xmean)
        
        if cov_diag:
            xcov_sq = jnp.linalg.cholesky(jnp.diag(xcov))
        else:
            xcov_sq = jnp.linalg.cholesky(xcov)
            
        lamb = alpha**2 *(n_dim+kappa) - n_dim

        x_sp0 = xmean
        x_sp1 = jnp.array([xmean]*n_dim) + jnp.sqrt(n_dim+lamb) *xcov_sq.T
        x_sp2 = jnp.array([xmean]*n_dim) - jnp.sqrt(n_dim+lamb) *xcov_sq.T
        x_sigma_pts = jnp.vstack([x_sp0,x_sp1,x_sp2])
        y_sigma_pts = vmap_fu(x_sigma_pts, args)

        W_m = jnp.array([lamb /(n_dim+lamb),                   *(0.5/(n_dim+lamb) for _ in range(2*n_dim))])
        W_c = jnp.array([lamb /(n_dim+lamb) + 1-alpha**2+beta, *(0.5/(n_dim+lamb) for _ in range(2*n_dim))])

        ymean = jnp.average(y_sigma_pts, weights = W_m, axis=0)

        yDiff      = y_sigma_pts - ymean
        yDiff_mat  = vmap_tensor_prod(jax.lax.expand_dims(yDiff, dimensions=[1]))
        ycov       = jnp.average(yDiff_mat,  weights = W_c, axis=0)
        if sigma_pts:
            return (ymean, ycov), (x_sigma_pts, y_sigma_pts)
        else:
            return ymean, ycov

    return UT_fu

# def get_2Dunsented_transform_fu1(vmap_fu, X_shape, kappa=kappa, alpha=alpha, beta=beta):

#     in_fu  = lambda x: (x.transpose(1,2,0)).reshape(-1,X_shape[0])
#     out_fu = lambda y: y.reshape(X_shape[1:])

#     covdiag = jax.vmap(lambda c: jnp.diag(c))
    
#     UT_fu = get_unsented_transform_fu(vmap_fu)
#     UT_vmap = jax.vmap(UT_fu, in_axes=(0,0,None), out_axes=([0,0]))
    
#     @jax.jit
#     def UT_fu(xmean, xcov, args):
#         xmean, xcov = in_fu(xmean), in_fu(xcov)
#         xcovm = covdiag(xcov)
#         ymean, ycov = UT_vmap(xmean, xcovm, args)
#         ymean, ycov = out_fu(ymean), out_fu(ycov)
#         return ymean, ycov

#     return UT_fu

def get_2Dunsented_transform_fu(vmap_fu, X_shape, ny=1, kappa=kappa, alpha=alpha, beta=beta, sigma_pts=False):

    vmap_tensor_prod = jax.vmap(lambda a: a.T @ a, in_axes=(0), out_axes=(0))

    in_fu  = lambda x: (x.transpose(1,2,0)).reshape(-1,X_shape[0])
    if ny == 1:
        out_fu = lambda y: y.reshape(X_shape[1:])
    else:
        out_fu = lambda y: (y.T).reshape(ny,*X_shape[1:])

    n_dim = X_shape[0]
    n_sgm = 2*n_dim+1
    lamb = alpha**2 *(n_dim+kappa) - n_dim

    def get_sigma_pts(xmean, xcov):
        xcov_sq = jnp.linalg.cholesky(xcov)

        x_sp0 = xmean
        x_sp1 = jnp.array([xmean]*n_dim) + jnp.sqrt(n_dim+lamb) *xcov_sq.T
        x_sp2 = jnp.array([xmean]*n_dim) - jnp.sqrt(n_dim+lamb) *xcov_sq.T
        x_sigma_pts = jnp.vstack([x_sp0,x_sp1,x_sp2])
        return x_sigma_pts

    def get_finaldist(y_sigma_pts):
        W_m = jnp.array([lamb /(n_dim+lamb),                   *(0.5/(n_dim+lamb) for _ in range(2*n_dim))])
        W_c = jnp.array([lamb /(n_dim+lamb) + 1-alpha**2+beta, *(0.5/(n_dim+lamb) for _ in range(2*n_dim))])

        ymean = jnp.average(y_sigma_pts, weights = W_m, axis=0)

        yDiff      = y_sigma_pts - ymean
        yDiff_mat  = vmap_tensor_prod(jax.lax.expand_dims(yDiff, dimensions=[1]))
        ycov       = jnp.average(yDiff_mat,  weights = W_c, axis=0)
        return ymean, ycov

    @jax.jit
    def UT_fu(xmean, xcov, args):
        xmean, xcov = in_fu(xmean), in_fu(xcov)
        xcov = jax.vmap(lambda c: jnp.diag(c))(xcov)
        x_sigma_pts = jax.vmap(get_sigma_pts, in_axes=(0,0))(xmean, xcov)
        x_sigma_pts = (x_sigma_pts.transpose(1,2,0)).reshape(n_sgm, *X_shape)

        y_sigma_pts = vmap_fu(x_sigma_pts, args)
        if ny != 1: 
            y_sigma_pts = (y_sigma_pts.transpose(2,3,0,1)).reshape(-1, n_sgm, ny)
        else:
            y_sigma_pts = (y_sigma_pts.transpose(1,2,0)).reshape(-1, n_sgm)

        ymean, ycov = jax.vmap(get_finaldist, out_axes=(0, 0))(y_sigma_pts)
        if ny != 1: ycov = jax.vmap(lambda c: jnp.diag(c))(ycov)
        ymean, ycov = out_fu(ymean), out_fu(ycov)

        if sigma_pts:
            return (ymean, ycov), (x_sigma_pts, y_sigma_pts)
        else:
            return ymean, ycov

    return UT_fu
##################################################################################################
if __name__ == "__main__":

    def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.

        Parameters
        ----------
        x, y : array-like, shape (n, )
            Input data.

        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`

        Returns
        -------
        matplotlib.patches.Ellipse
        """
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensional dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                        facecolor=facecolor, **kwargs)

        # Calculating the standard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the standard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    def fu(x, args):
        t = args
        return jnp.array((x[0]+x[1], 0.5*x[0]**2+x[1]**2))
        # return jnp.array((2*x[0]-0.5*x[1],1*x[0]-2.5*x[1]))

    vmap_fu = jax.vmap(fu, in_axes=(0,None), out_axes=(0))

    vmap_tensor_prod = jax.vmap(lambda a: a.T @ a, in_axes=(0), out_axes=(0))

    key = random.PRNGKey(0)

    mean = jnp.array([0,1])
    cov  = jnp.array([[32,15],[15,40]])

    key, subkey = random.split(key)
    x = random.multivariate_normal(subkey, mean, cov, shape=[1000])

    y = vmap_fu(x, 0)

    xmean,  xcov  = jnp.mean(x,0), jnp.cov(x.T)
    ymeant, ycovt = jnp.mean(y,0), jnp.cov(y.T)

    UT_fu = get_unsented_transform_fu(vmap_fu, sigma_pts=True, cov_diag=False)
    (ymean, ycov), (x_sigma_pts, y_sigma_pts) = UT_fu(xmean, xcov, 0)
       
    print('True:', ymeant, ycovt)
    print('Pred:', ymean, ycov)

    plt.close('all')
    fig,ax = plt.subplots(2, 1, figsize=(1*5, 2*4))
    ax[0].scatter(x[:,0], x[:,1])
    ax[0].scatter(x_sigma_pts[:,0], x_sigma_pts[:,1], c='r', marker="o")
    ax[0].plot(xmean[0], xmean[1], '^r')
    confidence_ellipse(x[:,0], x[:,1], ax[0], n_std=2.0,edgecolor='red')
    ax[1].scatter(y[:,0], y[:,1])
    ax[1].scatter(y_sigma_pts[:,0], y_sigma_pts[:,1], c='r', marker="o")
    ax[1].plot(ymeant[0], ymeant[1], '^r')
    ax[1].plot(ymean[0], ymean[1], 'ok', mfc='none', linewidth=3, markersize=12)
    confidence_ellipse(y[:,0], y[:,1], ax[1], n_std=2.0,edgecolor='red')

    key, subkey = random.split(key)
    y = random.multivariate_normal(subkey, ymean, ycov, shape=[1000])
    confidence_ellipse(y[:,0], y[:,1], ax[1], n_std=2.0,edgecolor='black')

    plt.savefig('./UT/UT22_dist.png')


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

    UT_fu = get_2Dunsented_transform_fu(vmap_fu, (2,10,10), 2, sigma_pts=True)
    (ymean, ycov), (x_sigma_pts, y_sigma_pts) = UT_fu(xmean, xcov, 0)
       
    plt.close('all')
    fig,ax = plt.subplots(2, 1, figsize=(1*5, 2*4))
    ax[0].scatter(x[:,0], x[:,1])
    ax[0].scatter(x_sigma_pts[0,:,0], x_sigma_pts[0,:,1], c='r', marker="o")
    ax[0].plot(xmean[0,0,0], xmean[1,0,0], '^r')
    confidence_ellipse(x[:,0], x[:,1], ax[0], n_std=2.0,edgecolor='red')
    ax[1].scatter(y[:,0], y[:,1])
    ax[1].scatter(y_sigma_pts[0,:,0], y_sigma_pts[0,:,1], c='r', marker="o")
    ax[1].plot(ymeant[0], ymeant[1], '^r')
    ax[1].plot(ymean[0,0,0], ymean[1,0,0], 'ok', mfc='none', linewidth=3, markersize=12)
    confidence_ellipse(y[:,0], y[:,1], ax[1], n_std=2.0,edgecolor='red')

    key, subkey = random.split(key)
    y = random.multivariate_normal(subkey, ymean[:,0,0], jnp.diag(ycov[:,0,0]), shape=[1000])
    confidence_ellipse(y[:,0], y[:,1], ax[1], n_std=2.0,edgecolor='black')

    plt.savefig('./UT/UT22_dist2.png')
