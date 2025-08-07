import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils.utils import uncertainty

cmap = 'RdGy'
# cmap = 'Reds'

def plot_pde_wtime(Grid, t_arr, Data, nsample_times, path):
    
    sample_times = [int(el) for el in jnp.linspace(0, len(t_arr)-1, nsample_times)]

    Udmin, Udmax = jnp.min(Data[:,0]), jnp.max(Data[:,0])
    Vdmin, Vdmax = jnp.min(Data[:,1]), jnp.max(Data[:,1])
    
    plt.close('all')
    fig, ax = plt.subplots(len(sample_times),4, figsize=(4*5, len(sample_times)*4))
    for i,t in enumerate(sample_times):
        U = Data[t,0]
        V = Data[t,1]
        pcmu = ax[i,0].contourf(Grid['grid_x'], Grid['grid_y'], U, levels = jnp.linspace(jnp.min(U), jnp.max(U),20), cmap=cmap)
        pcmv = ax[i,1].contourf(Grid['grid_x'], Grid['grid_y'], V, levels = jnp.linspace(jnp.min(V), jnp.max(V),20), cmap=cmap)
        fig.colorbar(pcmu , ax=ax[i,0], location='left')
        fig.colorbar(pcmv , ax=ax[i,1], location='right')

        pcmu = ax[i,2].contourf(Grid['grid_x'], Grid['grid_y'], U, levels = jnp.linspace(Udmin, Udmax,20), cmap=cmap)
        pcmv = ax[i,3].contourf(Grid['grid_x'], Grid['grid_y'], V, levels = jnp.linspace(Vdmin, Vdmax,20), cmap=cmap)
        fig.colorbar(pcmu , ax=ax[i,2], location='left')
        fig.colorbar(pcmv , ax=ax[i,3], location='right')
    plt.savefig(path+'/plots/data_rd')


def plot_pde_2Dpred(args, data, pred_rX, nsample_times, name=''):
    
    sample_times = [int(el) for el in jnp.linspace(0, len(data['t_arr']), nsample_times)]

    # X_mean = jnp.mean(pred_rX[0], axis=0)
    # sX_std = jnp.sqrt(jnp.mean(pred_rX[1] + pred_rX[0]**2, axis=0) - X_mean**2)
    # sX_ale = jnp.sqrt(jnp.mean(pred_rX[1], axis=0))
    # sX_eps = jnp.std(pred_rX[0], axis=0)

    X_mean, sX_tot, sX_ale, sX_eps = uncertainty(pred_rX)

    Udmin, Udmax = jnp.min(data['data_test'][:,0]), jnp.max(data['data_test'][:,0])
    Upmin, Upmax = jnp.min(X_mean[:,0]), jnp.max(X_mean[:,0])
    sUpmin, sUpmax = 0., jnp.max(sX_tot[:,0])
    Up, Vp = X_mean[:,0], X_mean[:,1]
    Ud, Vd = data['data_test'][:,0], data['data_test'][:,1]
    sU, sV = sX_tot[:,0], sX_tot[:,1]
    saU, saV = sX_ale[:,0], sX_ale[:,1]
    seU, seV = sX_eps[:,0], sX_eps[:,1]
    dftest  = jnp.abs(Up-Ud)
    dftrain = jnp.abs(Up[:len(data['data_train'])]-data['data_train'][:,0])
    dfmax   = 0.6 #max(jnp.max(dftest), jnp.max(dftrain))

    Grid = data['Grid']

    plt.close('all')
    fig, ax = plt.subplots(len(sample_times),7, figsize=(7*5, len(sample_times)*4))
    for i,t in enumerate(sample_times):
        pcm0 = ax[i,0].contourf(Grid['grid_x'], Grid['grid_y'], Ud[t],  levels = jnp.linspace(Udmin, Udmax,20), cmap=cmap)
        pcm1 = ax[i,1].contourf(Grid['grid_x'], Grid['grid_y'], Up[t],  levels = jnp.linspace(Udmin, Udmax,20), cmap=cmap, extend='min')
        pcm2 = ax[i,2].contourf(Grid['grid_x'], Grid['grid_y'], dftest[t],  levels = jnp.linspace(0, dfmax,20), extend='max')
        if t <= len(data['data_train']):
            pcm2 = ax[i,3].contourf(Grid['grid_x'], Grid['grid_y'], dftrain[t],  levels = jnp.linspace(0, dfmax,20), extend='max')

        pcm5 = ax[i,4].contourf(Grid['grid_x'], Grid['grid_y'], sU[t],  levels = jnp.linspace(sUpmin, sUpmax,20))
        pcm3 = ax[i,5].contourf(Grid['grid_x'], Grid['grid_y'], saU[t], levels = jnp.linspace(sUpmin, sUpmax,20))
        pcm4 = ax[i,6].contourf(Grid['grid_x'], Grid['grid_y'], seU[t], levels = jnp.linspace(sUpmin, sUpmax,20))
    fig.colorbar(pcm1 , ax=ax[i,0:2], location='bottom')
    # fig.colorbar(pcm0 , ax=ax[i,1], location='bottom')
    fig.colorbar(pcm2 , ax=ax[i,2:4], location='bottom')
    fig.colorbar(pcm3 , ax=ax[i,4:7], location='bottom')
    # fig.tight_layout()
    plt.savefig(args['path']+'/plots/pred_2Drd_'+name)


def plot_pde_1Dpred(args, data, pred_rX, name='', clr ='r', alpha=.15, std_ord=3):
    
    idx_list = [[5,5], [10,10], [15,15]]

    # X_mean = jnp.mean(pred_rX[0], axis=0)
    # sX_std = jnp.sqrt(jnp.mean(pred_rX[1] + pred_rX[0]**2, axis=0) - X_mean**2)
    # sX_ale = jnp.sqrt(jnp.mean(pred_rX[1], axis=0))
    # sX_eps = jnp.std(pred_rX[0], axis=0)

    X_mean, sX_tot, sX_ale, sX_eps = uncertainty(pred_rX)

    plt.close('all')
    nv=2
    fig, ax = plt.subplots(nv*len(idx_list),3, figsize=(3*5, nv*len(idx_list)*4))
    
    for n, idx in enumerate(idx_list):
        for i in args['train_var']:
            ax[2*n+i,2].plot(data['t_arr'][data['train_idx']],data['data_train'][:,i,*idx], '^m')
        for i in range(nv):
            for j in range(3):
                ax[2*n+i,j].plot(data['t_arr'],data['data_test'][:,i,*idx], '--m')
                ax[2*n+i,j].plot(data['t_arr'],X_mean[:,i,*idx], clr)
                ax[2*n+i,j].set_xlabel('t')
                ax[2*n+i,j].set_ylabel('X'+str(i))
            ax[2*n+i,0].fill_between(data['t_arr'], (X_mean[:,i,*idx]-std_ord*sX_ale[:,i,*idx]), (X_mean[:,i,*idx]+std_ord*sX_ale[:,i,*idx]), color=clr, alpha=alpha)
            ax[2*n+i,1].fill_between(data['t_arr'], (X_mean[:,i,*idx]-std_ord*sX_eps[:,i,*idx]), (X_mean[:,i,*idx]+std_ord*sX_eps[:,i,*idx]), color=clr, alpha=alpha)
            ax[2*n+i,2].fill_between(data['t_arr'], (X_mean[:,i,*idx]-std_ord*sX_tot[:,i,*idx]), (X_mean[:,i,*idx]+std_ord*sX_tot[:,i,*idx]), color=clr, alpha=alpha)

    plt.savefig(args['path']+'/plots/pred_1Drd_'+name)