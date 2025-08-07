import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils.utils import uncertainty

std_ord = 3

def plot_data(t_arr, data_train, data_test, train_idx, path, name='data'):
    fig, ax = plt.subplots(2,2, figsize=(2*5, 2*4))
    ax[0,0].plot(t_arr,data_test[:,0], '--')
    ax[0,0].plot(t_arr[train_idx],data_train[:,0], '^')
    ax[0,1].plot(t_arr,data_test[:,1], '--')
    ax[0,1].plot(t_arr[train_idx],data_train[:,1], '^')
    ax[1,1].plot(data_test[:,0],data_test[:,1], '--')
    ax[1,1].plot(data_train[:,0],data_train[:,1], '^')
    ax[0,0].set_xlabel('t')
    ax[0,1].set_xlabel('t')
    ax[0,0].set_ylabel('x')
    ax[0,1].set_ylabel('y')
    ax[1,1].set_xlabel('x')
    ax[1,1].set_ylabel('y')
    plt.savefig(path+'/plots/'+name+'.png')

def get_std_pred(ax, args, data, pred_rX, clr ='r', alpha=.15):

    n = pred_rX[0].shape[-1]
    # X_mean = jnp.mean(pred_rX[0], axis=0)
    # sX_std = jnp.sqrt(jnp.mean(pred_rX[1] + pred_rX[0]**2, axis=0) - X_mean**2)
    # sX_ale = jnp.sqrt(jnp.mean(pred_rX[1], axis=0))
    # sX_eps = jnp.std(pred_rX[0], axis=0)

    X_mean, sX_tot, sX_ale, sX_eps = uncertainty(pred_rX)

    for i in range(n):
        if i in args['train_var']:
            ax[i,2].plot(data['t_arr'][data['train_idx']],data['data_train'][:,i], '^m')
        for j in range(3):
            ax[i,j].plot(data['t_arr'],data['data_test'][:,i], '--m')
            ax[i,j].plot(data['t_arr'],X_mean[:,i], clr)
            # ax[i,j].set_xlabel('t')
            # ax[i,j].set_ylabel('X'+str(i))
            ax[i,j].set_ylim(jnp.min(X_mean[:,i]-std_ord*sX_tot[:,i]), jnp.max(X_mean[:,i]+std_ord*sX_tot[:,i]))
        ax[i,0].fill_between(data['t_arr'], (X_mean[:,i]-std_ord*sX_ale[:,i]), (X_mean[:,i]+std_ord*sX_ale[:,i]), color=clr, alpha=alpha)
        ax[i,1].fill_between(data['t_arr'], (X_mean[:,i]-std_ord*sX_eps[:,i]), (X_mean[:,i]+std_ord*sX_eps[:,i]), color=clr, alpha=alpha)
        ax[i,2].fill_between(data['t_arr'], (X_mean[:,i]-std_ord*sX_tot[:,i]), (X_mean[:,i]+std_ord*sX_tot[:,i]), color=clr, alpha=alpha)

    # ax[2,0].plot(data_test[:,0],data_test[:,1], '--m')
    # # ax[2,0].plot(data_train[:,0],data_train[:,1], '^m')
    # ax[2,1].plot(data_test[:,0],data_test[:,1], '--m')
    # ax[2,1].plot(data_train[:,0],data_train[:,1], '^m')
    # ax[2,0].plot(X_mean[:,0],X_mean[:,1], clr)
    # ax[0,0].set_xlabel('t')
    # ax[0,1].set_xlabel('t')
    # ax[0,2].set_xlabel('t')
    # ax[1,0].set_xlabel('t')
    # ax[1,1].set_xlabel('t')set_ylabel
    # ax[1,2].set_xlabel('t')
    # ax[0,0].set_ylabel('x')
    # ax[0,1].set_ylabel('x')
    # ax[0,2].set_ylabel('x')
    # ax[1,0].set_ylabel('y')
    # ax[1,1].set_ylabel('y')
    # ax[1,2].set_ylabel('y')
    # ax[2,0].set_xlabel('x')
    # ax[2,0].set_ylabel('y')
    # ax[2,1].set_xlabel('x')
    # ax[2,1].set_ylabel('y')
    return ax

def plot_ode_pred(args, data, pred_rX, name='pred_UT', clr ='r', alpha=.15):
    
    fontsize = 18

    n = pred_rX[0].shape[-1]
    plt.close('all')
    plt.rcParams['font.size'] = fontsize
    fig, ax = plt.subplots(n,3, figsize=(3*5, n*4), gridspec_kw = {'wspace':0, 'hspace':0})
    ax = get_std_pred(ax, args, data, pred_rX, clr, alpha)

    for i in [0,1]:
        ax[i,0].set_ylabel(r'$x_{}$'.format(i+1), size=20)
        for j in [1,2]:
            ax[i,j].tick_params(left = False)
            ax[i,j].axes.yaxis.set_ticklabels([])
    for j in [0,1,2]:
            ax[0,j].tick_params(bottom = False)
            ax[0,j].axes.xaxis.set_ticklabels([])
            ax[1,j].set_xlabel('t (sec)', size=20)

    ax[0,0].set_title('Aleatoric UQ', size=20)
    ax[0,1].set_title('Epistemic UQ', size=22)
    ax[0,2].set_title('Total UQ', size=22)
    # ax[0,0].text(-1, 0, r'$x_1$', horizontalalignment='right', verticalalignment='center', rotation='vertical', fontsize=fontsize*2)
    # ax[1,0].text(-1, 0, r'$x_2$', horizontalalignment='right', verticalalignment='center', rotation='vertical', fontsize=fontsize*2)
    fig.tight_layout()
    plt.savefig(args['path']+'/plots/'+name+'.png', dpi=300)


def plot_ode_val(args, data, pred_rX, HMC_rX, name='Pred_UTval', clr ='r', alpha=.15):
    
    fontsize = 18

    n = pred_rX[0].shape[-1]
    plt.close('all')
    plt.rcParams['font.size'] = fontsize
    fig, ax = plt.subplots(n,3, figsize=(3*5, n*4), gridspec_kw = {'wspace':0, 'hspace':0})

    ax = get_std_pred(ax, args, data, HMC_rX , clr='b', alpha=0.15)
    ax = get_std_pred(ax, args, data, pred_rX, clr='r', alpha=0.10)

    for i in [0,1]:
        ax[i,0].set_ylabel(r'$x_{}$'.format(i+1), size=20)
        for j in [1,2]:
            ax[i,j].tick_params(left = False)
            ax[i,j].axes.yaxis.set_ticklabels([])
    for j in [0,1,2]:
            ax[0,j].tick_params(bottom = False)
            ax[0,j].axes.xaxis.set_ticklabels([])
            ax[1,j].set_xlabel('t (sec)', size=20)

    ax[0,0].set_title('Aleatoric UQ', size=22)
    ax[0,1].set_title('Epistemic UQ', size=22)
    ax[0,2].set_title('Total UQ', size=22)
    fig.tight_layout()
    plt.savefig(args['path']+'/plots/'+name+'.png', dpi=300)


# def get_trajectory_pred(ax, pred_X, data_train, data_test, clr ='r'):

#     X_mean = jnp.mean(pred_X, axis=0)

#     ax[0].plot(data_test[:,0],data_test[:,1], '--m')
#     # ax[0].plot(data_train[:,0],data_train[:,1], '^m')
#     ax[1].plot(data_test[:,0],data_test[:,1], '--m')
#     ax[1].plot(data_train[:,0],data_train[:,1], '^m')
#     ax[0].plot(X_mean[:,0],X_mean[:,1], clr)
#     ax[0].set_xlabel('X0')
#     ax[0].set_ylabel('X1')
#     ax[1].set_xlabel('X0')
#     ax[1].set_ylabel('X1')
#     return ax

# def plot_2Dtrajectory(pred_X, data_train, data_test, path, name='trag_UT', clr ='r'):
    
#     n = pred_X.shape[-1]
#     plt.close('all')
#     fig, ax = plt.subplots(1,n, figsize=(n*5, 1*4))
#     ax = get_trajectory_pred(ax, pred_X, data_train, data_test, clr)
#     plt.savefig(path+'/plots/'+name+'.png')

def plot_ode_all(args, data, pred_X):
    plt.close('all')
    fig, ax = plt.subplots(2,2, figsize=(2*5, 2*4))
    ax[0,0].plot(data['t_arr'],data['data_test'][:,0], '--')
    ax[0,1].plot(data['t_arr'],data['data_test'][:,1], '--')
    ax[1,1].plot(data['data_test'][:,0],data['data_test'][:,1], '--')
    for nM in range(args['nModels']):
        ax[0,0].plot(data['t_arr'],pred_X[nM,:,0])
        ax[0,1].plot(data['t_arr'],pred_X[nM,:,1])
        ax[1,1].plot(pred_X[nM,:,0],pred_X[nM,:,1])
    ax[0,0].set_xlabel('t')
    ax[0,1].set_xlabel('t')
    ax[0,0].set_ylabel('x')
    ax[0,1].set_ylabel('y')
    ax[1,1].set_xlabel('x')
    ax[1,1].set_ylabel('y')
    plt.savefig(args['path']+'/plots/odeAll_lin.png')
