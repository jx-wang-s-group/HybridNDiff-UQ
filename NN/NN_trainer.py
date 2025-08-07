import jax
import jax.random as random
import jax.numpy as jnp
import jax.scipy.stats as stats
import optax
import haiku as hk
from jax.tree_util import Partial

from NN.NN_utils import MLP
from NN.NN_trainer_utils import *
from NN.SWAG_utils import get_update_SWAG, get_SWAG_sampler, get_SWAG_pred
from utils.utils import PyTree
from utils.plots_ode import plot_ode_pred, plot_ode_all
from utils.plots_pde import plot_pde_1Dpred, plot_pde_2Dpred

import matplotlib.pyplot as plt


def trainSWAG(key, nll_fu, solver_vmap, opt_fu, 
              params,mask_para,opt_state,params_init,
              cdata,args):

    print('-------------- Training --------------')
    SWAG_cov_size = 5
    model_select  = 0.8
    # nepoch_DE     = int(0.5*nepochs)
    SWAG_depoch   = 10
    nSWAG_iter    = 0

    # key          = random.split(key)
    params_ts    = PyTree.set_val(params, val=0)
    params_init_ts    = PyTree.set_val(params_init, val=0)
    opt_state_ts = PyTree.set_val(opt_state, val=0)
    update       = get_update_fu(opt_fu, nll_fu, mask_para, args['debug'])
    update_vmap  = jax.vmap(update, in_axes=(params_ts,opt_state_ts,params_init_ts,None), out_axes=(params_ts,opt_state_ts,0))
    update_SWAG_vmap  = jax.vmap(get_update_SWAG(args['debug']), in_axes=(None,params_ts, 0, 0), out_axes=(0,0,0))
    SWAG_samples_vmap = jax.vmap(get_SWAG_sampler(num_samples=5, debug=args['debug']), in_axes=(0, 0, 0, [0,]*SWAG_cov_size), out_axes=(0,0))
    
    TrainLossStore, PhyDEStore, PhySWAGStore = [], [], []


    # ------------------------------------- SWAG -----------------------------------------------------------------------------
    params_flat, pytree_fu = jax.flatten_util.ravel_pytree(PyTree.extract(params, 0))
    params_SWA, params_Sq = jnp.array([params_flat]*args['nModels']), jnp.array([params_flat**2]*args['nModels'])
    D_hat_list = [jnp.array([params_flat-params_flat]*args['nModels'])]*SWAG_cov_size
    SWAG_pred  = get_SWAG_pred(SWAG_samples_vmap, pytree_fu, solver_vmap, args['debug'])

    if args['epochstart'] > 0:
        state = PyTree.load(args['path']+'/checkpoints', name='state'+str(args['epochstart']))
        params, params_SWA, params_Sq, D_hat_list, nSWAG_iter = state['params'], state['params_SWA'], state['params_Sq'], state['D_hat_list'], state['nSWAG_iter']

    key = [key]
    print('---------- gathering SWAG parameters ----------')
    for epoch in range(args['epochstart']+1, args['nepochs']+1):
        epoch_ratio = 1#min(1., epoch/(0.8*args['nepochs']))
        # update
        params, opt_state, loss = update_vmap(params,opt_state, params_init, epoch_ratio)

        # loss
        sort_idx = loss.argsort()[:int(model_select*args['nModels'])]
        loss_m = jnp.mean(loss[sort_idx])
        if epoch % args['print_depoch'] == 0:
            print(epoch, loss_m, jnp.mean(jnp.abs(params['cov']), axis=0))
        TrainLossStore.append([epoch,loss])

        # # store Phy para
        # if 'Phy' in mask_para.keys():
        #     PhyDEStore.append([epoch, params['Phy']['value']['Coeff'][0][0], params['Phy']['value']['Coeff'][1][0]])

        # ----- Update SWAG parameters ---------------------
        # if (epoch >= 0.3*args['nepochs']) and (epoch % SWAG_depoch == 0):
        if epoch % SWAG_depoch == 0:
            # params_SWA, params_Sq, params_dev = update_SWAG((epoch-nepoch_DE)//SWAG_depoch, PyTree.extract(params, 0), params_SWA[0], params_Sq[0])
            nSWAG_iter += 1
            params_SWA, params_Sq, params_dev = update_SWAG_vmap(nSWAG_iter, params, params_SWA, params_Sq)

            # store Phy para
            if 'Phy' in mask_para.keys():
                p_SWA = jax.vmap(pytree_fu)(params_SWA)
                p_Sq = jax.vmap(pytree_fu)(params_Sq)
                PhyDEStore.append([epoch,  params['Phy']['value']['Coeff'][0][0], params['Phy']['value']['Coeff'][1][0]])
                PhySWAGStore.append([epoch, [p_SWA['Phy']['value']['Coeff'][0][0],   p_Sq['Phy']['value']['Coeff'][0][0]],
                                            [p_SWA['Phy']['value']['Coeff'][1][0],   p_Sq['Phy']['value']['Coeff'][1][0]]])

            # if len(D_hat_list) == SWAG_cov_size:
            del D_hat_list[0]
            D_hat_list.append(params_dev)

        if epoch % args['save_depoch'] == 0:
            subkey = random.split(key[0], num=args['nModels'])
            key, pred_X, pred_cX, params_samples = SWAG_pred(subkey, params_SWA, params_Sq, D_hat_list, sort_idx)
            PostPross_Data = {  'TrainLossStore':PyTree.combine(TrainLossStore),
                                'pred_rX': (pred_X, pred_cX),
                                'sort_idx':sort_idx,
                                'cdata': cdata}
            if 'Phy' in mask_para.keys():  
                PostPross_Data['PhyDEStore']   = PyTree.combine(PhyDEStore)
                PostPross_Data['PhySWAGStore'] = PyTree.combine(PhySWAGStore)
            state = {'params':params,
                     'params_SWA':params_SWA,
                     'params_Sq':params_Sq,
                     'D_hat_list':D_hat_list,
                     'nSWAG_iter':nSWAG_iter}

            PyTree.save(PostPross_Data, args['path']+'/checkpoints', name='PostPross_Data'+str(epoch))
            PyTree.save(state, args['path']+'/checkpoints', name='state'+str(epoch))
            del pred_X, pred_cX, params_samples, PostPross_Data, state


        # ----- plot ---------------------
        if epoch % args['plot_depoch'] == 0:

        # # ----- plot SWAG-mean ---------------------
        #     params_samples  = jax.vmap(pytree_fu)(params_SWA)
        #     (pred_X, pred_cX), (dXdt, dcXdt) = solver_vmap(params_samples)
        #     pred_X, pred_cX, dXdt, dcXdt = pred_X[sort_idx], pred_cX[sort_idx], dXdt[sort_idx], dcXdt[sort_idx]
        #     if 'PDE' in args['path']:
        #         plot_pde_2Dpred(args, data, (pred_X, pred_cX), 5, name='SWAG')
        #         plot_pde_1Dpred(args, data, (pred_X, pred_cX), name='SWAG')
        #     else:
        #         plot_ode_pred(args, data, pred_X, pred_cX, name='SWAG')

        # ----- plot SWAG ---------------------
            subkey = random.split(key[0], num=args['nModels'])
            key, pred_X, pred_cX, params_samples = SWAG_pred(subkey, params_SWA, params_Sq, D_hat_list, sort_idx)
            if 'PDE' in args['path']:
                plot_pde_2Dpred(args, cdata, (pred_X, pred_cX), 5, name='SWAG_train')
                plot_pde_1Dpred(args, cdata, (pred_X, pred_cX), name='SWAG_train')
            else:
                plot_ode_pred(args, cdata, (pred_X, pred_cX), name='SWAG_train')
                plot_ode_all(args, cdata, pred_X)
            del pred_X, pred_cX, params_samples

    subkey = random.split(key[0], num=args['nModels'])
    key, pred_X, pred_cX, params_samples = SWAG_pred(subkey, params_SWA, params_Sq, D_hat_list, sort_idx)        
    PostPross_Data = {'TrainLossStore':PyTree.combine(TrainLossStore),
                      'pred_rX': (pred_X, pred_cX),
                      'sort_idx':sort_idx,
                      'cdata': cdata}
    if 'Phy' in mask_para.keys():  
        PostPross_Data['PhyDEStore']   = PyTree.combine(PhyDEStore)
        PostPross_Data['PhySWAGStore'] = PyTree.combine(PhySWAGStore)

    state = {'params':params,
            'params_SWA':params_SWA,
            'params_Sq':params_Sq,
            'D_hat_list':D_hat_list,
            'nSWAG_iter':nSWAG_iter}
    
    PyTree.save(PostPross_Data, args['path']+'/checkpoints', name='PostPross_Data'+str(epoch))
    PyTree.save(state, args['path']+'/checkpoints', name='state'+str(epoch))

    return (params, sort_idx), (Partial(SWAG_pred, params_SWA=params_SWA, params_Sq=params_Sq, D_hat_list=D_hat_list, sort_idx=sort_idx), 
                                jax.vmap(jax.vmap(pytree_fu)), params_SWA, params_Sq, D_hat_list)

 
def test(subkey, solver_vmap, cdata, args):
        
    print('-------------- Testing --------------')

    SWAG_cov_size = 5
    PostPross_Data = PyTree.load(args['path']+'/checkpoints', name='PostPross_Data'+str(args['nepochs']))
    state = PyTree.load(args['path']+'/checkpoints', name='state'+str(args['nepochs']))
    params     = state['params']
    params_SWA, params_Sq, D_hat_list = state['params_SWA'], state['params_Sq'], state['D_hat_list']
    sort_idx = PostPross_Data['sort_idx']

    params_flat, pytree_fu = jax.flatten_util.ravel_pytree(PyTree.extract(params, 0))
    SWAG_samples_vmap = jax.vmap(get_SWAG_sampler(num_samples=5, debug=args['debug']), in_axes=(0, 0, 0, [0,]*SWAG_cov_size), out_axes=(0,0))
    SWAG_pred  = get_SWAG_pred(SWAG_samples_vmap, pytree_fu, solver_vmap, args['debug'])
    subkey = random.split(subkey, num=args['nModels'])
    key, pred_X, pred_cX, params_samples = SWAG_pred(subkey, params_SWA, params_Sq, D_hat_list, sort_idx)
    
    PostPross_Data = {'pred_rX': (pred_X, pred_cX),
                      'sort_idx':sort_idx,
                      'cdata': cdata}
    PyTree.save(PostPross_Data, args['path']+'/checkpoints', name='PostPross_Data_test')

    if 'PDE' in args['path']:
        plot_pde_2Dpred(args, cdata, (pred_X, pred_cX), 5, name='SWAG_test')
        plot_pde_1Dpred(args, cdata, (pred_X, pred_cX), name='SWAG_test')
    else:
        plot_ode_pred(args, cdata, (pred_X, pred_cX), name='SWAG_test')
        plot_ode_all(args, cdata, pred_X)

    return 0