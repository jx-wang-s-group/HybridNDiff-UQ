import os, errno, time, copy
import jax
import jax.numpy as jnp
import jax.random as random

from solver.ODE.ode_sys import def_ode_sys
from solver.diff_eq_solver import get_RHS, get_solver
from solver.ODE.special_ode_cases import get_ode_sepcial_case
from NN.NN_trainer import NN_init, cov_init, get_nnl_fu, trainSWAG, test
from MCMC.MCMC_infer import get_MCMC
from utils.plots_ode import get_std_pred, plot_ode_pred, plot_data, plot_ode_val
from utils.utils import PyTree

import matplotlib.pyplot as plt

# from jax.config import config
# config.update("jax_debug_nans", True)
# config.parse_flags_with_absl()

HOME = os.getcwd()

def solve_odesystem(args):

    data = {'Grid':[]}

    key = jax.random.PRNGKey(10)
    
    if args['debug']: print('***************** Running debug mode *****************')

    ####################### Generate data ##################################
    
    # dXdt_list, dt, train_idx, t_arr, X0 = def_ode_sys(prob = prob)
    args['train_var'],DNN_fu_list,data_error_in, dt, train_idx, t_arr, X0 = get_ode_sepcial_case(args['prob_name'], args['case_no'])
    dXdt_list, dt, data['train_idx'], data['t_arr'], X0 = def_ode_sys(args['prob_name'], UT=False, dt=dt, train_idx=train_idx,t_arr=t_arr,X_initial=X0)

    
    dir = args['prob_name']+'_C'+str(args['case_no'])+'_D'+str(args['train_var'])+'_N'+str(DNN_fu_list)
    dir += '_heto' if args['Heteroscedastic'] else '_homo'
    print('case = '+dir)
    args['path'] = os.path.join(HOME, "output/ODE/"+dir)
    try:
        os.makedirs(args['path'] )
        os.makedirs(args['path'] +'/plots')
        os.makedirs(args['path'] +'/checkpoints')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    rhs1  = lambda params, time, rX: dXdt_list[0](time, rX)
    rhs2  = lambda params, time, rX: dXdt_list[1](time, rX)
    fu_list = [[rhs1], [rhs2]]
    RHSp = get_RHS(fu_list, UP=True)
    solver = get_solver(RHSp, X0, dt, data['t_arr'], pde=False, osdesolve = args['osdesolve'] , debug = args['debug'])
    
    params={'cov':jnp.zeros_like(X0)}
    Data   = solver(params)[0][0]

    data['data_train'] = jnp.array(Data)[data['train_idx']]
    key, subkey = jax.random.split(key)
    std_error = jax.random.normal(subkey, data['data_train'].shape)
    if args['Heteroscedastic']:
        std_error = std_error *jnp.repeat(jnp.arange(data['data_train'].shape[0]).reshape(data['data_train'].shape[0],1),2,axis=1)/data['data_train'].shape[0] *5
    assert(len(data_error_in) == data['data_train'].shape[-1])
    error = jnp.stack([data_error_in[i]*std_error[:,i] for i in range(data['data_train'].shape[-1])], axis=-1)
    data['data_train'] += error
    data['data_test']  = jnp.array(Data)

    PyTree.save(data, args['path']+'/checkpoints', name='Data')
    plot_data(data['t_arr'], data['data_train'], data['data_test'], data['train_idx'], args['path'])

    ####################### hybrid-UQ model ##################################
    in_fu  = lambda x: x
    out_fu = lambda y: y.at[1].set(jax.nn.softplus(y[1]))

    key, subkey1, subkey2, subkey3 = jax.random.split(key, num=4)
    DNN_fu, Nopt, NNparams, Nopt_state     = NN_init(subkey1, args['nModels'], args['nepochs'],args['learning_rate'], layers=[2*data['data_train'].shape[-1], 8,2], in_fu=in_fu, out_fu=out_fu)
    Copt, Cparams, Copt_state             = cov_init(subkey2, args['nModels'], args['nepochs'],args['learning_rate'], X0.shape)

    if args['Heteroscedastic']:
        in_fu  = lambda x: x
        out_fu = lambda y: jax.nn.relu(y-0.)
        DNN_cov, cNopt, cNNparams, cNopt_state = NN_init(subkey3, args['nModels'], args['nepochs'],args['learning_rate'], layers=[2*data['data_train'].shape[-1]+1, 8, data['data_train'].shape[-1]], in_fu=in_fu, out_fu=out_fu)
    else:
        DNN_cov, cNopt, cNNparams, cNopt_state = [None]*4

    params    = {'fuNN':NNparams,   'cov':Cparams,    'cNN':cNNparams}
    opt_state = {'fuNN':Nopt_state, 'cov':Copt_state, 'cNN':cNopt_state}
    opt_fu    = {'fuNN':Nopt,       'cov':Copt,       'cNN':cNopt}

    params_init = copy.deepcopy(params)

    mask_para = {}

    dXdt_list = def_ode_sys(args['prob_name'], UT=True, dt=dt, train_idx=train_idx,t_arr=t_arr,X_initial=X0)[0]

    rhsnn = lambda params, time, rX: DNN_fu.apply(params['fuNN'], jnp.concatenate(rX))
    fu_list = [fu_list[i] if nni == 0 else [rhsnn] for i, nni in enumerate(DNN_fu_list)]
    RHSp = get_RHS(fu_list, UP=True)
    solver = get_solver(RHSp, X0, dt, data['t_arr'], pde=False, osdesolve = args['osdesolve'] , debug = args['debug'])

    params_ts = PyTree.set_val(params, val=0)
    nll_fu  = get_nnl_fu(solver, args, data)
    solver_vmap = jax.vmap(solver, in_axes=(params_ts,), out_axes=((0,0),(0,0)))

    ####################### Train NN model ##################################

    if args['NN_train']:
        starttime = time.time()

        key, subkey = jax.random.split(key)
        (params,sort_idx), SWAG_args = trainSWAG(subkey, nll_fu, solver_vmap, opt_fu, 
                                                 params,mask_para,opt_state,params_init,
                                                 data, args)

        endtime = time.time()
        print("NN time = ",time.strftime("%H:%M:%S", time.gmtime(endtime - starttime)))
        (UT_X, UT_cX), _ = solver_vmap(params)
        DE_UT_X, DE_UT_cX = UT_X[sort_idx], UT_cX[sort_idx]

        key, subkey = jax.random.split(key)
        subkey = random.split(subkey, num=args['nModels'])
        SWAG_pred = SWAG_args[0]
        _, UT_X, UT_cX, _ = SWAG_pred(subkey)

    if args['NN_test']:
        starttime = time.time()
        test(subkey, solver_vmap, data, args)
        endtime = time.time()
        print("NN test time = ",time.strftime("%H:%M:%S", time.gmtime(endtime - starttime)))


    ####################### Train HMC model ##################################

    if args['HMC_train']:
        print("Full MCMC inference Started =============================================")
        starttime = time.time()

        state = PyTree.load(args['path']+'/checkpoints', name='state'+str(args['nepochs']))
        params= state['params']

        params = jax.tree_util.tree_map(lambda X: X[0], params)
        sampler = get_MCMC( step_size = 1e-4,
                            num_integration_steps=30, 
                            MCMC = 'HMC')

        # params = jax.tree_util.tree_map(lambda X: X[0], params_init)
        # sampler = get_MCMC( step_size = 1e-5,
        #                     num_integration_steps=10, 
        #                     MCMC = 'HMC')

        params, states = sampler(nll_fu, params, params_init,
                                num_samples=100_000, burnout=000, num_chains=1, path=args['path'])
        # params = PyTree.load('MCMC')

        endtime = time.time()
        print("HMC time = ",time.strftime("%H:%M:%S", time.gmtime(endtime - starttime)))

        (MCMC_X, MCMC_cX), _ = solver_vmap(params)
        PyTree.save({"HMC_pred": (MCMC_X, MCMC_cX), "HMC_params":params }, args['path']+'/checkpoints', name='HMC')

        # plot_ode_pred(data['t_arr'], MCMC_X, MCMC_cX, data['data_train'], data['data_test'], data['train_idx'], args['train_var'], args['path'], name='pred_MCMC', clr ='b')
        plot_ode_pred(args, data, (MCMC_X, MCMC_cX), name='pred_MCMC', clr ='b')

        plt.close('all')
        fig,ax = plt.subplots(3, 1, figsize=(1*5, 3*2))
        for i, n in enumerate((random.uniform(random.PRNGKey(1),shape=(3,)))*states.shape[1]):
            ax[i].plot(states[:,int(n)])
        plt.savefig(args['path']+'/plots/trace.png')

    if args['HMC_train'] and args['NN_train']:
        plot_ode_val(args, data, (UT_X, UT_cX), (MCMC_X, MCMC_cX))

if __name__ == "__main__":

    args = {}
    args['debug']     = False
    args['epochstart']= 0000
    args['nepochs']   = 10_000
    args['nModels']   = 10
    args['plot_depoch']=500
    args['save_depoch']=10000
    args['print_depoch']=500
    args['osdesolve'] = 'rk4'
    args['learning_rate'] = 0.01
    
    args['NN_train']  = True
    args['HMC_train'] = True
    args['NN_test']   = False

    args['prob_name'] = 'Hamiltonian'  ## option: Linear, VolterraLodka, Hamiltonian, Linhard, Rossler, Lorentz
    args['case_no']   =  0
    args['Heteroscedastic'] = 0

    solve_odesystem(args)