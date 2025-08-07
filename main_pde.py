import os, errno, time, copy
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

from UT.unsented_transform import get_2Dunsented_transform_fu
from solver.PDE.rd_pde import *
from solver.diff_eq_solver import get_RHS, get_solver
from NN.NN_trainer_utils import NN_init, cov_init, Phypara_init, get_nnl_fu
from NN.NN_trainer import trainSWAG, test
from MCMC.MCMC_infer import get_MCMC
from utils.utils import PyTree
from utils.plots_pde import plot_pde_wtime

# jax.config.update("jax_debug_nans", True)

HOME = os.getcwd()

def add_err(key, Data):

    key, subkey = random.split(key)
    white_err =  5e-2*random.normal(subkey, shape=Data.shape)
    # space_err =  1.*jnp.array([0.5*(Grid['grid_x']+1) + (Grid['grid_y']+1)]*Data.shape[1])
    Data = Data.at[1:].set(Data[1:] + white_err[1:])#*space_err[1:])
    return key, Data

def get_Data(key, args, **argsdata):
    print("-------------- Generating Data --------------")

    data = {**argsdata}
    data['Grid'] = grid(nNodes = data['nNodes'])
    data['X_shape'] = (2,data['nNodes'],data['nNodes'])

    # ---------------- Reaction-DIffusion data generation --------------------------------------------

    # dx = min(jnp.min(data['Grid']['grid_dx']), jnp.min(data['Grid']['grid_dy']))       # space step
    # data['Tend'] = 5.0             # total time
    # dt = 4.5 * dx**2     # simulation time step
    # data['dt'] = dt     # simulation time step
    data['t_arr']  = jnp.arange(0,data['Tend'],data['dt'])
    data['train_idx'] = jnp.arange(0,int(len(data['t_arr'])*args['train_frac']))

    nsample_times = 5
    # sample_times  = [int(el) for el in jnp.linspace(0, len(data['t_arr']), nsample_times)]

    key, subkey = random.split(key)

    X0 = initial_condotion(subkey, data['Grid'], n1=20, ic=args['InitCond'], length_scale_bounds=(0.5, 10.01))        #-- ic = 'random' or 'GP'
    X0 = jax.image.resize(X0, (2,data['Grid']['nNodes'],data['Grid']['nNodes']), "bilinear")
    if args['diff_dom']:X0 = X0 * (X0 > 0.3)

    # ---------------- Reaction-DIffusion RHS functions ------------------------
    lap1fu, lap2fu = get_laplacian(data['Grid'], UT=False)
    src1fu, src2fu = get_sourceRxn(case='FN', X_shape=X0.shape, UT=False, alpha=-0.005, beta=10.)
    # Coeff   = [[2.8e-4, 1.],
    #            [5.0e-2, 1.]]
    # src1fu, src2fu = get_sourceRxn(case='FN', alpha=0.01, beta=0.25)
    # Coeff   = [[1, 1.],
    #            [100, 1.]]

    # fu_list = [[lap1fu, src1fu],
    #            [lap2fu, src2fu]]
    # RHS = get_RHS(fu_list, Coeff)

    lap1fud = lambda params, time, rX: lap1fu(time, rX)
    lap2fud = lambda params, time, rX: lap2fu(time, rX)
    src1fud = lambda params, time, rX: src1fu(time, rX)
    src2fud = lambda params, time, rX: src2fu(time, rX)
    fu_list = [[lap1fud, src1fud],
               [lap2fud, src2fud]]
    RHSd = get_RHS(fu_list, UP=True, Coeff='from_para')

    # ################################### Deterministic ###################################
    # args['Phyparams'] = {'value':{ 'Coeff':Coeff},
    #              'order':{ 'Coeff':[[0., 0.], [0., 0.]]}}
    params={'cov':jnp.zeros_like(X0), 'Phy':args['Phyparams']}

    if False:
        solver = get_solver(RHSd, X0, 0.001, jnp.arange(0,0.5,0.001), osdesolve = 'rk4')
        X0   = solver(params)[0][0][-1]

    # X = generate_data(RHS, X0, dt, data['t_arr'], osdesolve = args['osdesolve'] )
    solver = get_solver(RHSd, X0, data['dt'], data['t_arr'], osdesolve = args['osdesolve'] , debug = args['debug'])

    starttime = time.time()
    Data   = solver(params)[0][0]
    endtime = time.time()
    print("Gen data time = ",endtime - starttime, " sec")

    data['data_test'] = copy.deepcopy(Data)
    key, Data = add_err(key, Data)

    data['data_train'] = Data[data['train_idx']]

    #------------------- plot  -------------------------
    plot_pde_wtime(data['Grid'], data['t_arr'], data['data_test'], nsample_times, args['path'])

    return key, data


def solve_pdesystem(key, args, data, **argsdata):

    cdata = {**argsdata}
    cdata['SR_shape'] = data['X_shape']
    cdata['Grid'] = grid(nNodes = cdata['nNodes'])
    cdata['X_shape'] = (2,cdata['nNodes'],cdata['nNodes'])

    cdata['t_arr']  = jnp.arange(0,cdata['Tend'],cdata['dt'])
    cdata['train_idx'] = jnp.arange(0,int(len(cdata['t_arr'])*args['train_frac']))
    tskip = int(args['dt'][1]//args['dt'][0])
    # data_train_idx = jnp.arange(0,int(len(data['t_arr'])*args['train_frac']))[tskip*cdata['train_idx']]

    cData = jax.image.resize(data['data_test'], (*data['data_test'].shape[:-2],*cdata['X_shape'][1:]), "bilinear")
    data_test_idx_ext = tskip*jnp.arange(len(cData)//tskip)
    cdata['data_test'] = cData[data_test_idx_ext]

    key, cdata['data_train'] = add_err(key, copy.deepcopy(cdata['data_test'][cdata['train_idx']]))
    
    # cdata['data_train'] = jax.image.resize(data['data_train'], (*data['data_train'].shape[:-2],*cdata['X_shape'][1:]), "bilinear")
    # cdata['data_test']  = jax.image.resize(data['data_test'], (*data['data_test'].shape[:-2],*cdata['X_shape'][1:]), "bilinear")
    # cdata['data_train'] = cdata['data_train'][data_train_idx]
    # cdata['data_test']  = cdata['data_test'][tskip*jnp.arange(len(cdata['data_test'])//tskip)]

    print('Data Mesh = ',data['nNodes'],'x',data['nNodes'],'\t Sim Mesh = ',cdata['nNodes'],'x',cdata['nNodes'])
    print('Data dt   = ',data['dt'],'s',                   '\t Sim dt   = ',cdata['dt'], 's')

    # ################################### Hybrid-UQ ###################################

    #------------------- trainable init  -------------------------
    in_fu, out_fu = get_wraperNN(cdata['X_shape'])
    key, subkey1, subkey2, subkey3 = jax.random.split(key, num=4)
    DNN_fu, Nopt, NNparams, Nopt_state = NN_init(subkey1, args['nModels'], args['nepochs'],args['learning_rate'], layers=[2*data['data_test'].shape[1], 32, 2], cdecay = 0.01, in_fu=in_fu, out_fu=out_fu)
    Copt, Cparams, Copt_state         = cov_init(subkey2, args['nModels'], args['nepochs'],0.1*args['learning_rate'], cdata['X_shape'], cdecay = 0.1)
    if args['train_cof']:
        Popt, Pparams, Popt_state = Phypara_init(subkey3, args['nModels'], args['nepochs'],100*args['learning_rate'], args['Phyparams'], args['mask_phy'], cdecay = 1)

    params    = {'fuNN':NNparams,   'cov':Cparams}
    opt_state = {'fuNN':Nopt_state, 'cov':Copt_state}
    opt_fu    = {'fuNN':Nopt,       'cov':Copt}
    if args['train_cof']:
        params['Phy'] = Pparams
        opt_state['Phy'] = Popt_state
        opt_fu['Phy'] = Popt
    else:
        params['Phy'] = PyTree.combine_copy(args['Phyparams'], l=args['nModels'])

    params_init = copy.deepcopy(params)

    mask_para = {'fuNN':PyTree.set_val(NNparams, val=1),
                 'cov': PyTree.set_val(Cparams , val=1)}
    if args['train_cof']:
        mask_para['Phy'] = args['mask_phy']


    # ---------------- Reaction-DIffusion RHS functions ------------------------
    lap1fu, lap2fu = get_laplacian(cdata['Grid'], UT=True)
    src1fu, src2fu = get_sourceRxn(case='FN', X_shape=cdata['X_shape'], UT=True, alpha=-0.005, beta=10.)


    def lap1fup(params, time, rX, UQ=True): 
        rX  = lap1fu(time, rX)
        rX1 = rX[1] if UQ else jnp.zeros_like(rX[1])
        return rX[0], rX1
    def lap2fup(params, time, rX, UQ=True): 
        rX  = lap2fu(time, rX)
        rX1 = rX[1] if UQ else jnp.zeros_like(rX[1])
        return rX[0], rX1
    def src1fup(params, time, rX): 
        return src1fu(time, rX)
        # return DNN_fu.apply(params['fuNN'], rX)
    def src2fup(params, time, rX): 
        # return src2fu(time, rX)
        return DNN_fu.apply(params['fuNN'], rX)

    fu_list = [[lap1fup, src1fup],
               [lap2fup, src2fup]]
    RHSp = get_RHS(fu_list, UP=True, Coeff='from_para')

    #------------------- train  -------------------------
    params_ts = PyTree.set_val(params, val=0)
    solver = get_solver(RHSp, cdata['data_test'][0], cdata['dt'], cdata['t_arr'], osdesolve = args['osdesolve'] , debug = args['debug'])
    nll_fu = get_nnl_fu(solver, args, data, cdata)
    solver_vmap = jax.vmap(solver, in_axes=(params_ts,), out_axes=((0,0),(0,0)))

    key, subkey = jax.random.split(key)

    if args['NN_train']:
        starttime = time.time()
        (params,sort_idx), SWAG_args = trainSWAG(subkey, nll_fu, solver_vmap, opt_fu, 
                                                params,mask_para,opt_state,params_init,
                                                cdata, args)
        endtime = time.time()
        print("NN train time = ",time.strftime("%H:%M:%S", time.gmtime(endtime - starttime)))

    if args['NN_test']:
        starttime = time.time()
        test(subkey, solver_vmap, cdata, args)
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
                                num_samples=1_000, burnout=000, num_chains=1, path=args['path'])
        # params = PyTree.load('MCMC')

        endtime = time.time()
        print("HMC time = ",time.strftime("%H:%M:%S", time.gmtime(endtime - starttime)))

        (MCMC_X, MCMC_cX), _ = solver_vmap(params)
        PyTree.save({"HMC_pred": (MCMC_X, MCMC_cX), "HMC_params":params }, args['path']+'/checkpoints', name='HMC')

        # plot_ode_pred(data['t_arr'], MCMC_X, MCMC_cX, data['data_train'], data['data_test'], data['train_idx'], args['train_var'], args['path'], name='pred_MCMC', clr ='b')
        # plot_ode_pred(args, data, (MCMC_X, MCMC_cX), name='pred_MCMC', clr ='b')

        # plt.close('all')
        # fig,ax = plt.subplots(3, 1, figsize=(1*5, 3*2))
        # for i, n in enumerate((random.uniform(random.PRNGKey(1),shape=(3,)))*states.shape[1]):
        #     ax[i].plot(states[:,int(n)])
        # plt.savefig(args['path']+'/plots/trace.png')


if __name__ == "__main__":

    key  = random.PRNGKey(123)

    args = {}
    args['debug']     = False

    args['gen_data']  = True
    args['InitCond']  = 'GP'            ## = 'random' or 'GP'
    args['nNodes']    = [20,20]
    args['dt']        = [0.01,0.01]    ## 2nd dt should be int multiple oif 1st one
    args['Tend']      = 5.0
    args['diff_dom']  = False


    args['NN_train']  = True
    args['HMC_train'] = True
    args['train_cof'] = False
    args['NN_test']   = False
    args['train_frac']= 0.5
    args['epochstart']= 0
    args['nepochs']   = 500
    args['nModels']   = 10
    args['plot_depoch']=50
    args['save_depoch']=5000
    args['print_depoch']=10
    args['osdesolve']  = 'rk4'          ## = 'euler' or 'rk4'
    args['learning_rate'] = 1e-2

    args['train_var'] = [1]

    args['prob_name'] = 'RD'            ## option: RD
    args['case_no']   =  0
    args['Heteroscedastic'] = 0

    if args['debug']: print('***************** Running debug mode *****************')

    dir = args['prob_name']+'_C'+str(args['case_no'])+'_D'+str(args['train_var'])+'_Diff'
    print('case = '+dir)
    args['path'] = os.path.join(HOME, "output/PDE/"+dir)
    try:
        os.makedirs(args['path'])
        os.makedirs(args['path']+'/plots')
        os.makedirs(args['path']+'/checkpoints')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


    if args['diff_dom']:
        args['Phyparams'] = {'value':{ 'Coeff':[[28., 1.],
                                                [50., 1.]]},
                            'order':{ 'Coeff':[ [-4., -1.],
                                                [-4., -1.]]}}
    else:
        args['Phyparams'] = {'value':{ 'Coeff':[[2.8, 1.],
                                                [5.0, 1.]]},
                            'order':{ 'Coeff':[ [-4., 0.],
                                                [-2., 0.]]}}
    if args['train_cof']:
        args['mask_phy']  = {'value':{ 'Coeff':[[1, 0],
                                                [1, 0]]},
                            'order':{ 'Coeff':[ [0, 0],
                                                [0, 0]]}}


    if args['gen_data']:
        key, data = get_Data(key, args, nNodes=args['nNodes'][0], dt=args['dt'][0], Tend=args['Tend'])
        PyTree.save(data, args['path']+'/checkpoints', name='Data_'+str(args['nNodes'][0])+'x_'+str(args['dt'][0])+'t')

    if args['NN_train'] or args['NN_test']:
        data = PyTree.load(args['path']+'/checkpoints', name='Data_'+str(args['nNodes'][0])+'x_'+str(args['dt'][0])+'t')
        args['dt'][1] = (args['dt'][1]//args['dt'][0]) * args['dt'][0]
        solve_pdesystem(key, args, data, nNodes=args['nNodes'][1], dt=args['dt'][1], Tend=args['Tend'])
