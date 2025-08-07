import jax
import jax.numpy as jnp

from UT.unsented_transform import get_unsented_transform_fu

def def_ode_sys(prob_name = 'Linear', UT=True, **args):
    '''Create system of ODE for given var {prob_name}
    Parameters
    ----------
    prob_name: str
        define the problem to solve

    Returns
    -------
    dXdt_list: list of functions
        list containg dX/dt functions
    dt: float
        time step
    train_idx: jax array
        index of t_arr data that will be used for training
    t_arr: jax array
        time 
    X_initial: jax array
        initial X state values
    '''
    if 'Linear' in prob_name:
        # ---------------------- Linear sytem -------------------
        if 'A' not in args.keys():
            A = jnp.array([ [-1,2],
                            [-2,1]])
            A +=jnp.array([ [0,  0],
                            [0, +0.2]])
        else:
            A = args['A']
                            
        dx0dt_data = lambda t, X: A[0]@X
        dx1dt_data = lambda t, X: A[1]@X
        dXdt_list  = [dx0dt_data, dx1dt_data]

        dt = 0.1                        if 'dt'         not in args.keys() else args['dt']
        train_idx = jnp.arange(20,60)   if 'train_idx'  not in args.keys() else args['train_idx']
        t_arr  = jnp.arange(0,8,dt)     if 't_arr'      not in args.keys() else args['t_arr']
        X_initial = jnp.array([1.,1.])  if 'X_initial'  not in args.keys() else args['X_initial']

    elif 'VolterraLodka' in prob_name:
        # ---------------------- VOLTERRA-LODKA SYSTEMS -------------------
        # a,b,g,d = 0.4, 0.4, 0.1, 0.2
        a,b,g,d = 2/3, 4/3, 1, 1
        dx0dt_data = lambda t, X:  a*X[0] - b*X[0]*X[1]
        dx1dt_data = lambda t, X: -g*X[1] + d*X[0]*X[1]
        dXdt_list  = [dx0dt_data, dx1dt_data]

        dt = 0.1                         if 'dt'         not in args.keys() else args['dt']
        train_idx = jnp.arange(0,100)    if 'train_idx'  not in args.keys() else args['train_idx']
        t_arr  = jnp.arange(0,15,dt)     if 't_arr'      not in args.keys() else args['t_arr']
        X_initial = jnp.array([0.5,1.2]) if 'X_initial'  not in args.keys() else args['X_initial']

    elif 'Hamiltonian' in prob_name:
        # ------------------ HAMILTONIAN SYSTEMS: PENDULUM (H=X[1]**2/2) -------------------
        dx1dt_data = lambda t, X:  X[1]
        dx2dt_data = lambda t, X: -jnp.sin(X[0])

        if UT:
            vmap_dx1dt = jax.vmap(lambda X, args: dx1dt_data(args, X), in_axes=(0, None), out_axes=(0))
            vmap_dx2dt = jax.vmap(lambda X, args: dx2dt_data(args, X), in_axes=(0, None), out_axes=(0))
            UT_dx1dt   = get_unsented_transform_fu(vmap_dx1dt)
            UT_dx2dt   = get_unsented_transform_fu(vmap_dx2dt)
            rhs1       = lambda t, rX: UT_dx1dt(xmean=rX[0], xcov=rX[1], args=t)
            rhs2       = lambda t, rX: UT_dx2dt(xmean=rX[0], xcov=rX[1], args=t)
        else:
            rhs1       = lambda t, rX: (dx1dt_data(t, rX[0]), jnp.zeros_like(rX[0][0]))
            rhs2       = lambda t, rX: (dx2dt_data(t, rX[0]), jnp.zeros_like(rX[0][0]))

        dXdt_list  = [rhs1, rhs2]

        dt = 0.1                        if 'dt'         not in args.keys() else args['dt']
        train_idx = jnp.arange(0,150)   if 'train_idx'  not in args.keys() else args['train_idx']
        t_arr  = jnp.arange(0,30,dt)    if 't_arr'      not in args.keys() else args['t_arr']
        X_initial = jnp.array([0.,1.5]) if 'X_initial'  not in args.keys() else args['X_initial']

    elif 'Linhard' in prob_name:
        # ------------------ LIENHARD SYSTEMS: VAN DER POL EQUATION -------------------
        mu = 8.
        dx0dt_data = lambda t, X:  mu *(X[1] - (X[0] - X[0]**3/3 - X[1]))
        dx1dt_data = lambda t, X:  X[0] / mu
        dXdt_list  = [dx0dt_data, dx1dt_data]

        dt = 0.1                        if 'dt'         not in args.keys() else args['dt']
        train_idx = jnp.arange(0,100)   if 'train_idx'  not in args.keys() else args['train_idx']
        t_arr  = jnp.arange(0,15,dt)    if 't_arr'      not in args.keys() else args['t_arr']
        X_initial = jnp.array([0.,0.2]) if 'X_initial'  not in args.keys() else args['X_initial']

    elif 'Rossler' in prob_name:
        # ------------------ ROESSLER SYSTEMS ------------------------
        a = 0.2   if 'a' not in args.keys() else args['a']
        b = 0.2   if 'b' not in args.keys() else args['b']
        c = 5.7   if 'c' not in args.keys() else args['c']

        dx0dt_data = lambda t, X: -X[1] - X[2]
        dx1dt_data = lambda t, X:  X[0] + a*X[1]
        dx2dt_data = lambda t, X:  b + X[0]*X[2] - c*X[2]
        dXdt_list  = [dx0dt_data, dx1dt_data, dx2dt_data]

        dt = 0.05                               if 'dt'         not in args.keys() else args['dt']
        train_idx = jnp.arange(0,100)           if 'train_idx'  not in args.keys() else args['train_idx']
        t_arr  = jnp.arange(0,5,dt)             if 't_arr'      not in args.keys() else args['t_arr']
        X_initial = jnp.array([0.2, 0.2, 0.2])  if 'X_initial'  not in args.keys() else args['X_initial']

    elif 'Lorentz' in prob_name:
        # ------------------ LORENTZ SYSTEMS ------------------------
        sigma = 10. if 'sigma' not in args.keys() else args['sigma']
        rho = 0.2   if 'rho'   not in args.keys() else args['rho']
        beta = 8/3  if 'beta'  not in args.keys() else args['beta']

        dx0dt_data = lambda t, X: sigma*(X[1] - X[0])
        dx1dt_data = lambda t, X: X[0]*(rho - X[2]) - X[1]
        dx2dt_data = lambda t, X: X[0]*X[1] - beta*X[2]
        dXdt_list  = [dx0dt_data, dx1dt_data, dx2dt_data]

        dt = 0.05                               if 'dt'         not in args.keys() else args['dt']
        train_idx = jnp.arange(0,100)           if 'train_idx'  not in args.keys() else args['train_idx']
        t_arr  = jnp.arange(0,5,dt)             if 't_arr'      not in args.keys() else args['t_arr']
        X_initial = jnp.array([0.2, 0.2, 0.2])  if 'X_initial'  not in args.keys() else args['X_initial']

    return dXdt_list, dt, train_idx, t_arr, X_initial