import jax
import jax.numpy as jnp

def get_RHS(fu_list, UP=False, Coeff=None):
    """
    Function to compUPes the RHS
    """
    if Coeff is None:
        Coeff = jnp.ones((len(fu_list), len(fu_list[0])))

    if Coeff == 'from_para':
        def RHS_UP(params, t, rX):
            rX_dot = [[], []]
            for Cf_rhs_v, Cf_rhs_o, fu_rhs in zip(params['Phy']['value']['Coeff'], params['Phy']['order']['Coeff'], fu_list):
                rXi_dot = [0, 0]
                for Civ, Cio, fui in zip(Cf_rhs_v, Cf_rhs_o, fu_rhs):
                    Ci  =  Civ * 10**Cio
                    rXi = fui(params, t, rX)
                    rXi_dot[0] += rXi[0] * Ci
                    rXi_dot[1] += rXi[1] * Ci**2
                rX_dot[0].append(rXi_dot[0])
                rX_dot[1].append(rXi_dot[1])
            return jnp.stack(rX_dot[0]), jnp.stack(rX_dot[1])
    else:
        def RHS_UP(params, t, rX):
            rX_dot = [[], []]
            for Cf_rhs, fu_rhs in zip(Coeff, fu_list):
                rXi_dot = [0, 0]
                for Ci, fui in zip(Cf_rhs, fu_rhs):
                    rXi = fui(params, t, rX)
                    rXi_dot[0] += rXi[0] * Ci
                    rXi_dot[1] += rXi[1] * Ci**2
                rX_dot[0].append(rXi_dot[0])
                rX_dot[1].append(rXi_dot[1])
            return jnp.stack(rX_dot[0]), jnp.stack(rX_dot[1])
    
    def RHS(t, X):
        X_dot = []
        for Cf_rhs, fu_rhs in zip(Coeff, fu_list):
            Xi_dot = 0
            for Ci, fui in zip(Cf_rhs, fu_rhs):
                Xi_dot += Ci * fui(t, X)
            X_dot.append(Xi_dot)
        return jnp.stack(X_dot)
    
    if UP:
        return RHS_UP
    else:
        return RHS

def get_euler(RHS, UP=False, debug = False):

    def euler_UP(params, t, rX, dt):

        k1 = RHS(params, t, rX)    
        rX = (rX[0] + k1[0] * dt, 
              rX[1] + k1[1] * dt**2)
        return rX, k1

    def euler(t, X, dt):
        k1 = RHS(t, X)
        X  = X + dt * k1
        return X

    if UP:
        return jax.jit(euler_UP) if not debug else euler_UP
    else:
        return jax.jit(euler) if not debug else euler

def get_rk4(RHS, UP=False, debug = False):

    def rk4_UP(params, t, rX, dt):
        h = dt
        k1 = RHS(params, t      , (rX[0]            , rX[1]))
        k2 = RHS(params, t+0.5*h, (rX[0]+0.5*h*k1[0], rX[1]+k1[1]*(0.5*h)**2))
        k3 = RHS(params, t+0.5*h, (rX[0]+0.5*h*k2[0], rX[1]+k2[1]*(0.5*h)**2))
        k4 = RHS(params, t+  1*h, (rX[0]+  1*h*k3[0], rX[1]+k3[1]*(  1*h)**2))

        rX  = (rX[0] +   h/6.0 *     (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]), 
               rX[1] +  (h/6.0)**2 * (k1[1] + 4*k2[1] + 4*k3[1] + k4[1]))
        return rX, k1

    
    def rk4(t, X, dt):
        h = dt
        k1 = RHS(t, X) 
        k2 = RHS(t + h/2, X + h * k1 / 2)
        k3 = RHS(t + h/2, X + h * k2 / 2)
        k4 = RHS(t + h, X + h * k3)
        X  = X +  1.0 / 6.0 * h * (k1 + 2 * k2 + 2 * k3 + k4)
        return X
    
    if UP:
        return jax.jit(rk4_UP) if not debug else rk4_UP
    else:
        return jax.jit(rk4) if not debug else rk4

def generate_data(RHS, X0, dt, t_arr, osdesolve = 'euler'):

    if osdesolve == 'rk4':
        time_step = get_rk4(RHS)
    elif osdesolve == 'euler':
        time_step = get_euler(RHS)

    def step(carry, t):
        X = carry

        # time step
        X = time_step(t, X, dt)

        # Neumann conditions: derivatives at the edges are null.
        X = jnp.pad(X[:,1:-1,1:-1],((0,0),(1,1),(1,1)), mode='edge')

        carry = X
        return carry, carry
    
    _, store = jax.lax.scan(step, init=X0, xs=t_arr[1:])

    return store


def get_solver(RHS, X0, dt, t_arr, pde = True, osdesolve = 'euler', debug = False):
    debug = False

    if osdesolve == 'rk4':
        time_step = get_rk4(RHS, UP=True, debug = debug)
    elif osdesolve == 'euler':
        time_step = get_euler(RHS, UP=True, debug = debug)

    if debug:
        def solver(params):

            # rX_init  = (X0, jnp.abs(params['cov']))
            rX_init  = (X0, jnp.abs(jnp.array([params['cov'][i]*jnp.ones_like(X0[i]) for i in range(2)])))
            X, cX = [], []
            rX = rX_init

            for t in t_arr[1:]:

                # time step
                rX, dXdt = time_step(params, t, rX, dt)
                rX = jnp.clip(rX[0], a_min=-10., a_max=10.), jnp.clip(rX[1], a_min=-100., a_max=100.)

                # Neumann conditions: derivatives at the edges are null.
                if pde:
                    rX = (jnp.pad(rX[0][:,1:-1,1:-1],((0,0),(1,1),(1,1)), mode='edge'), 
                          jnp.pad(rX[1][:,1:-1,1:-1],((0,0),(1,1),(1,1)), mode='edge'))

                X.append(rX[0])
                cX.append(rX[1])

            rX = jnp.array(X), jnp.array(cX)
            rX = jnp.concatenate([jnp.expand_dims(rX_init[0], axis=0),rX[0]]), jnp.concatenate([jnp.expand_dims(rX_init[1], axis=0),rX[1]])
            return rX, dXdt 
    else:
        @jax.jit
        def solver(params):

            @jax.checkpoint
            def step(carry, t):
                rX = carry

                # time step
                rX, dXdt = time_step(params, t, rX, dt)
                rX = jnp.clip(rX[0], a_min=-10., a_max=10.), jnp.clip(rX[1], a_min=-100., a_max=100.)

                # Neumann conditions: derivatives at the edges are null.
                if pde:
                    rX = (jnp.pad(rX[0][:,1:-1,1:-1],((0,0),(1,1),(1,1)), mode='edge'), 
                          jnp.pad(rX[1][:,1:-1,1:-1],((0,0),(1,1),(1,1)), mode='edge'))

                carry = rX
                store = (rX, dXdt)
                return carry, store

            # rX_init  = (X0, jnp.abs(params['cov']))
            rX_init  = (X0, jnp.abs(jnp.array([params['cov'][i]*jnp.ones_like(X0[i]) for i in range(2)])))
            _,store = jax.lax.scan(f=step, init=rX_init, xs=t_arr[1:])
            rX, dXdt = store
            rX = jnp.concatenate([jnp.expand_dims(rX_init[0], axis=0),rX[0]]), jnp.concatenate([jnp.expand_dims(rX_init[1], axis=0),rX[1]])
            return rX, dXdt    

    return solver
