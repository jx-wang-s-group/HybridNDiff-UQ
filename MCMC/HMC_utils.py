import jax
import jax.numpy as jnp
import jax.random as random

def inference_loop(rng_key, kernel, initial_state, num_samples):
    
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

def inference_loop_multiple_chains(rng_key, kernel, initial_state, num_samples, num_chains):

    @jax.jit
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, info = jax.vmap(kernel)(keys, states)
        return states, states

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

def inference_loop_multiple_chains1(rng_key, kernel, initial_state, num_samples, num_chains, num_loop):

    @jax.jit
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, info = jax.vmap(kernel)(keys, states)
        return states, states

    states_list = []
    rng_key = jax.random.split(rng_key, num_loop)
    for n in range(num_loop):
        print(n)
        keys = jax.random.split(rng_key[n], num_samples//num_loop)
        initial_state, states = jax.lax.scan(one_step, initial_state, keys)
        states_list.append(states.position)

    return states

#################################################################################################    
def get_HMCsamples( neg_logdensity,
                    init_theta,
                    step_size=0.25,
                    inv_mass_matrix=False,
                    num_integration_steps=4,
                    num_samples=1_000, 
                    rng_key=random.PRNGKey(10)):

    if not inv_mass_matrix:
        inv_mass_matrix=jnp.eye(len(init_theta))

    K           = lambda p: 0.5*p.T@inv_mass_matrix@p
    U           = neg_logdensity
    dUdq        = jax.grad(U) # gradient 
    accept_q0   = lambda q0, q1: q0
    accept_q1   = lambda q0, q1: q1

    @jax.jit
    def leapfrog_intg(step_size, q1,p1):
        # leapfrog integration begin
        p1 -= step_size*dUdq(q1)/2 # half-step
        for _ in range(num_integration_steps): 
            q1 += step_size*p1 
            p1 -= step_size*dUdq(q1)  
        p1 += step_size*dUdq(q1)/2 # revise half-step   
        p1 = -1*p1 #flip momentum for reversibility     
        return q1,p1

    def step(carry, rng_key):
        q0 = jnp.copy(carry)
        p0 = random.normal(rng_key[0], shape=q0.shape) 
        q1 = jnp.copy(q0)       
        p1 = jnp.copy(p0)  

        q1,p1 = leapfrog_intg(step_size, q1,p1)
        
        #metropolis acceptance
        H_old = U(q0) + K(p0)
        H_new = U(q1) + K(p1)
        acceptance = H_old - H_new

        event = jnp.log(random.uniform(rng_key[1]))
        accept_q = jax.lax.cond(event <= acceptance, accept_q1, accept_q0, q0,q1)

        carry = accept_q
        store = accept_q
        return carry, store
    
    # generate samples
    keys = random.split(rng_key, 2*num_samples)
    keys = jnp.stack((keys[:num_samples],keys[num_samples:]), axis=-2)
    _, samples = jax.lax.scan(step, init=init_theta, xs=keys, length=num_samples)

    return samples
