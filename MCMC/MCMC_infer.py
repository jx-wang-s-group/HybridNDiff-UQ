import multiprocessing
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import blackjax
import jax.random as random
import pickle

from MCMC.HMC_utils import inference_loop, inference_loop_multiple_chains
from utils.utils import PyTree

def get_MCMC(step_size,num_integration_steps, MCMC = 'HMC'):

    if MCMC == 'HMC':
        def HMC_sampler(nll_fu, params, params_init,
                        num_samples, burnout, num_chains, path):
            # if multiprocessing.cpu_count() < num_chains: raise ValueError("num_chains is greater than no. of cpu")

            para_flatten, pytree_fu = jax.flatten_util.ravel_pytree(params)

            # @jax.jit
            # def logdensity_fn(state):
            #     params = pytree_fu(state)
            #     ll = -nll_fu(params, params_init, 1)
            #     l_prior = jnp.sum(stats.norm.logpdf(jnp.array([jax.tree_util.tree_leaves(state)]), 0, 10))
            #     return l_prior + ll

            @jax.jit
            def logdensity_fn(state):
                params = pytree_fu(state)
                ll = jnp.exp(-nll_fu(params, params_init, 1))
                l_prior = jnp.mean(stats.norm.pdf(jnp.array([jax.tree_util.tree_leaves(state)]), 0, 10))
                return 100*(l_prior + ll)
            
            # Build the kernel
            # step_size = 1e-5
            # num_integration_steps=100
            inverse_mass_matrix = jnp.array([1.]*len(para_flatten))
            hmc  = blackjax.hmc(logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps)

            # Initialize the state
            initial_states = jnp.array([para_flatten]*num_chains)
            initial_states = jax.vmap(hmc.init, in_axes=(0))(initial_states)

            # Iterate
            rng_key = random.PRNGKey(99)
            _, rng_key = random.split(rng_key)
            states_hmcs = inference_loop_multiple_chains(rng_key, hmc.step, initial_states, num_samples, num_chains)
            # states_hmcs = inference_loop_multiple_chains1(rng_key, hmc.step, initial_states, num_samples, num_chains, 10)

            states   = (jnp.array(jax.tree_util.tree_leaves(states_hmcs.position.block_until_ready()[burnout:]))).reshape(-1,len(para_flatten))
            params = jax.vmap(pytree_fu)(states)
            # PyTree.save(params, path+'/checkpoints', name='HMC_params')
            return params, states
        return HMC_sampler