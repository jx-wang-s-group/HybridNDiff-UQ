import jax
import jax.numpy as jnp
import jax.random as random

def get_update_SWAG(debug=False):
    
    def update_SWAG(n, params, params_SWA, params_Sq):
        params_flat = jax.flatten_util.ravel_pytree(params)[0]
        params_SWA = (n*params_SWA+params_flat) / (n+1)
        params_Sq  = (n*params_Sq+params_flat**2)/(n+1)
        params_dev = params_flat - params_SWA
        return params_SWA, params_Sq, params_dev
    return jax.jit(update_SWAG) if not debug else update_SWAG

def get_SWAG_sampler(num_samples=5, dev=0.1, debug=False):

    def SWAG_samples(key, params_SWA, params_Sq, D_hat_list):

        d = len(params_SWA)
        K = len(D_hat_list)
        key, subkey = random.split(key)
        z = random.normal(subkey, shape=(num_samples,d+K)) * dev
        
        Sigma_diag_Sq = jnp.sqrt(jnp.abs(params_Sq - params_SWA**2))
        D_hat = jnp.array(D_hat_list).T

        def get_samples(z):
            return params_SWA + Sigma_diag_Sq*z[:d]/jnp.sqrt(2) + D_hat @ z[d:]/jnp.sqrt(2*(K-1))
        
        SWA_flat_sample = jax.vmap(get_samples)(z)
        return key, SWA_flat_sample
    return jax.jit(SWAG_samples) if not debug else SWAG_samples

def get_SWAG_pred(SWAG_samples_vmap, pytree_fu, solver_vmap, debug=False):

    def SWAG_pred(subkey, params_SWA, params_Sq, D_hat_list, sort_idx):
        # key, params_samples = SWAG_samples(key[0],params_SWA[0], params_Sq[0], [D_hat_list[0][0],D_hat_list[0][0]])
        key, SWA_flat_sample = SWAG_samples_vmap(subkey, params_SWA, params_Sq, D_hat_list)
        params_samples = jax.vmap(jax.vmap(pytree_fu))(SWA_flat_sample)
        (pred_X, pred_cX), _ = jax.vmap(solver_vmap)(params_samples)
        pred_X, pred_cX = pred_X[sort_idx], pred_cX[sort_idx]
        pred_X, pred_cX = pred_X.reshape(-1,*pred_X.shape[2:]), pred_cX.reshape(-1,*pred_cX.shape[2:])
        return key, pred_X, pred_cX, params_samples
    return jax.jit(SWAG_pred) if not debug else SWAG_pred
