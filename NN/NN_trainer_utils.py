import jax
import jax.random as random
import jax.numpy as jnp
import jax.scipy.stats as stats
import optax
import haiku as hk

from NN.NN_utils import MLP
from utils.utils import PyTree

class MLP_Net():
    def __init__(self, layers = [4, 4,8,1], inout_fu = [lambda y: y]*2) -> None:
        self.input_size = layers[0]
        self.inout_fu   = inout_fu
        self.DbinModel  = hk.transform(lambda x: MLP(layers[1:], activation=['elu',None])(x))
        
    def get_NNparams(self, key=1337):
        self.seq = hk.PRNGSequence(key)
        representative_input = jnp.ones((self.input_size,))
        NNparams = self.DbinModel.init(next(self.seq), representative_input)
        return NNparams

    def apply(self,NNparams,x):
        x = self.inout_fu[0](x)
        y = self.DbinModel.apply(NNparams,None,x)
        return self.inout_fu[1](y)


def NN_init(key, nModels, nepochs, learning_rate, layers, cdecay = 0.01, **args):
    
    cosdecay = optax.cosine_decay_schedule(learning_rate, decay_steps=nepochs//2, alpha=cdecay)
    nodecay  = optax.constant_schedule(learning_rate*cdecay)
    jointopt = optax.join_schedules([cosdecay, nodecay], boundaries=[nepochs//2])
    Nopt     = optax.adam(learning_rate=jointopt)
    NNparams, Nopt_state = [], []

    in_fu  = (lambda x: x) if 'in_fu'  not in args.keys() else args['in_fu']
    out_fu = (lambda y: y) if 'out_fu' not in args.keys() else args['out_fu']

    DNN_fu = MLP_Net(layers = layers, inout_fu = [in_fu, out_fu])
    for nM in range(nModels):
        key, subkey = random.split(key)
        NNparams.append(DNN_fu.get_NNparams(key=subkey))
        Nopt_state.append(Nopt.init(NNparams[nM]))

    NNparams   = PyTree.combine(NNparams)
    Nopt_state = PyTree.combine(Nopt_state)
    return DNN_fu, Nopt, NNparams, Nopt_state

def cov_init(key, nModels, nepochs, learning_rate, X_shape, cdecay=0.01):

    cosdecay = optax.cosine_decay_schedule(learning_rate, decay_steps=nepochs//2, alpha=cdecay)
    # nodecay  = optax.constant_schedule(learning_rate*cdecay)
    # jointopt = optax.join_schedules([cosdecay, nodecay], boundaries=[nepochs//2])
    # cosdecay = optax.cosine_decay_schedule(learning_rate, decay_steps=30, alpha=cdecay)
    # nodecay  = optax.constant_schedule(1e-10*cdecay)
    # jointopt = optax.join_schedules([cosdecay, nodecay], boundaries=[30])

    Copt     = optax.adam(learning_rate=cosdecay, b1=0.8)
    Cparams, Copt_state = [], []
    cov_in = []

    for nM in range(nModels):
        key, subkey = random.split(key)
        # cov_in_ = 1e-2*random.uniform(subkey, shape=(X_shape[0],), minval=0.5, maxval=1.5)
        # cov_in_ = 1e-1/6 *jnp.ones((X_shape[0],))
        cov_in_ = 1e-2*random.uniform(subkey, shape=(X_shape[0],))
        # Cparams.append(jnp.stack([cov_in[i]*jnp.ones_like(X_initial[i]) for i in range(len(cov_in))]))
        # Copt_state.append(Copt.init(Cparams[nM]))
        cov_in.append(cov_in_)
        Copt_state.append(Copt.init(cov_in[nM]))

    Cparams    = PyTree.combine(cov_in)
    Copt_state = PyTree.combine(Copt_state)
    return Copt, Cparams, Copt_state

def Phypara_init(key, nModels, nepochs, learning_rate, Phyparams, mask_phy, cdecay=0.01):

    cosdecay = optax.cosine_decay_schedule(learning_rate, decay_steps=nepochs//2, alpha=cdecay)
    # nodecay  = optax.constant_schedule(learning_rate*cdecay)
    # jointopt = optax.join_schedules([cosdecay, nodecay], boundaries=[nepochs//2])
    Popt     = optax.adam(learning_rate=cosdecay, b1=0.8)
    Pparams, Popt_state = [], []

    mask_phy, _        = jax.flatten_util.ravel_pytree(mask_phy)
    phyflat, pytree_fu = jax.flatten_util.ravel_pytree(Phyparams)

    for nM in range(nModels):
        for i, ele in enumerate(mask_phy):
            if ele == 1:
                key, subkey = random.split(key)
                phyflat = phyflat.at[i].set(random.uniform(subkey, minval=1., maxval=100.))
        Pparams.append(pytree_fu(phyflat))
        Popt_state.append(Popt.init(Pparams[nM]))
                
    Pparams    = PyTree.combine(Pparams)
    Popt_state = PyTree.combine(Popt_state)
    return Popt, Pparams, Popt_state

def get_nnl_fu(solver, args, data, cdata=False):

    data_train = cdata['data_train'] if cdata else data['data_train']
    train_idx  = cdata['train_idx']  if cdata else data['train_idx']
    
    def nll_fu(params, params_init, epoch_ratio):
        rX ,_   = solver(params)
        pred_X  = rX[0][train_idx]
        pred_cX = rX[1][train_idx]

        pred_cX_aug = epoch_ratio*pred_cX + (1-epoch_ratio)*jnp.mean(pred_cX, axis=0)
    
        # nll = - jnp.sum(stats.norm.logpdf(data_train, pred_X, jnp.sqrt(pred_cX)), axis=0) \
        #         / jnp.sum(jnp.abs(data_train))
        
        # nll = - jnp.sum(stats.norm.logpdf(data_train, pred_X, jnp.sqrt(jnp.mean(pred_cX, axis=0))), axis=0) \
        #         / jnp.sum(jnp.abs(data_train))
        
        nll = - jnp.mean(stats.norm.logpdf(data_train, pred_X, jnp.sqrt(pred_cX_aug)), axis=0)# \
                # / jnp.sum(jnp.abs(data_train))
        
        prior = 0
        # if 'fuNN' in params_init.keys():
        #     params_flat = jax.flatten_util.ravel_pytree(params['fuNN'])[0]
        #     prior -= 1e-4*jnp.sum(stats.norm.logpdf(params_flat))

        # if 'Phy' in params_init.keys():
        #     params_flat = jax.flatten_util.ravel_pytree(params['Phy'])[0]
        #     params_init_flat = jax.flatten_util.ravel_pytree(params_init['Phy'])[0]
        #     prior -=1e-3 *jnp.sum(stats.norm.logpdf(params_flat, params_init_flat, jnp.ones_like(params_init_flat)))

        # return jnp.sum(sum(nll[i] for i in args['train_var'])) + prior
        return jnp.mean( sum(nll[i] for i in args['train_var']) /len(args['train_var']) ) + prior
        
    return nll_fu

def get_update_fu(opt, nll_fu, mask_para, debug = False):
    
    def update_fu(params, opt_state, params_init, epoch_ratio):
        if debug:
            loss = nll_fu(params, params_init, epoch_ratio)

        loss, grads  = jax.value_and_grad(nll_fu)(params, params_init, epoch_ratio)
        if 'Phy' in mask_para.keys():
            grads['Phy'] = jax.tree_util.tree_map(lambda g, m: g*m, grads['Phy'], mask_para['Phy'])
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -10., 10.), grads)

        for key in opt.keys():
            if opt[key] is not None:
                update, opt_state[key] = opt[key].update(grads[key], opt_state[key])
                params[key] = optax.apply_updates(params[key], update)

        return params, opt_state, loss
    return jax.jit(update_fu) if not debug else update_fu

 