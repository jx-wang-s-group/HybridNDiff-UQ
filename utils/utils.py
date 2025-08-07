import jax
import jax.numpy as jnp
import pickle

class PyTree():
    
    @staticmethod
    def set_val(pytree, val):
        return jax.tree_util.tree_map(lambda x: val, pytree)

    @staticmethod
    def extract(pytreeb, n):
        return jax.tree_util.tree_map(lambda x: x[n], pytreeb)

    @staticmethod
    def extract_all(pytreeb):
        l = len(jax.tree_util.tree_leaves(pytreeb)[0])
        return [jax.tree_util.tree_map(lambda x: x[i], pytreeb) for i in range(l)], l

    @staticmethod
    def combine(pytree):
        pytreeb   = jax.tree_util.tree_map(lambda *pt: jnp.stack(pt, axis=0), *pytree)
        # pytreeb   = jax.tree_util.tree_map(lambda x: jax.lax.expand_dims(x,[0]), pytree[0])
        # for nM in range(1,len(pytree)):
        #     pytreeb   = jax.tree_util.tree_map(lambda x,y: jnp.concatenate((x,jax.lax.expand_dims(y,[0])), axis=0), pytreeb, pytree[nM])
        return pytreeb

    @staticmethod
    def combine_copy(pytree, l):
        pytreeb   = jax.tree_util.tree_map(lambda x: jax.lax.expand_dims(x,[0]), pytree)
        for _ in range(1,l):
            pytreeb   = jax.tree_util.tree_map(lambda x,y: jnp.concatenate((x,jax.lax.expand_dims(y,[0])), axis=0), pytreeb, pytree)
        return pytreeb

    @staticmethod
    def random_split_like_tree(rng_key, pytree, treedef=None):
        if treedef is None:
            treedef = jax.tree_structure(pytree)
        keys = jax.random.split(rng_key, treedef.num_leaves)
        return jax.tree_unflatten(treedef, keys)
        
    @staticmethod
    def save(pytree, path, name):
        print("Saving "+name+" pytree")
        path = path+'/'+name+'_pytree.hdf5'
        with open(path, 'wb') as file:
            pickle.dump(pytree, file)

    @staticmethod
    def load(path, name):
        print("loading "+name+" parameters")
        try:
            path = path+'/'+name+'_pytree.hdf5'
            with open(path, 'rb') as file:
                pytree = pickle.load(file)
            print('Found '+name+' parameters')
        except FileNotFoundError:
            print('Error: Could not find parameters')
            return
        return  pytree


def laplacian2D(rXi, dx, dy, UT=True):
    """
    Function to computes the discrete Laplace operator of
    a 2D variable on the grid (using a five-point stencil
    finite difference method.)
    """
    Xi, cXi = rXi
    dx, dy   = dx[1:-1,1:-1], dy[1:-1,1:-1]
    Xym, Xyp = Xi[0:-2,1:-1], Xi[2:,1:-1]
    Xxm, Xxp = Xi[1:-1,0:-2], Xi[1:-1,2:]
    Xc = Xi[1:-1,1:-1]
    lap_X =  (Xxm + Xxp - 2 * Xc) / dx**2 + (Xym + Xyp - 2 * Xc) / dy**2 
    
    lap_X = jnp.pad(lap_X,((1,1),(1,1)), constant_values=((0,0),(0,0)))

    if UT:
        cXym, cXyp = cXi[0:-2,1:-1], cXi[2:,1:-1]
        cXxm, cXxp = cXi[1:-1,0:-2], cXi[1:-1,2:]
        cXc = cXi[1:-1,1:-1]
        lap_c =  (cXxm + cXxp + 2**2 * cXc) / dx**4 + (cXym + cXyp + 2**2 * cXc) / dy**4 
        
        lap_c = jnp.pad(lap_c,((1,1),(1,1)), constant_values=((0,0),(0,0)))
    else:
        lap_c = jnp.zeros_like(lap_X)

    return lap_X, lap_c



# def laplacian2D(Xi, dx, dy):
#     """
#     Function to computes the discrete Laplace operator of
#     a 2D variable on the grid (using a five-point stencil
#     finite difference method.)
#     """
#     dx, dy   = dx[1:-1,1:-1], dy[1:-1,1:-1]
#     Xym, Xyp = Xi[0:-2,1:-1], Xi[2:,1:-1]
#     Xxm, Xxp = Xi[1:-1,0:-2], Xi[1:-1,2:]
#     Xc = Xi[1:-1,1:-1]
#     lap_c =  (Xxm + Xxp - 2 * Xc) / dx**2 + (Xym + Xyp - 2 * Xc) / dy**2 
    
#     lap = jnp.pad(lap_c,((1,1),(1,1)), constant_values=((0,0),(0,0)))
#     return lap

def uncertainty(rX):
    X_mean = jnp.mean(rX[0], axis=0)
    # sX_var = jnp.sqrt(jnp.mean(rX[1] + rX[0]**2, axis=0) - X_mean**2)
    vX_ale = jnp.mean(rX[1], axis=0)
    vX_eps = jnp.var(rX[0], axis=0)
    vX_tot = vX_ale + vX_eps

    sX_ale = jnp.sqrt(vX_ale)
    sX_eps = jnp.sqrt(vX_eps)
    sX_tot = jnp.sqrt(vX_tot)
    return X_mean, sX_tot, sX_ale, sX_eps