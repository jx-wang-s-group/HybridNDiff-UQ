import jax
import haiku as hk

def activation_fu(activation):
    if activation == 'softplus':
        return jax.nn.softplus
    elif activation == 'leaky_relu':
        return jax.nn.leaky_relu
    elif activation == 'elu':
        return jax.nn.elu
    elif activation == 'relu':
        return jax.nn.relu
    elif activation == 'sigmoid':
        return jax.nn.sigmoid
    elif activation == 'tanh':
        return jax.nn.tanh


class lblock(hk.Module):
    def __init__(self, hidd_size, activation):
        super().__init__()
        self.net = hk.Sequential([
            hk.Linear(hidd_size),
            # hk.LayerNorm(axis=-1,
            #           create_scale=True,
            #           create_offset=True,),
            activation_fu(activation),
            hk.Linear(hidd_size),
        ])
        self.ln = hk.LayerNorm(axis=-1,
                      create_scale=True,
                      create_offset=True,)
    
    def __call__(self, x):
        return self.ln(self.net(x)) + x
        # return self.net(x) + x

class MLP(hk.Module):

  def __init__(self, hidd_size, activation=['leaky_relu',None]):
    super().__init__()
    self.hidd_size = hidd_size
    # mlp = [hk.Linear(hidd_size[0]), activation_fu(activation[0])]
    mlp = [hk.Linear(hidd_size[0])]
    for i in range(len(hidd_size)-1):
        mlp.append(lblock(hidd_size[i], activation[0]))
        mlp.append(activation_fu(activation[0]))
        mlp.append(hk.Linear(hidd_size[i+1]))
    if activation[1]:
        mlp.append(activation_fu(activation[1]))
    self.mlp = hk.Sequential(mlp)

  def __call__(self, x):
    return self.mlp(x)

class MLP_dropout(hk.Module):

  def __init__(self, hidd_size, dropout_rate = 0., activation=['leaky_relu',None]):
    super().__init__()
    self.hidd_size = hidd_size
    self.dropout_rate = dropout_rate
    self.mlpi = hk.Linear(hidd_size[0])
    for i in range(len(self.hidd_size)-1):
        setattr(self,'lbc'+str(i),hk.Sequential([lblock(hidd_size[i], activation[0]),activation_fu(activation[0])]))
        setattr(self,'mlp'+str(i),hk.Linear(hidd_size[i+1]))
    if activation[1]:
        setattr(self,'mlp'+str(i),hk.Sequential([hk.Linear(hidd_size[i+1]),activation_fu(activation[1])]))

  def __call__(self, x, key):
    seq = hk.PRNGSequence(key)
    x = self.mlpi(x)
    for i in range(len(self.hidd_size)-1):
        x = getattr(self,'lbc'+str(i))(x)
        x = hk.dropout(next(seq), self.dropout_rate, x)
        x = getattr(self,'mlp'+str(i))(x)
    return x


class cblock(hk.Module):
    def __init__(self, hidd_channels,ksize, activation):
        super().__init__()
        self.net = hk.Sequential([
            hk.Conv2D(output_channels=hidd_channels, kernel_shape=(ksize,ksize), padding="SAME", with_bias=False, data_format='NCHW'),
            hk.LayerNorm(axis=[-3,-2,-1], create_scale=True, create_offset=True,),
            activation_fu(activation),
            hk.Conv2D(output_channels=hidd_channels, kernel_shape=(ksize,ksize), padding="SAME", with_bias=False, data_format='NCHW'),
        ])
        self.ln = hk.LayerNorm(axis=[-3,-2,-1],
                      create_scale=True,
                      create_offset=True,)
    
    def __call__(self, x):
        # return self.ln(self.net(x)) + x
        return self.net(x) + x

class CNN(hk.Module):

  def __init__(self, hidd_size, ksize, activation=['leaky_relu',None]):
    super().__init__()
    self.hidd_size = hidd_size
    cnn = [hk.Conv2D(output_channels=hidd_size[0], kernel_shape=(ksize,ksize), padding="SAME", with_bias=False, data_format='NCHW'), activation_fu(activation[0])]
    for i in range(len(hidd_size)-1):
        cnn.append(cblock(hidd_size[i], ksize, activation[0]))
        cnn.append(activation_fu(activation[0]))
        cnn.append(hk.Conv2D(output_channels=hidd_size[i+1], kernel_shape=(ksize,ksize), padding="SAME", with_bias=False, data_format='NCHW'))
    if activation[1]:
        cnn.append(activation_fu(activation[1]))
    self.cnn = hk.Sequential(cnn)

  def __call__(self, x):
    return self.cnn(x)