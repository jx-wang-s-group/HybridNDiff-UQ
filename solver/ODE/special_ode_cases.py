import jax.numpy as jnp

def get_ode_sepcial_case(prob='Linear', case_no=0):

    if 'Linear' in prob:
        if case_no == 0:
            train_var     = [1]
            DNN_fu_list   = [1, 0]
            data_error_in = [0.5, 0.5]
            dt = 0.1
            train_idx = jnp.arange(0,60)
            t_arr  = jnp.arange(0,8,dt)
            X_initial = jnp.array([1.,1.])
        
    elif 'Linhard' in prob:
        if case_no == 0:
            train_var     = [0, 1]
            DNN_fu_list   = [1, 0]
            data_error_in = [0.01, 0.01]
            dt = 0.01
            train_idx = jnp.arange(0,2)
            t_arr  = jnp.arange(0,2000,dt)
            X_initial = jnp.array([0.,1.5])

    elif 'Hamiltonian' in prob:
        if case_no == 0:
            train_var     = [0, 1]
            DNN_fu_list   = [1, 0]
            data_error_in = [0.5, 0.2]
            dt = 0.1
            train_idx = jnp.arange(0,200)
            t_arr  = jnp.arange(0,30,dt)
            X_initial = jnp.array([0.,1.5])

        if case_no == 1000:
            train_var     = [0, 1]
            DNN_fu_list   = [0, 0]
            data_error_in = [0.5, 0.2]
            dt = 0.1
            train_idx = jnp.arange(0,200)
            t_arr  = jnp.arange(0,30,dt)
            X_initial = jnp.array([0.,1.5])

        elif case_no == 11:
            train_var     = [0, 1]
            DNN_fu_list   = [1, 0]
            data_error_in = [0.5, 0.2]
            dt = 0.1
            train_idx = jnp.arange(50,200)
            t_arr  = jnp.arange(0,30,dt)
            X_initial = jnp.array([0.,1.5])

        elif case_no == 12:
            # time sparsity
            train_var     = [0, 1]
            DNN_fu_list   = [1, 0]
            data_error_in = [0.5, 0.2]
            dt = 0.1
            train_idx = jnp.arange(0,200, 8)
            t_arr  = jnp.arange(0,30,dt)
            X_initial = jnp.array([0.,1.5])

        elif case_no == 13:
            train_var     = [0, 1]
            DNN_fu_list   = [1, 0]
            data_error_in = [0.5, 0.2]
            dt = 0.1
            train_idx = jnp.arange(0,100)
            t_arr  = jnp.arange(0,30,dt)
            X_initial = jnp.array([0.,1.5])

        elif case_no == 21:
            train_var     = [0]
            DNN_fu_list   = [1, 0]
            data_error_in = [0.5, 0.2]
            dt = 0.1
            train_idx = jnp.arange(0,200)
            t_arr  = jnp.arange(0,30,dt)
            X_initial = jnp.array([0.,1.5])

        elif case_no == 22:
            train_var     = [1]
            DNN_fu_list   = [1, 0]
            data_error_in = [0.5, 0.2]
            dt = 0.1
            train_idx = jnp.arange(0,200)
            t_arr  = jnp.arange(0,30,dt)
            X_initial = jnp.array([0.,1.5])

        elif case_no == 31:
            train_var     = [0, 1]
            DNN_fu_list   = [1, 0]
            data_error_in = [1., 1.]
            dt = 0.1
            train_idx = jnp.arange(0,200)
            t_arr  = jnp.arange(0,30,dt)
            X_initial = jnp.array([0.,1.5])

        elif case_no == 32:
            train_var     = [0, 1]
            DNN_fu_list   = [1, 0]
            data_error_in = [0.1, 0.1]
            dt = 0.1
            train_idx = jnp.arange(0,200)
            t_arr  = jnp.arange(0,30,dt)
            X_initial = jnp.array([0.,1.5])

        elif case_no == 41:
            train_var     = [1]
            DNN_fu_list   = [1, 0]
            data_error_in = [0.5, 0.2]
            dt = 0.1
            train_idx = jnp.concatenate([jnp.arange(70,150),jnp.arange(150,200)])
            t_arr  = jnp.arange(0,30,dt)
            X_initial = jnp.array([0.,1.5])

        elif case_no == 100:
            train_var     = [0, 1]
            DNN_fu_list   = [0, 1]
            data_error_in = [0.5, 0.2]
            dt = 0.1
            train_idx = jnp.arange(0,200)
            t_arr  = jnp.arange(0,30,dt)
            X_initial = jnp.array([0.,1.5])
            
        elif case_no == 121:
            train_var     = [0]
            DNN_fu_list   = [0, 1]
            data_error_in = [0.5, 0.2]
            dt = 0.1
            train_idx = jnp.arange(0,200)
            t_arr  = jnp.arange(0,30,dt)
            X_initial = jnp.array([0.,1.5])

        elif case_no == 122:
            train_var     = [1]
            DNN_fu_list   = [0, 1]
            data_error_in = [0.5, 0.2]
            dt = 0.1
            train_idx = jnp.arange(0,200)
            t_arr  = jnp.arange(0,30,dt)
            X_initial = jnp.array([0.,1.5])

    elif 'Rossler' in prob:
        if case_no == 0:
            train_var     = [0, 1, 2]
            DNN_fu_list   = [1, 0, 0]
            data_error_in = [0.2, 0.2, 0.01]
            dt = 0.05
            train_idx = jnp.arange(0,200)
            t_arr  = jnp.arange(0,10,dt)
            X_initial = jnp.array([0.2,0.2,0.01])


    return train_var,DNN_fu_list,data_error_in, \
            dt,train_idx,t_arr,X_initial