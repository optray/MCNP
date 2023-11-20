class Config_E1(object):
    device = 'cuda:2'
    num_test = 200
    num_val = 200
    num_train = 1000
    total_time = 2
    record_steps = 200
    b = 0.1
    N = 10
    kappa = 0.005
    size = 64
    sup_size = 1024

class Config_E2(object):
    device = 'cuda:2'
    num_test = 200
    num_val = 200
    num_train = 1000
    total_time = 2
    record_steps = 200
    b = 0.1
    N = 10
    kappa = 0.01
    size = 64
    sup_size = 1024