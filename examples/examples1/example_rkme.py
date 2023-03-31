import numpy as np
import learnware.specification as specification


if __name__ == "__main__":
    data_X = np.random.randn(10000, 20)
    spec1 = specification.utils.generate_rkme_spec(X=data_X, gamma=0.1, cuda_idx=-1)
    spec2 = specification.rkme.RKMESpecification()
    spec1.generate_stat_spec_from_data(data_X)
    spec1.save("spec.json")
    
    beta = spec1.get_beta()
    z = spec1.get_z()
    print(type(beta), beta.shape)
    print(type(z), z.shape)
    
    spec2.load("spec.json")
    beta = spec1.get_beta()
    z = spec1.get_z()
    print(type(beta), beta.shape)
    print(type(z), z.shape)
    
    print(spec1.inner_prod(spec2))
    print(spec1.dist(spec2))