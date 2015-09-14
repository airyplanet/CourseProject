class DataParameters:
    """class, containing the parameters of data being generated"""
    def __init__(self, sigma_f, sigma_l, l):
        self.sigma_f, self.sigma_l, self.l = sigma_f, sigma_l, l


class CommonParameters:
    """class, containg the common parameters of the data"""
    def __init__(self, num_of_examples, dim, x0, x1, density):
        self.n, self.d, self.x0, self.x1, self.density = num_of_examples, dim, x0, x1, density


class ModelParameters:
    """class, containing the hyper-parameters of the model prior distribution"""
    def __init__(self, sigma_f, sigma_l, l):
        self.sigma_f, self.sigma_l, self.l = sigma_f, sigma_l, l

common_params = CommonParameters(num_of_examples=20, dim=1, x0=-5, x1=5, density=100)
data_params = DataParameters(sigma_f=1.0, sigma_l=0.0, l=0.5)
model_params = ModelParameters(sigma_f=1.0, sigma_l=0.0, l=0.5)