from torch import Tensor, rand


def initial_candidate_generator(n: int, **kwargs) -> Tensor:
    # Generate n initial conditions that satisfy your constraints
    # This is a simple example; you'll need to adapt it to your specific constraints
    samples = rand(n, d)  # d is the dimension of your search space
    while not all(constraint(samples) <= 0):
        samples = rand(n, d)
    return samples
