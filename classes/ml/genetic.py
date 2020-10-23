from typing import List, Optional

import numpy


class Genetic:

    @staticmethod
    def crossover(parent1: numpy.ndarray, parent2: numpy.ndarray, eta: float):
        # Ref: https://github.com/Chrispresso/SnakeAI/blob/master/genetic_algorithm/crossover.py
        """
        This crossover is specific to floating-point representation.
        Simulate behavior of one-point crossover for binary representations.

        For large values of eta there is a higher probability that offspring will be created near the parents.
        For small values of eta, offspring will be more distant from parents

        Equation 9.9, 9.10, 9.11
        """
        rand = numpy.random.random(parent1.shape)
        gamma = numpy.empty(parent1.shape)
        gamma[rand <= 0.5] = (2 * rand[rand <= 0.5]) ** (1.0 / (eta + 1))  # First case of equation 9.11
        gamma[rand > 0.5] = (1.0 / (2.0 * (1.0 - rand[rand > 0.5]))) ** (1.0 / (eta + 1))  # Second case

        # Calculate Child 1 chromosome (Eq. 9.9)
        chromosome1 = 0.5 * ((1 + gamma) * parent1 + (1 - gamma) * parent2)
        # Calculate Child 2 chromosome (Eq. 9.10)
        chromosome2 = 0.5 * ((1 - gamma) * parent1 + (1 + gamma) * parent2)

        return chromosome1, chromosome2

    @staticmethod
    def mutate(chromosome: numpy.ndarray, prob_mutation: float,
               mu: Optional[float] = None, sigma: Optional[float] = None,
               scale: Optional[float] = None) -> None:
        # Ref: https://github.com/Chrispresso/SnakeAI/blob/master/genetic_algorithm/mutation.py
        """
        Perform a gaussian mutation for each gene in an individual with probability, prob_mutation.

        If mu and sigma are defined then the gaussian distribution will be drawn from that,
        otherwise it will be drawn from N(0, 1) for the shape of the individual.
        """
        # Determine which genes will be mutated
        mutation_array = numpy.random.random(chromosome.shape) < prob_mutation
        # If mu and sigma are defined, create gaussian distribution around each one
        if mu is not None and sigma is not None:
            gaussian_mutation = numpy.random.normal(mu, sigma, size=chromosome.shape)
        else:
            # Otherwise center around N(0,1)
            gaussian_mutation = numpy.random.normal(size=chromosome.shape)

        if scale:
            gaussian_mutation[mutation_array] *= scale

        # Update
        chromosome[mutation_array] += gaussian_mutation[mutation_array]
