from typing import Optional

import numpy


class Genetic:

    @staticmethod
    def crossover_uniform(parent1: numpy.ndarray, parent2: numpy.ndarray, prob_crossover: float = 0.5):
        # prob_crossover of 0.0 will not have any crossover (i.e. offspring1 = parent1 and offspring2 = parent2)
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()

        mask = numpy.random.random(size=offspring1.shape) < prob_crossover
        offspring1[mask] = parent2[mask]
        offspring2[mask] = parent1[mask]

        return offspring1, offspring2

    @staticmethod
    def crossover_binary(parent1: numpy.ndarray, parent2: numpy.ndarray, eta: float):
        # Ref: https://github.com/Chrispresso/SnakeAI/blob/master/genetic_algorithm/crossover.py
        """
        This crossover is specific to floating-point representation.
        Simulate behavior of one-point crossover for binary representations.

        For large values of eta there is a higher probability that offspring will be created near the parents.
        For small values of eta, offspring will be more distant from parents

        Equation 9.9, 9.10, 9.11
        """
        rand = numpy.random.random(size=parent1.shape)
        gamma = numpy.empty(shape=parent1.shape)
        gamma[rand <= 0.5] = (2 * rand[rand <= 0.5]) ** (1.0 / (eta + 1))  # First case of equation 9.11
        gamma[rand > 0.5] = (1.0 / (2.0 * (1.0 - rand[rand > 0.5]))) ** (1.0 / (eta + 1))  # Second case

        # Calculate Child 1 chromosome (Eq. 9.9)
        chromosome1 = 0.5 * ((1 + gamma) * parent1 + (1 - gamma) * parent2)
        # Calculate Child 2 chromosome (Eq. 9.10)
        chromosome2 = 0.5 * ((1 - gamma) * parent1 + (1 + gamma) * parent2)

        return chromosome1, chromosome2

    @staticmethod
    def crossover_single_point(parent1: numpy.ndarray, parent2: numpy.ndarray, major='r'):
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()

        if len(parent2.shape) == 1:
            # 1 dimension shape
            (rows,) = parent2.shape
            row = numpy.random.randint(0, rows)
            if major == 'r':
                offspring1[:row] = parent2[:row]
                offspring2[:row] = parent1[:row]
            elif major == 'c':
                pass
            return offspring1, offspring2

        elif len(parent2.shape) == 2:
            # 2 dimensions shape
            rows, cols = parent2.shape
            row = numpy.random.randint(0, rows)
            col = numpy.random.randint(0, cols)
            if major == 'r':
                offspring1[:row, :] = parent2[:row, :]
                offspring2[:row, :] = parent1[:row, :]
            elif major == 'c':
                offspring1[:, :col] = parent2[:, :col]
                offspring2[:, :col] = parent1[:, :col]
            return offspring1, offspring2

        else:
            raise NotImplementedError

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
        mask = numpy.random.random(size=chromosome.shape) < prob_mutation
        # If mu and sigma are defined, create gaussian distribution around each one
        if mu is not None and sigma is not None:
            gaussian_mutation = numpy.random.normal(mu, sigma, size=chromosome.shape)
        else:
            # Otherwise center around N(0,1)
            gaussian_mutation = numpy.random.normal(size=chromosome.shape)

        if scale is not None:
            gaussian_mutation *= scale

        # Update
        chromosome[mask] += gaussian_mutation[mask]
