"""Least squares optimisation of a quadratic using an evolutionary process."""

import random

import numpy as np
import matplotlib.pyplot as plt


class Organism():
    """
    The smallest individual unit of a population.

    Each organism represents a triple of coefficients determined by
    the genetic code associated with the organism.
    """

    def __init__(self, population, genome=None):
        """Create a new organism belonging to a certain population."""
        if not genome:
            genome = {
                gene: random.uniform(alleles[0], alleles[1])
                for gene, alleles in population.gene_pool.items()
            }
        self.genome = genome
        self.population = population

        # ID for debugging only
        self.id = random.randrange(10 ** 6)
        self._fitness = None

    @property
    def fitness(self):
        """Return the fitness of the given organism."""
        if not self._fitness:
            y_hat = self.genome['a'] * self.population.x**2 + \
                self.genome['b'] * self.population.x + \
                self.genome['c']
            self._fitness = -np.mean(np.square(y_hat - self.population.y))
        return self._fitness

    def mutate(self):
        """Mutate the genetic code of an organism."""
        for gene, alleles in self.population.gene_pool.items():
            if self.population.mutation_chance > random.random():
                # Convex combination of current allele and random
                # choice from the gene pool
                p = (1 + random.random()) / 2
                self.genome[gene] = p * self.genome[gene] + \
                    (1-p) * random.uniform(alleles[0], alleles[1])

    @staticmethod
    def breed(mother, father, num_children=2):
        """Combine the genetic code of two organisms."""
        if mother.population is not father.population:
            raise ValueError("cannot breed members of different " +
                             "populations")

        children = []
        for __ in range(num_children):
            genome = {}

            for gene in mother.population.gene_pool.keys():
                # Convex combination of parents' alleles
                p = random.random()
                genome[gene] = p * mother.genome[gene] + \
                    (1-p) * father.genome[gene]

            child = Organism(mother.population, genome)
            child.mutate()

            children.append(child)

        return children


class Population():
    """A collection of organisms."""

    def __init__(self, size, gene_pool, mutation_chance, retain_prop,
                 random_select, ground_truth, seed=None):
        """Create a new population consisting of multiple organisms.
        
        Args:
            size (int): Size of the population.
            gene_pool (dict[str, Tuple(int, int)]): A dictionary where each
                key represents a gene and each value, a two-tuple, giving the
                range of corresponding alleles.
            mutation_chance (float): The chance that each gene will
                independently mutate at 'birth'.
            retain_prop: (float): The proportion of fittest individuals to
                definitely survive at each generation.
            random_select: (float): The probablility of each individual
                that is not retained independently surviving (provided the
                population size has not been surpassed).
            ground_truth: (Array[2, :]): The true pair x, y used for
                evaluating fitness
            seed: (int): Random seed to ensure reproducibility.
        """
        self.size = size
        self.gene_pool = gene_pool
        self.mutation_chance = mutation_chance
        self.retain_prop = retain_prop
        self.random_select = random_select
        self.x = ground_truth[0]
        self.y = ground_truth[1]

        if seed:
            random.seed(seed)

        self.organisms = [Organism(self) for __ in range(size)]
        self.retain_length = int(self.size * self.retain_prop)

    @property
    def average_fitness(self):
        """Return the average fitness of the population."""
        return sum(org.fitness for org in self.organisms) / self.size

    def evolve(self):
        """Evolve the population."""
        ranked_organisms = sorted(self.organisms,
                                  key=lambda org: org.fitness,
                                  reverse=True)

        survived = ranked_organisms[:self.retain_length]
        for org in ranked_organisms[self.retain_length:]:
            if len(survived) >= self.size:
                break
            if self.random_select > random.random():
                survived.append(org)

        # Determine how many children to create
        shortfall = self.size - len(survived)

        children = []
        while len(children) < shortfall:

            mother = random.choice(survived)
            father = random.choice([s for s in survived
                                   if s is not mother])

            children.extend(
                Organism.breed(mother, father,
                               min(2, shortfall - len(children)))
            )

        self.organisms = survived + children

if __name__ == "__main__":
    # Lazy imports
    import numpy as np

    # Problem parameters
    A = -1  # quadratic coeff
    B = 1  # linear coeff
    C = 2  # intercept
    N = 20  # number of samples
    S2 = 0.1  # variance of errors

    # Create data
    np.random.seed(1729)
    x = np.random.normal(1, 1, N)
    y = A * x**2 + B * x + C + np.random.normal(0, np.sqrt(S2), N)

    # Model parameters
    P = 20  # population size
    G = 20  # number of generations
    GP = {
        'a': (-3, 3),
        'b': (-3, 3),
        'c': (-3, 3),
    }  # gene pool
    MC = 0.05  # mutation chance
    RP = 0.4  # retained proportion
    RS = 0.1  # chance of random selection
    GT = (x, y)  # ground truth
    SEED = 42

    # Create population and evolve
    population = quadratic_regression.Population(P, GP, MC, RP, RS, GT, SEED)
    print(f"Initial average fitness: {population.average_fitness:.03f}")
    for i in range(2, G+1):
        population.evolve()
        print(f"Fitness at generation {i}: {population.average_fitness:.03f}")
