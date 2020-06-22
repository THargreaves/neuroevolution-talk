"""Neural network tuning using an evolutionary process."""

import random

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow import random as tfr


# Load mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Preprocess predictors
X_train = X_train.reshape(60000, 784).astype('float32') / 255.
X_test = X_test.reshape(10000, 784).astype('float32') / 255.
# Preprocess response
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


class Organism():
    """
    The smallest individual unit of a population.

    Each organism represents a neural network model determined by
    the genetic code associated with the organism.
    """

    def __init__(self, population, genome=None):
        """Create a new organism belonging to a certain population."""
        if not genome:
            genome = {
                gene: random.choice(alleles)
                for gene, alleles in population.gene_pool.items()
            }
        self.genome = genome
        self.population = population

        # ID for debugging only
        self.id = random.randrange(10 ** 6)
        self._fitness = None
        self.model = self.create_model(genome)

    @property
    def fitness(self):
        """Return the fitness of the given organism."""
        # Cached for effiency
        if not self._fitness:
            self._fitness = self.train_and_evaluate()
        return self._fitness

    def train_and_evaluate(self):
        """Train this organism's model and report the final accuarcy."""
        if self.population.verbose:
            print(f"Training model {self.id}")
        # For prototype, only train for one epoch
        self.model.fit(X_train, y_train,
                       batch_size=1024,
                       epochs=1,
                       verbose=False,
                       validation_data=(X_test, y_test))

        score = self.model.evaluate(X_test, y_test, verbose=False)
        return score[1]  # accuracy

    def mutate(self):
        """Mutate the genetic code of an organism."""
        for gene, alleles in self.population.gene_pool.items():
            if self.population.mutation_chance > random.random():
                self.genome[gene] = random.choice(alleles)

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
                genome[gene] = random.choice((
                    mother.genome[gene], father.genome[gene]
                ))

            child = Organism(mother.population, genome)
            child.mutate()

            children.append(child)

        return children

    @staticmethod
    def create_model(genome):
        """Create a neural network model based on given genome."""
        model = Sequential()

        for i in range(genome['num_layers']):
            layer_args = {
                'units': genome['num_neurons'],
                'activation': genome['activation'],
                'use_bias': genome['use_bias']
            }
            if i == 0:
                layer_args['input_shape'] = (784,)
            model.add(Dense(**layer_args))
            if genome['dropout_rate'] > 0:
                model.add(Dropout(genome['dropout_rate']))

        model.add(Dense(10, activation='softmax'))

        model.compile(
            loss='categorical_crossentropy',
            optimizer=genome['optimiser'],
            metrics=['accuracy']
        )

        return model


class Population():
    """A collection of organisms."""

    def __init__(self, size, gene_pool, mutation_chance, retain_prop,
                 random_select, seed=None, verbose=True):
        """Create a new population consisting of multiple organisms.
        
        Args:
            size (int): Size of the population.
            gene_pool (dict[str, list]): A dictionary where each
                key represents a gene and each value, a list, giving the
                set of corresponding alleles.
            mutation_chance (float): The chance that each gene will
                independently mutate at 'birth'.
            retain_prop: (float): The proportion of fittest individuals to
                definitely survive at each generation.
            random_select: (float): The probablility of each individual
                that is not retained independently surviving (provided the
                population size has not been surpassed).
            seed: (int): Random seed to ensure reproducibility.
            verbose: (bool): If `True`, print ID before training model
        """
        self.size = size
        self.gene_pool = gene_pool
        self.mutation_chance = mutation_chance
        self.retain_prop = retain_prop
        self.random_select = random_select
        self.verbose = verbose

        if seed:
            random.seed(seed)
            tfr.set_seed(seed)

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
    # Model parameters
    P = 20  # population size
    G = 20  # number of generations
    GP = {
        'num_neurons': [32, 64, 128, 256, 512, 768, 1024, 1536],
        'num_layers': [1, 2, 3, 4, 5, 6],
        'use_bias': [True, False],
        'dropout_rate': [0., .1, .2, .3, .4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimiser': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
    }  # gene pool
    MC = 0.05  # mutation chance
    RP = 0.4  # retained proportion
    RS = 0.1  # chance of random selection
    SEED = 42

    # Create population and evolve
    population = Population(P, GP, MC, RP, RS, SEED)
    print(f"Initial average fitness: {population.average_fitness:.03f}")
    for i in range(2, G+1):
        population.evolve()
        print(f"Fitness at generation {i}: {population.average_fitness:.03f}")
