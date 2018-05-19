import random

from deap import base
from deap import creator
from deap import tools

from CombinedRecommender import CombinedRecommender
from openrec import ImplicitModelTrainer
from openrec.utils.evaluators import ImplicitEvalManager
from openrec.utils import ImplicitDataset, Dataset
from openrec.utils.evaluators import AUC, Recall, MSE
from openrec.utils.samplers import PointwiseSampler

import numpy as np

# recommender settings

batch_size = 10
test_batch_size = 10
display_itr = 10

max_user = 5551
max_item = 16980

# max_user = 5551
# max_item = 16980

# twiddle with these things ###:

MAX_GENERATIONS = 5  # how many generations to go through
MAX_FITNESS = 0  # the ideal individual
POP_SIZE = 300  # total individuals
IND_SIZE = 3  # number of floats in each individual

# probability of cross-over — swapping weights between indices, i.e mating :^)
CROSS_OVER_PROBABILITY = 0.05
# probability of mutation (selected mutation is shuffling)
MUTATION_PROBABILITY = 0.05


# RECOMMENDER STUFF
raw_data = dict()


raw_data['max_user'] = max_user
raw_data['max_item'] = max_item

raw_data['train_data'] = np.load('dataset/citeulike/user_data_train.npy')
raw_data['val_data'] = np.load('dataset/citeulike/user_data_val.npy')
raw_data['test_data'] = np.load('dataset/citeulike/user_data_test.npy')

combined_recommender = CombinedRecommender(
    batch_size=batch_size, max_user=max_user, max_item=max_item)

csv = np.genfromtxt('Movies_ratings_small_merged_reduced.csv', delimiter=",",
                    dtype='int,int,float,float,int,int,str,str,float,int,int,str,bool', names=True, encoding='utf8')

train_dataset = ImplicitDataset(
    raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')

sampler = PointwiseSampler(
    dataset=train_dataset, batch_size=batch_size, pos_ratio=0.2, num_process=5)

val_dataset = ImplicitDataset(
    raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val')

# add evaluators
evaluators = [AUC()]

model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size,
                                     train_dataset=train_dataset, model=combined_recommender, sampler=sampler)
model_trainer._eval_manager = ImplicitEvalManager(evaluators=evaluators)
model_trainer._exclude_positives([val_dataset])
#================================GENETIC ALGORITHM======================================#

# EVALUATION FUNCTION - change to min


def evalOneMin(individual):
    print('evaluation')
    print(model_trainer._evaluate_full(val_dataset))
    return sum(individual) * random.random(),


def random_single_point_float():
    return round(random.uniform(0, 1), 1)

# ignore 'No Attribute' errors, creator adds them as static members during runtime


creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# for initialising individuals and then population

toolbox.register("attr_float", random_single_point_float)

toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_float, IND_SIZE)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# genetic operators

toolbox.register("evaluate", evalOneMin)
toolbox.register("mate", tools.cxTwoPoint)  # two-point crossover

# p = 0.05 of shuffling mutation
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=MUTATION_PROBABILITY)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    random.seed(64)
    pop = toolbox.population(n=POP_SIZE)
    print("=== EVOLUTION PHASE ===")

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("Evaluated %i individuals" % len(pop))
    fits = [ind.fitness.values[0] for ind in pop]

    gen_count = 0

    while max(fits) < 100 and gen_count < MAX_GENERATIONS:
        gen_count = gen_count + 1
        print('-- GENERATION: %i --' % gen_count)

        offspring = toolbox.select(pop, len(pop))

        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):

              # cross two individuals with probability CXPB
            if random.random() < CROSS_OVER_PROBABILITY:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values
            for mutant in offspring:
                # mutate an individual with probability MUTPB
                if random.random() < MUTATION_PROBABILITY:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

                    # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5

            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
    print("=== EVOLUTION END ===")
    best_individual = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" %
          (best_individual, best_individual.fitness.values))


if __name__ == "__main__":
    main()
