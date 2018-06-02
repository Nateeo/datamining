import random

from deap import base
from deap import creator
from deap import tools

from openrec import ImplicitModelTrainer
from openrec.utils.evaluators import ImplicitEvalManager
from openrec.utils import ImplicitDataset, Dataset
from openrec.utils.evaluators import AUC, Recall, Precision, NDCG
from openrec.utils.samplers import PointwiseSampler

from CombinedRecommender import CombinedRecommender

from ImplicitDatasetWithExplicitConversion import ImplicitDatasetWithExplicitConversion
from PairwiseSamplerWithExplicitConversion import PairwiseSamplerWithExplicitConversion

import numpy as np

# recommender settings

import training_config as tc

if __name__ == "__main__":

	max_user = 10657
	max_item = 4251

	# max_user = 5551
	# max_item = 16980

	# twiddle with these things ###:

	USER = 10656

	MAX_GENERATIONS = 5  # how many generations to go through
	MAX_FITNESS = 1  # the ideal individual
	POP_SIZE = 100  # total individuals
	IND_SIZE = 3  # number of floats in each individual

	# probability of cross-over — swapping weights between indices, i.e mating :^)
	CROSS_OVER_PROBABILITY = 0.05
	# probability of mutation (selected mutation is shuffling)
	MUTATION_PROBABILITY = 0.05


	# RECOMMENDER STUFF
	raw_data = dict()
	raw_data['max_user'] = tc.max_user
	raw_data['max_item'] = tc.max_item
	fileToLoad = 'rec1_time_test.csv'
	raw_data['test_data'] = np.genfromtxt(fileToLoad, delimiter=",", dtype=[
						int, int, float, float, float, int, int, int, int, int], names=True, encoding='utf8')
	# csv = np.genfromtxt('movies_medium.csv', delimiter=",", dtype='int,int,float,bool,float,float', names=True, encoding='ansi')
	# csv = np.genfromtxt('Movies_ratings_small_merged_reduced.csv', delimiter=",", dtype='int,int,float,float,int,int,str,str,float,int,int,str,bool', names=True, encoding='ansi')

	# Permute all the data, then subsection it off - using temp AND THEN numpy

	test1Temp = []
	for entry in raw_data['test_data']:
		if entry['user_id'] == USER:
			test1Temp.append(entry)

	print(test1Temp)
	uniq = np.array(np.unique(test1Temp))
	raw_data['test_1_data'] = uniq

	test_dataset = ImplicitDatasetWithExplicitConversion(
		raw_data['test_1_data'], raw_data['max_user'], raw_data['max_item'], name='Test')

	# maybe consider multiple eval datasets
	# eval_dataset =

	# add evaluators
	recall_evaluator = Recall(recall_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
	precision_evaluator = Precision(precision_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
	ndcg_evaluator = NDCG(ndcg_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
	evaluators = [AUC(), recall_evaluator, precision_evaluator, ndcg_evaluator]
	sampler = PairwiseSamplerWithExplicitConversion(
		dataset=test_dataset, batch_size=tc.batch_size, num_process=3)

	combined_recommender = CombinedRecommender(
		batch_size=tc.batch_size, max_user=tc.max_user, max_item=tc.max_item)

	# need correct train_dataset (to exclude positives) and batch sizes for evaluation
	model_trainer = ImplicitModelTrainer(batch_size=tc.batch_size, test_batch_size=tc.test_batch_size,
										 train_dataset=test_dataset, model=combined_recommender, sampler=sampler)

	model_trainer._eval_manager = ImplicitEvalManager(evaluators=evaluators)
	model_trainer._excluded_positives = {}
	model_trainer._excluded_positives[USER] = set()

	#================================GENETIC ALGORITHM======================================#

	# EVALUATION FUNCTION - change to min


	def evalOneMin(individual):
		# calling _evaluate_full manually w/o sampler should be single process so ok to set then evaluate
		combined_recommender.set_ensemble(individual)
		user = test_dataset.get_unique_user_list()[0]
		print('user')
		print(user)
		print('evaluation')
		print('score full items')
		scores = model_trainer._score_full_items(users=[user])
		print(scores)
		print('pos_samples')
		print(test_dataset.get_interactions_by_user_gb_item(user))
		print('excl_pos_samples')
		print(model_trainer._excluded_positives[USER])
		print('predictions')
		print(scores[0])

		eval_metrics = model_trainer._evaluate_full(test_dataset)
		print(eval_metrics)
		return eval_metrics['AUC']


	def random_single_point_float():
		return round(random.uniform(0, 2), 2)

	# ignore 'No Attribute' errors, creator adds them as static members during runtime


	# target is minimising fitness function output
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
	toolbox.register("mutate", tools.mutShuffleIndexes,
					 indpb=MUTATION_PROBABILITY)
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
		glob_min = 1
		glob_max = 0
		
		while max(fits) < 2 and gen_count < MAX_GENERATIONS:
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
				glob_min = min(glob_min, min(fits))
				glob_max = max(glob_max, max(fits))
				mean = sum(fits) / length
				sum2 = sum(x*x for x in fits)
				std = abs(sum2 / length - mean**2)**0.5
				print("POPULATION DATA")
				print("  Size: %s" % length)
				print("  Min %s" % min(fits))
				print("  Max %s" % max(fits))
				print("  Avg %s" % mean)
				print("  Std %s" % std)
				
		print("=== EVOLUTION END ===")
		print("GLOBAL STATISTICS")
		print("GLOBAL MIN: %s" % glob_min)
		print("GLOBAL MAX: %s" % glob_max)
		best_individual = tools.selBest(pop, 1)[0]
		print("Best individual is %s, %s" %
			  (best_individual, best_individual.fitness.values))
	
	main()
