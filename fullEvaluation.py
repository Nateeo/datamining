import pickle
import numpy as np
from EvolveCombinedRecommender import run_full_eval

if __name__ == '__main__':

	final_results = {}

	TEST_USERS = [12, 971, 1136, 1346, 1799, 2350, 2627, 3278, 4073, 4294, 4860, 5512, 5974, 6320, 6357, 6680, 7107, 7144, 8563, 10656]

	for user in TEST_USERS:
		final_results[user] = run_full_eval(user)
		print('done user %s' % user)
		
	print('COMPLETED')
	print(final_results)

	print('saving results...')
	pickle.dump(final_results, open('final_results.p', 'wb'))
		