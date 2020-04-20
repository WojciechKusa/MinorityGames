import matplotlib
# matplotlib.use('Agg')

import random
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger()
logging.basicConfig(level='INFO')


from agent import Agent
from minority_game import MinorityGame


def simulate_phases():
	N = [1, 3, 5, 7, 9, 11, 21, 51, 151, 251]#, 501, 1001, 2001, 5001]
	M = [1, 2, 3, 4, 5]

	alpha = np.zeros(( len(N), len(M) ))
	volatility = np.zeros(( len(N), len(M) ))
	attendence = np.zeros(( len(N), len(M) ))
	loops = 40

	# N = 1001
	# M = 3
	# S = 2
	# mg = MinorityGame(N, S, M)
	# mg.simulate(10)
	# mg.printStatistics()


	# fig = plt.figure()
	for i in range(N):
		logger.info(f"N[i] - {N[i]}")
		for j in range(M):
			logger.info(f"M[j] -{M[j]}")
			# N = 1001
			# M = 3
			S = 2
			for k in range(loops):
				mg = MinorityGame(N[i], S, M[j])
				mg.simulate(30)
				mg.printStatistics()

				logger.debug(f"attendence: {mg.attendence}")

				attendence[i][j] += np.mean(mg.attendence)
				alpha[i][j] += mg.alpha
				volatility[i][j] += mg.volatility

			logger.info(attendence[i][j])
			attendence[i][j] /= loops
			alpha[i][j] /= loops
			volatility[i][j] /= loops
			logger.info(loops)
			logger.info(attendence[i][j])
		plt.plot(alpha[i], volatility[i], '--', marker='x', label=str(N[i]))

		np.savetxt('data/alpha.txt', alpha)
		np.savetxt('data/attendence.txt', attendence)
		np.savetxt('data/volatility.txt', volatility)


	plt.xscale('log')
	plt.yscale('log')
	plt.grid(True)
	plt.legend()
	plt.xlabel('alpha = (2^M)/N', fontsize=18)
	plt.ylabel('volatility = (sigma^2)/N', fontsize=16)
	# plt.show()
	plt.savefig('plots/mg.png')


def simulate_symmetric():
	# symmetric
	N = 2001
	M = 4
	S = 2
	mg = MinorityGame(N, S, M)
	mg.simulate(400)
	mg.printStatistics()

	ii = np.zeros((2**M))
	probabilities = np.zeros((2**M))
	for i in range(2**M):
		ii[i] = i+1
		probabilities[i] = mg.historyOfOneInMinority[i]/mg.historyOccurance[i]

	plt.plot(ii, probabilities, '--', marker='o')
	plt.xlabel('mu', fontsize=18)
	plt.ylabel('P(1|mu)', fontsize=16)
	plt.ylim([0,1])
	plt.xlim([1,2**M])
	plt.grid(True)
	# plt.show()
	plt.savefig('plots/symmetric.png')


def simulate_asymmetric():
	# asymmetric
	loops = 3
	N = 6
	M = 5
	S = 2
	ii = np.zeros((2**M))
	probabilities = np.zeros((2**M))

	for k in range(loops):
		mg = MinorityGame(N, S, M)
		mg.simulate(300)
		mg.printStatistics()

		for i in range(2**M):
			ii[i] = i+1
			if mg.historyOccurance[i] > 0:
				probabilities[i] += mg.historyOfOneInMinority[i]/mg.historyOccurance[i]
			else:
				probabilities[i] += 0

	for i in range(2**M):
		probabilities[i] /= loops

	plt.plot(ii, probabilities, '--', marker='o')
	plt.xlabel('mu', fontsize=18)
	plt.ylabel('P(1|mu)', fontsize=16)
	plt.ylim([0,1])
	plt.xlim([1,2**M])
	plt.grid(True)
	# plt.show()
	plt.savefig('plots/asymmetric.png')


def simulate_predictability():
	# predictability simulations
	# H
	# N = [3, 5, 7, 9, 11]
	N = [1, 3, 5, 7, 9, 11, 21, 51]
	M = [1, 2, 3, 4, 5, 6]

	alpha = np.zeros(( len(N), len(M) ))
	volatility = np.zeros(( len(N), len(M) ))
	attendence = np.zeros(( len(N), len(M) ))
	predictability = np.zeros(( len(N), len(M) ))
	loops = 1

	# N = 1001
	# M = 3
	# S = 2
	# mg = MinorityGame(N, S, M)
	# mg.simulate(10)
	# mg.printStatistics()

	# fig = plt.figure()
	for i in range(N):
		logger.info(N[i])
		for j in range(M):
			logger.info(M[j])
			# N = 1001
			# M = 3
			S = 2
			for k in range(loops):
				mg = MinorityGame(N[i], S, M[j])
				mg.simulate(20)
				mg.printStatistics()

				logger.info(mg.attendence)

				attendence[i][j] += np.mean(mg.attendence)
				alpha[i][j] += mg.alpha
				volatility[i][j] += mg.volatility
				predictability[i][j] += mg.predictability

			# logger.info(attendence[i][j])
			predictability[i][j] /= N[i]

			attendence[i][j] /= loops
			alpha[i][j] /= loops
			volatility[i][j] /= loops
			predictability[i][j] /= loops
			# logger.info(loops)
			logger.info(attendence[i][j])

		# plot each N independently
		plt.plot(alpha[i], predictability[i], '', marker='x', label=str(N[i]))

		np.savetxt('data/alpha.txt', alpha)
		np.savetxt('data/attendence.txt', attendence)
		np.savetxt('data/volatility.txt', volatility)
		np.savetxt('data/predictability.txt', predictability)


	plt.xscale('log')
	# plt.yscale('log')
	plt.grid(True)
	plt.legend()
	plt.xlabel('alpha = (2^M)/N', fontsize=18)
	plt.ylabel('volatility = (sigma^2)/N', fontsize=16)
	# plt.show()
	plt.savefig('plots/predictability.png')



if __name__ == '__main__':
	random.seed(42)

	simulate_phases()
	plt.clf()

	simulate_symmetric()
	plt.clf()

	simulate_asymmetric()
	plt.clf()

	simulate_predictability()
