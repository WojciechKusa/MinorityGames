import random
import numpy as np
import matplotlib.pyplot as plt
###import pylab
import logging

logger = logging.getLogger()
logging.basicConfig(level='DEBUG')


from agent import Agent
from minority_game import MinorityGame



if __name__ == '__main__':

	# N = [3, 5, 7, 9, 11]
	# N = [1, 3, 5, 7, 9, 11, 21, 51, 151, 251, 501, 1001, 2001, 5001]
	# M = [1, 2, 3, 4, 5]

	# alpha = np.zeros(( len(N), len(M) ))
	# volatility = np.zeros(( len(N), len(M) ))
	# attendence = np.zeros(( len(N), len(M) ))
	# loops = 40

	# # N = 1001
	# # M = 3
	# # S = 2
	# # mg = MinorityGame(N, S, M)
	# # mg.simulate(10)
	# # mg.printStatistics()

	# random.seed()

	# # fig = plt.figure()
	# for i in range(0, len(N)):
	# 	logger.info(N[i])
	# 	for j in range(0, len(M)):
	# 		logger.info(M[j])
	# 		# N = 1001
	# 		# M = 3
	# 		S = 2
	# 		for k in range(0, loops):
	# 			mg = MinorityGame(N[i], S, M[j])
	# 			mg.simulate(30)
	# 			mg.printStatistics()

	# 			logger.info(mg.attendence)

	# 			attendence[i][j] += np.mean(mg.attendence)
	# 			alpha[i][j] += mg.alpha
	# 			volatility[i][j] += mg.volatility

	# 		logger.info(attendence[i][j])
	# 		attendence[i][j] /= loops
	# 		alpha[i][j] /= loops
	# 		volatility[i][j] /= loops
	# 		logger.info(loops)
	# 		logger.info(attendence[i][j])
	# 	plt.plot(alpha[i], volatility[i], '--', marker='x', label=str(N[i]))

	# 	np.savetxt('data/alpha.txt', alpha)
	# 	np.savetxt('data/attendence.txt', attendence)
	# 	np.savetxt('data/volatility.txt', volatility)


	# plt.xscale('log')
	# plt.yscale('log')
	# plt.grid(True)
	# plt.legend()
	# plt.xlabel('alpha = (2^M)/N', fontsize=18)
	# plt.ylabel('volatility = (sigma^2)/N', fontsize=16)
	# plt.show()
	# pylab.savefig('mg.png')


	# # symmetric
	# N = 2001
	# M = 4
	# S = 2
	# mg = MinorityGame(N, S, M)
	# mg.simulate(400)
	# mg.printStatistics()

	# ii = np.zeros((2**M))
	# probabilities = np.zeros((2**M))
	# for i in range(0,2**M):
	# 	ii[i] = i+1
	# 	probabilities[i] = mg.historyOfOneInMinority[i]/mg.historyOccurance[i]

	# plt.plot(ii, probabilities, '--', marker='o')
	# plt.xlabel('mu', fontsize=18)
	# plt.ylabel('P(1|mu)', fontsize=16)
	# pylab.ylim([0,1])
	# pylab.xlim([1,2**M])
	# plt.grid(True)
	# # plt.show()
	# pylab.savefig('symmetric.png')

	# pylab.clf()

	# # asymmetric
	# loops = 3
	# N = 6
	# M = 5
	# S = 2
	# ii = np.zeros((2**M))
	# probabilities = np.zeros((2**M))

	# for k in range(0, loops):
	# 	mg = MinorityGame(N, S, M)
	# 	mg.simulate(300)
	# 	mg.printStatistics()

	# 	for i in range(0,2**M):
	# 		ii[i] = i+1
	# 		if mg.historyOccurance[i] > 0:
	# 			probabilities[i] += mg.historyOfOneInMinority[i]/mg.historyOccurance[i]
	# 		else:
	# 			probabilities[i] += 0

	# for i in range(0,2**M):
	# 	probabilities[i] /= loops

	# plt.plot(ii, probabilities, '--', marker='o')
	# plt.xlabel('mu', fontsize=18)
	# plt.ylabel('P(1|mu)', fontsize=16)
	# pylab.ylim([0,1])
	# pylab.xlim([1,2**M])
	# plt.grid(True)
	# # plt.show()
	# pylab.savefig('asymmetric.png')


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

	random.seed()

	# fig = plt.figure()
	for i in range(0, len(N)):
		logger.info(N[i])
		for j in range(0, len(M)):
			logger.info(M[j])
			# N = 1001
			# M = 3
			S = 2
			for k in range(0, loops):
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
####		plt.plot(alpha[i], predictability[i], '', marker='x', label=str(N[i]))

		np.savetxt('data/alpha.txt', alpha)
		np.savetxt('data/attendence.txt', attendence)
		np.savetxt('data/volatility.txt', volatility)
		np.savetxt('data/predictability.txt', predictability)


"""	plt.xscale('log')
	# plt.yscale('log')
	plt.grid(True)
	plt.legend()
	plt.xlabel('alpha = (2^M)/N', fontsize=18)
	plt.ylabel('volatility = (sigma^2)/N', fontsize=16)
	# plt.show()


    ##### pylab.savefig('predictability.png')
"""
