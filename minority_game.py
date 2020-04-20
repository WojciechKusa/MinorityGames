import random
import numpy as np
import logging

logger = logging.getLogger()

from agent import Agent

class MinorityGame(object):
	"""docstring for MinorityGame
		N - agents amount
		S - strategies amount
		M - brain of an agent
	"""

	def __init__(self, N, S, M):
		super(MinorityGame, self).__init__()

		if N%2 == 0:
			N += 1

		self.N = N
		self.S = S
		self.M = M
		self.P = 2**M


		self.alpha = (2**self.M)/self.N

		self.agents = []
		self.attendence = []
		self.history = []
		self.historyOccurance = [0]*(2**M)
		self.historyAttendence = [0]*(2**M)
		self.historyOfOneInMinority = [0]*(2**M)
		self.predictability = 0
		self.strategyTable = 1
		self.volatility = 0


		# history as a binary number

		# self.strategyTable = np.zeros( (2**2**M) )

		for i in range(0, N):
			self.agents.append( Agent(S, M, i) )


	def printStatistics(self):
		logger.debug("sum attendence: %f" % (sum(self.attendence)))
		logger.debug("mean attendence: %f" % (np.mean(self.attendence)))
		logger.debug("var attendence: %f" % (np.var(self.attendence)))

		logger.info("volatility: %f" % (np.var(self.attendence)/self.N))

		self.volatility = np.var(self.attendence)/(self.N)
		logger.debug("alpha: %f" % self.alpha)

		for i in range(0, self.P):
			if self.historyOccurance[i] > 0:
				self.predictability += (self.historyAttendence[i]/self.historyOccurance[i])**2
			else:
				self.predictability += 0

		self.predictability = self.predictability/((2**self.M))



	def simulate(self, endTime = 10):

		# create some history
		for i in range(0, self.M):
			self.history.append( random.choice([0, 1]) )

		t = 0
		while t < endTime:
			attendence = 0


			# prepare history
			j = 1
			history = 0
			for i in range(len(self.history) - self.M, len(self.history)):
				history += j*self.history[i]
				j *= 2
			# print ("history:", history)

			self.historyOccurance[history] += 1

			attendence = 0
			# for each agent make a decision and calc attendence
			for agent in self.agents:
				# logger.info("agent.id", agent.id)
				attendence += agent.decide(history)

			self.attendence.append(attendence)
			# logger.info("attendence:", attendence)

			self.historyAttendence[history] += attendence

			if attendence > 0:
				self.history.append( 1 )
				self.historyOfOneInMinority[history] += 1
			elif attendence < 0:
				if -attendence == len(self.agents):
					c = 0
					logger.info("history: %s" % history)
					for agent in self.agents:
						logger.info("agentx: %s %s" % (agent.id, agent.decide(history)))
						c += agent.decide(history)
					logger.info("c = %d", c)

				self.history.append( 0 )
			else:
				c = 0
				for agent in self.agents:
					logger.info("agentx: %s %s" % (agent.id, agent.decide(history)))
					c += agent.decide(history)
				logger.info("c = %d" % c)

			# update strategy payoff
			for agent in self.agents:
				agent.updateStrategies(attendence, history)

			logger.info("simulation time: %d" % t)
			logger.info("  ____  ")
			t += 1
