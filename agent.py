import random
import numpy as np
import logging

logger = logging.getLogger()


class Agent(object):
	"""docstring for Agent"""

	def __init__(self, S, M, id):
		super(Agent, self).__init__()

		self.agentPayoff = 0
		self.agentDecision = 0
		self.strategiesPayoff = dict()

		# multiply because rand not generate big odd numbers
		while len(self.strategiesPayoff) < S:
			self.strategiesPayoff[int(np.random.uniform(0, (2**2**M) - 1))] = 0

		self.id = id


	def decide(self, history):

		maxVal = max(self.strategiesPayoff.values())
		maxVals = []
		for k in iter(self.strategiesPayoff):
			if self.strategiesPayoff[k] == maxVal:
				maxVals.append((k, maxVal))

		strategy = random.choice(maxVals)

		logger.debug("strategy: %s", (strategy))
		logger.debug("cross: %f " % (strategy[0] & 2**history))

		if strategy[0] & 2**history == 2**history:
			self.agentDecision = 1
			return 1
		else:
			logger.debug("str: %f " % (strategy[0]))
			logger.debug("hist: %f" % (2**history))
			self.agentDecision = -1
			return -1

	def updateStrategies(self, attendence, history):
		for i in self.strategiesPayoff:
			if ((attendence < 0) and (i & 2**history == 2**history)) or ((attendence > 0) and (i & 2**history != 2**history)):
				self.strategiesPayoff[i] += 1
			else:
				self.strategiesPayoff[i] -= 1

		if self.agentDecision*attendence < 0:
			self.agentPayoff += 1
		else:
			self.agentPayoff -= 1
