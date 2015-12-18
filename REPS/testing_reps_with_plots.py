import numpy as np
import scipy.optimize as sop
from scipy.misc import logsumexp

## normalize the policy passed as numpy array
def normalize_policy(pol):
	row, col = pol.shape
	for i in range(row):
		pol[i,:] /= sum(pol[i,:])
	
## returns a map with #keys = #of states
def get_unit_features(n):
	temp = np.eye(n, n) ## for bias
	hm = {}
	for i in range(1, n + 1): ## for one indexing
		#temp[i-1][-1] = 1.0 ## attaching one bias
		hm[i] = temp[i-1]
	return hm

class repsPolicy:
	
	def __init__(self, S, A, epsilon, T, model):
		self.S = S ## number of states
		self.A = A ## number of actions
		self.epsilon = epsilon ## the KL parameter
		self.T = T ##length of trajectory
		self.policy = np.zeros((S, A))

		## value for each state
		self.valuest = {}
		for i in range(1, self.S + 1):
			self.valuest[i] = 0.0
	
		model = {(1, 1): (2, 1), (1, 2):(0, 2), (2, 1):(0, 1), (2, 2):(1, 2)}
		self.model = model

		## assigning random policy at the beginning	
#		for i in range(self.S):
#			self.policy[i,:] = np.random.rand(A)
		
		self.policy[0,0] = 0.5
		self.policy[0,1] = 0.5
		self.policy[1,0] = 0.5
		self.policy[1,1] = 0.5

#		normalize_policy(self.policy)
	
		self.dataset = []  ## list of 4-tuples
		self.features = {(1,1) : np.array([1.0,0.0,0.0,0.0]),  (1,2): np.array([0.0, 1.0, 0.0,0.0]), (2,1): np.array([0.0,0.0,1.0,0.0]), (2,2):np.array([0.0,  0.0, 0.0, 1.0])}

		self.theta = np.zeros(self.S*self.A) ## the critic parameter for value
		self.etahat = 1.0
#		self.bellmanerr = {} ## bellman error for each state-action pair
#		self.featdiff = {} ## feature difference for each state-action pair
		
	def getmaps(self, val):
		N  = len(self.dataset)
		berr = {}
		fdif = {}
	
		savisits = {}
		sumrew = {}
		sumfeat = {}
		sumdots = {}
		## initializing the maps
		for i in range(1, self.S + 1):
			for j in range(1, self.A + 1):
				savisits[(i, j)] = 0.0
				sumrew[(i, j)] = 0.0
				sumfeat[(i, j)] = np.zeros(self.S*self.A ) # for bias
				sumdots[(i, j)] = 0.0
			
				berr[(i,j)] = 0.0
				fdif[(i,j)] = np.zeros(self.S*self.A ) # for bias
	
		## now populating the 3 maps
		for i in range(N - 1):
			curstate = self.dataset[i][0]
			curaction = self.dataset[i][1]
			curreward = self.dataset[i][2]
			nextstate = self.dataset[i][3]
			nextstateaction = self.dataset[i+1][1]

			savisits[(curstate, curaction)] += 1.0
			sumrew[(curstate, curaction)] += float(curreward)
			sumfeat[(curstate, curaction)] += (self.features[nextstate, nextstateaction] - self.features[curstate, curaction])
			sumdots[(curstate, curaction)] += np.dot((self.features[nextstate, nextstateaction] - self.features[curstate, curaction]), val)

		## now populating the berr and fdif maps
		for i in range(1, self.S + 1):
			for j in range(1, self.A + 1):
				if savisits[(i,j)] == 0.0:
					savisits[(i,j)] = 1.0
				berr[(i,j)] = (sumrew[(i,j)] + sumdots[(i,j)])/savisits[(i,j)]
				fdif[(i,j)] = sumfeat[(i,j)] / savisits[(i,j)]
			
		return [berr, fdif]		

	def dualfunction(self, param):
		
		vect = param[:self.S*self.A ] # 
		etahat = param[-1]
		[bellmanerr, featdiff] = self.getmaps(vect)
		
		N = len(self.dataset)
		bellr = []
		for i in range(N):
			curstate = self.dataset[i][0]
			curaction = self.dataset[i][1]
		#	curreward = self.dataset[i][2]
		#	nextstate = self.dataset[i][3]
			bellr.append(etahat * bellmanerr[(curstate, curaction)])
	
		return (self.epsilon/etahat) + (logsumexp(bellr, b = (1.0/N))/etahat)


	def get_next_action(self, cs, theta):
       		#print
        	#print "current state is:", cs

        	val1 = np.exp(np.dot(theta, self.features[(cs,1)]))
        	val2 = np.exp(np.dot(theta, self.features[(cs,2)]))
        	a = np.array([val1, val2])
        	a /= sum(a)
        	val = np.random.random()
        	action = -1
        	if val < a[0]:
                	action =  1
        	else:
                	action = 2

	        return action

	
	def getdata(self):
		
	    self.dataset = [] ## delete the previous contents 
	    for starts in range(30):
		#curstate = 1
		curstate = np.random.randint(1, self.S + 1)
		
		for i in range(self.T):
			temp = np.cumsum(self.policy[curstate - 1]) / sum(self.policy[curstate - 1])
			randraw = np.random.rand()
			nt = np.where(randraw < temp)[0]
			#print "nt is ", nt
			curaction = nt[0] + 1 ## choosing the minimum and doing inc for 1-indexing
			
			cureward, nextstate = self.model[(curstate, curaction)]
			self.dataset.append((curstate, curaction, cureward, nextstate))
			curstate = nextstate		
	
	def getavgreward(self):
		R = 0.0
		for i in range(len(self.dataset)):
			R += self.dataset[i][2]		
		return R/float(len(self.dataset))

	
	def updatepolicy(self, param):
		vect = param[:self.S*self.A ]  ## for bias
		etahat = param[-1]
		berr, fdif = self.getmaps(vect)
		print "bellman error is", berr	
		for i in range(1, self.S + 1):
			for j in range(1, self.A + 1):
				self.policy[i-1, j-1] *= np.exp(berr[(i, j)] * etahat)
		print "okay so update policy looks like:", self.policy
		normalize_policy(self.policy)

		self.theta = vect
		self.etahat = etahat
#		for i in range(1,self.S+1):
#			self.valuest[i] = np.dot(self.theta, self.features[i])
	
	def printit(self):
		print
		print "-----current policy-----"
		print self.policy
		print "-----current values-----"
		print self.valuest
		print "-----current theta------"
		print self.theta
		print "-----current eta------"
		print self.etahat

		berr, fdif = self.getmaps(self.theta)
		print "--------- bellman error -----------"
		print berr

	def printdataset(self):
		print self.dataset
	
def main():
	T = 100
	eps = 0.1
	S = 2
	A = 2
	## (state, action) : (reward, nextstate)
	model = {(1, 1): (1, 1), (1, 2):(0, 2), (2, 1):(2, 2), (2, 2):(0, 1)}

#	model = {(1,1): (0, 2), (1, 2):(2, 1), \
#		 (2,1): (0, 3), (2, 2):(2, 1), \
#		 (3,1): (0, 4), (3, 2):(2, 1), \
#		 (4,1): (0, 5), (4, 2):(2, 1), \
#		 (5,1): (10, 5), (5, 2): (2, 1)}

	p = repsPolicy(S, A, eps,  T, model)

	print p.features
	num = 50
#	bnds = ((None, None), (None, None), (None, None), (None, None), (None,None), (0.01, None))

	#bnds = ((None, None), (None, None), (None, None), (0.01, None)) #with bias
	bnds = ((None, None), (None, None),(None, None), (None, None), (0.00001, 10))
#	p.printit()	
	for i in range(num):
	
		p.printit()
		p.getdata()
	#	p.printdataset()
		guessval = np.random.rand(5)
		#result = sop.minimize(p.dualfunction, guessval, method = "SLSQP", jac = p.gradientval)
	#	result = sop.minimize(p.dualfunction, guessval, method = "L-BFGS-B", jac = p.gradientval, bounds = bnds)
	#	result = sop.minimize(p.dualfunction, guessval, method = "Nelder-Mead")
		result = sop.minimize(p.dualfunction, guessval, method = "L-BFGS-B", bounds = bnds)

		print result
		p.updatepolicy(result.x)
		print "yeh reward hai kya", p.getavgreward()

#	p.printit()
		
if __name__ == "__main__":
	main()
