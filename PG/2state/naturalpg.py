import numpy as np
import sys


def returngrad(theta, state, action, featvec):
        tr1 = np.dot(theta, featvec[(1,1)])
        tr2 = np.dot(theta, featvec[(1,2)])

        v1 = max(tr1, tr2)

        tr3 = np.dot(theta, featvec[(2,1)])
        tr4 = np.dot(theta, featvec[(2,2)])

        v2 = max(tr3, tr4)

        c1 = np.exp(tr1 - v1)
        c2 = np.exp(tr2 - v1)

        c3 = np.exp(tr3 - v2)
        c4 = np.exp(tr4 - v2)

        if state == 1 and action == 1:
                return featvec[(1,1)] - ( (c1*featvec[(1,1)] + c2*featvec[(1,2)])/(c1 + c2))
        elif state == 1 and action == 2:
                return featvec[(1,2)]  - ( (c1*featvec[(1,1)] + c2 * featvec[(1,2)])/(c1 + c2))

        elif state == 2 and action == 1:
                return featvec[(2,1)] - ( (c3*featvec[(2,1)] + c4 * featvec[(2,2)])/(c3 + c4))
        elif state == 2 and action == 2:
                return featvec[(2,2)] - ( (c3*featvec[(2,1)] + c4 * featvec[(2,2)])/(c3 + c4))


def converged(theta1, theta2):
	eps = 0.008
#	print abs(theta1 - theta2)
#	print theta1, theta2
#	print
	return (abs(theta1 - theta2) < eps).all()

def get_estimate(theta, E, H, model, featvec):
	noi= 1
	cc = 0
	flag = 0
	newest = np.zeros(len(theta))
	prev = np.zeros(len(theta))
	curtheta = np.copy(theta)
	data = []
	while(cc < noi):
#		curtheta = theta + alpha * prev
		data =  sampletraj(curtheta , E, H, model, featvec)
		rews = avgrew(data)
		## we need 
		## 		policy derivatives psi_k_avg  <psi_k>
		## 		fisher matrix Ftheta = avg of (sum of psi_k) * (sum of psi_k)T
		## 		vanilla gradient  = avg of {(sum of psi_k) * (sum of rew)}
		phi_ks = []
		rew_ks = []
		print np.mean(np.array(rews))
		for i in range(len(data)):
			temp_phi = np.zeros(len(theta))
			rew_ks.append(rews[i])
			for j in range(len(data[i])):
				temp_phi += returngrad(curtheta, data[i][j][0], data[i][j][1], featvec) 
			phi_ks.append(temp_phi)

		Ftheta = np.zeros((len(theta), len(theta)))
		elig_phi = np.zeros(len(theta))
		vanillagrad = np.zeros(len(theta))	
		avr = sum(rew_ks)/float(len(rew_ks)) #no further processing

		for i in range(len(phi_ks)):
			Ftheta += np.outer(phi_ks[i], phi_ks[i])
			vanillagrad += phi_ks[i] * rew_ks[i]
			elig_phi += phi_ks[i]
		
	
		Ftheta /= len(phi_ks)
		vanillagrad /= len(phi_ks)
		elig_phi /= len(phi_ks)	
		
		M = len(phi_ks)
		Finv = np.linalg.pinv(Ftheta)
		Q = 1.0 + np.dot(elig_phi ,   np.dot( np.linalg.pinv(M*Ftheta - np.outer(elig_phi, elig_phi) ), elig_phi )   )
		Q = Q/float(M)
		b = Q * (avr - np.dot( elig_phi,  np.dot(Finv, vanillagrad)  ) )
		
		natgrad = np.dot(Finv, vanillagrad - elig_phi *b )
		cc += 1
		return natgrad

def avgrew(data):
	lst = []
	for i in range(len(data)):
		val = 0.0
		count = 0.0
		for j in range(len(data[i])):
			val += data[i][j][2]
			count += 1
		lst.append(val/count)
	return lst
def get_next_action(cs, theta, featvec):
	#print
	#print "current state is:", cs
	t1 = np.dot(theta, featvec[(cs, 1)])
	t2 = np.dot(theta, featvec[(cs, 2)])
		
	v = max(t1, t2)	
	val1 = np.exp(t1 - v)
	val2 = np.exp(t2 - v)
	a = np.array([val1, val2])
	a /= sum(a)
	val = np.random.random()
	action = -1
	if val < a[0]:
		action =  1
	else:
		action = 2
	#print "values are:", a[0], a[1]
	#print "flip gave:", val, "action chosen:", action
	
	return action
	
def sampletraj(theta, D, H, model, featvec):
	lst = []
	for i in range(D):
	    tl = []
	    cs = np.random.randint(1,3)
            for j in range(H):
		action = get_next_action(cs, theta, featvec)
		rew, ns = model[(cs, action)]
		tl.append((cs, action, rew, ns))
		cs = ns
	    lst.append(tl)	
	return lst


def main():
	
	
	npp = 4 #number of policy parameters
	theta = np.random.random(npp)
	#theta = np.array([0.5, 0.5, 0.5, 0.5])
	featvec = {(1,1) : np.array([1.0,0.0,0.0,0.0]),  (1,2): np.array([0.0, 1.0, 0.0,0.0]), (2,1): np.array([0.0,0.0,1.0,0.0]), (2,2):np.array([0.0,  0.0, 0.0, 1.0])}
	
	model = {(1,1):(2.0, 1), (1, 2):(0.0, 2), (2,1):(0.0, 1), (2,2):(1.0, 2)}

	E = 10 #number of episodes
	H = 50 # horizon length
	
	total = 200 # total number of iterations
	alpha = 0.18
	i = 1
	p = 0.005
	while (i < total):
		newt = get_estimate(theta, E, H, model, featvec)
	#	print "new estimate became", newt
		theta = (1 + p)*theta + alpha * newt
		i += 1
#		print "++++++++++++++++++++++++++++"
	#for i in range(total):
	#	newest = get_estimate(theta, E, H, model, featvec)
	#	theta = theta + alpha * newest
	#	data = sampletraj(theta, 1, H, model, featvec)
	#	print "average reward is ", avgrew(data)
	
	#curtheta = [ 3.84964882, -3.07000536, -0.21214966 , 0.51792267]
#	data = sampletraj(theta, 1, H, model, featvec)
#	print avgrew(data)[0]
if __name__ == "__main__":
	main()	
