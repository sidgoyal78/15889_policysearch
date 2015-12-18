import numpy as np
import sys

def reward(x):
        position = x[0]
        # bound for position; the goal is to reach position = 0.45
        bpright  = 0.45

        r = -1
        f = False
        if  position >= bpright:
            r = 100
            f = True

        return (r,f)


def returnindexofstate(statelist, cur):
	v = (statelist - cur)**2
	m = []
	for i in range(v.shape[0]):
		m.append(np.linalg.norm(v[i,:]))
	m = np.array(m)
	return np.argmin(m)

def getinitialstate():
        initial_position = -0.5
        initial_speed    =  0.0
        return  np.array([initial_position,initial_speed])


def doaction(force, cur):
	position = cur[0]
	speed = cur[1]	
	# position bound
	bpleft = -1.5
	# speed bound
	bsleft = -0.07
	bsright = 0.07
	
        speedt1= speed + (0.001*force) + (-0.0025 * np.cos( 3.0*position) )
        speedt1= speedt1 * 0.999 # thermodynamic law, for a more real system with friction.

        if speedt1<bsleft:
        	speedt1=bsleft
        elif speedt1>bsright:
        	speedt1=bsright

        post1 = position + speedt1

        if post1<=bpleft:
        	post1=bpleft
        	speedt1=0.0

        xp = np.array([post1,speedt1])
        return xp

	
def buildactionlist():
	return np.array([-1.0, 0.0, 1.0])

def buildstatelist():
	xdiv = (0.55-(-1.5))   / 10.0;
	xpdiv = (0.07-(-0.07)) / 5.0;
	
	x = np.arange(-1.5, 0.5, xdiv)
	xp = np.arange(-0.07, 0.071, xpdiv)
	
	N = len(x)
	M = len(xp)
	
	arr = np.zeros((N*M, 2))
	
	count = 0
	for i in range(N):
		for j in range(M):	
			arr[count, :] = np.array([x[i], xp[j]])
			count += 1
	return arr
 
def returngrad(theta, cs, action):
	vec = []
	vec.append(np.concatenate((1*cs, 0*cs, 0*cs), axis = 0))
	vec.append(np.concatenate((0*cs, 1*cs, 0*cs), axis = 0))
	vec.append(np.concatenate((0*cs, 0*cs, 1*cs), axis = 0))
	
	val1 = np.dot(theta, vec[0])
	val2 = np.dot(theta, vec[1])
	val3 = np.dot(theta, vec[2])
	
#	a = max(val1, val2, val3)
	a = 0
	c1 = np.exp(val1 - a)
	c2 = np.exp(val2 - a)
	c3 = np.exp(val3 - a)
#	print "c values", c1, c2, c3
#	print "vec[action] is ", vec[action]
#	print "bachka kucha is ", -((c1 * vec[0] + c2 * vec[1] + c3 * vec[2])/(c1 + c2 + c3))	
#	print
	est = vec[action] - ((c1 * vec[0] + c2 * vec[1] + c3 * vec[2])/(c1 + c2 + c3))
#	print "--- estimate is ==", est
	return est

def converged(theta1, theta2):
	eps = 0.005
	print abs(theta1 - theta2)
	print theta1, theta2
	print
	return (abs(theta1 - theta2) < eps).all()
def get_estimate(theta, E, H, statelist, actionlist):
  noi= 1
  cc = 0
  flag = 0
  newest = np.zeros(len(theta))
  prev = np.zeros(len(theta))
  curtheta = np.copy(theta)
  data = []
  while(cc < noi):
#   curtheta = theta + alpha * prev
    data =  sampletraj(curtheta , E, H, statelist, actionlist)
    rews = avgrew(data)
    ## we need 
    ##    policy derivatives psi_k_avg  <psi_k>
    ##    fisher matrix Ftheta = avg of (sum of psi_k) * (sum of psi_k)T
    ##    vanilla gradient  = avg of {(sum of psi_k) * (sum of rew)}
    phi_ks = []
    rew_ks = []
    print np.mean(np.array(rews))
    for i in range(len(data)):
      temp_phi = np.zeros(len(theta))
      rew_ks.append(rews[i])
      for j in range(len(data[i])):
        temp_phi += returngrad(curtheta, data[i][j][0], data[i][j][1])
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
		#lst.append(val)
		lst.append(val/count)
	return lst
def get_next_action_index(cs, theta):
	vec = []
	vec.append(np.concatenate((1*cs, 0*cs, 0*cs), axis = 0))
	vec.append(np.concatenate((0*cs, 1*cs, 0*cs), axis = 0))
	vec.append(np.concatenate((0*cs, 0*cs, 1*cs), axis = 0))
	
	val1 = np.dot(theta, vec[0])
	val2 = np.dot(theta, vec[1])
	val3 = np.dot(theta, vec[2])

	m1 =0
	#m1 = max(val1, val2, val3)
	c1 = np.exp(val1 - m1)
	c2 = np.exp(val2 - m1)
	c3 = np.exp(val3 - m1)

	a = np.array([c1, c2, c3])
	a /= sum(a)

#	print a
	val = np.random.random()
	actionidx = np.where((val <= a.cumsum()) == True)[0][0]	
	return actionidx
	
def sampletraj(theta, D, H, statelist, actionlist):
	lst = []
	for i in range(D):
	    tl = []
	    cs = getinitialstate()  #note, current state is a array having 2 elements 
            for j in range(H):
		actionindex = get_next_action_index(cs, theta)
		
		force = actionlist[actionindex]
		actualnstate = doaction(force, cs)
		rew, tobreak = reward(actualnstate)

		nextstateidx = returnindexofstate(statelist, actualnstate)
#		ns = statelist[nextstateidx]
		ns = actualnstate
		tl.append((cs, actionindex, rew, ns))
		cs = ns
                if tobreak == True:
			break
	    lst.append(tl)	
	return lst


def main():
	
	E = 10
	H = 1000
	
	npp = 6 #number of policy parameters
	theta = np.random.random(npp)
	total = 100 # total number of iterations
	alpha = 0.1
	statelist =  buildstatelist()
	actionlist = buildactionlist()
	
	f = open('writelog_npg.txt', 'w')
	
        total = 100 # total number of iterations
        alpha = 0.2
        i = 1
        while (i <= total):
                newt = get_estimate(theta, E, H, statelist, actionlist)
#               print "new estimate became", newt
#               if converged(theta, theta + alpha * newt) == True:
#                       print "convergence!"
#                       print "print number of ieaerations", i
#                       break
                theta = theta + alpha * newt
		
		strin = [str(j) for j in theta]
		f.write(",".join(strin) +   "\n")
		i += 1

	f.close()
#	curtheta =  get_estimate(theta, E, H, statelist, actionlist)
	#for i in range(total):
	#	newest = get_estimate(theta, E, H, model, featvec)
	#	theta = theta + alpha * newest
	#	data = sampletraj(theta, 1, H, model, featvec)
	#	print "average reward is ", avgrew(data)
	
	#curtheta = [ 3.84964882, -3.07000536, -0.21214966 , 0.51792267]
	data = sampletraj(curtheta, 1, H, statelist, actionlist)
	print avgrew(data)[0]

if __name__ == "__main__":
	main()	
