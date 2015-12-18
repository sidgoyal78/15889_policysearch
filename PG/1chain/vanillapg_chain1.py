import numpy as np
import sys


def returngrad(theta, state, action, featvec):

  c1 = np.dot(theta, featvec[(1, 1)]); 
  c2 = np.dot(theta, featvec[(1, 2)]); 
  c3 = np.dot(theta, featvec[(2, 1)]); 
  c4 = np.dot(theta, featvec[(2, 2)]); 
  c5 = np.dot(theta, featvec[(3, 1)]); 
  c6 = np.dot(theta, featvec[(3, 2)]); 
  c7 = np.dot(theta, featvec[(4, 1)]); 
  c8 = np.dot(theta, featvec[(4, 2)]); 
  c9 = np.dot(theta, featvec[(5, 1)]); 
  c10 = np.dot(theta, featvec[(5, 2)]); 

  t = np.amax([c1, c2]); c1 -= t; c2 -= t;
  t = np.amax([c3, c4]); c3 -= t; c4 -= t;
  t = np.amax([c5, c6]); c5 -= t; c6 -= t;
  t = np.amax([c7, c8]); c7 -= t; c8 -= t;
  t = np.amax([c9, c10]); c9 -= t; c10 -= t;

  c1 = np.exp(c1); 
  c2 = np.exp(c2); 
  c3 = np.exp(c3); 
  c4 = np.exp(c4); 
  c5 = np.exp(c5); 
  c6 = np.exp(c6); 
  c7 = np.exp(c7); 
  c8 = np.exp(c8); 
  c9 = np.exp(c9); 
  c10 = np.exp(c10); 
  
  if state == 1 and action == 1:
    return featvec[(1,1)] - ( (c1*featvec[(1,1)] + c2*featvec[(1,2)])/(c1 + c2))
  elif state == 1 and action == 2:
    return featvec[(1,2)]  - ( (c1*featvec[(1,1)] + c2 * featvec[(1,2)])/(c1 + c2))

  elif state == 2 and action == 1:
    return featvec[(2,1)] - ( (c3*featvec[(2,1)] + c4 * featvec[(2,2)])/(c3 + c4))
  elif state == 2 and action == 2:
    return featvec[(2,2)] - ( (c3*featvec[(2,1)] + c4 * featvec[(2,2)])/(c3 + c4))

  elif state == 3 and action == 1:
    return featvec[(state,action)] - ( (c5*featvec[(state,action)] + c6 * featvec[(state, action + 1)])/(c5 + c6))
  elif state == 3 and action == 2:
    return featvec[(state,action)] - ( (c5*featvec[(state,action - 1)] + c6 * featvec[(state, action)])/(c5 + c6))
  
  elif state == 4 and action == 1:
    return featvec[(state,action)] - ( (c7*featvec[(state,action)] + c8 * featvec[(state, action + 1)])/(c7 + c8))
  elif state == 4 and action == 2:
    return featvec[(state,action)] - ( (c7*featvec[(state,action - 1)] + c8 * featvec[(state, action)])/(c7 + c8))
  
  elif state == 5 and action == 1:
    return featvec[(state,action)] - ( (c9*featvec[(state,action)] + c10 * featvec[(state, action + 1)])/(c9 + c10))
  elif state == 5 and action == 2:
    return featvec[(state,action)] - ( (c9*featvec[(state,action - 1)] + c10 * featvec[(state, action)])/(c9 + c10))

def converged(theta1, theta2):
  eps = 0.0005
# print abs(theta1 - theta2)
# print theta1, theta2
# print
  return (abs(theta1 - theta2) < eps).all()

def get_estimate(theta, E, H, model, featvec):
  global finfo
  noi= 1
  cc = 0
  flag = 0
  newest = np.zeros(len(theta))
  prev = np.zeros(len(theta))
  curtheta = np.copy(theta)
  data = []
  ans = None
  while(cc < noi):
    data =  sampletraj(curtheta , E, H, model, featvec)
    rews = avgrew(data)
    ## we need 
    ##    policy derivatives psi_k_avg  <psi_k>
    ##    fisher matrix Ftheta = avg of (sum of psi_k) * (sum of psi_k)T
    ##    vanilla gradient  = avg of {(sum of psi_k) * (sum of rew)}
    phi_ks = []
    rew_ks = []
  # print rews
    for i in range(len(data)):
      temp_phi = np.zeros(len(theta))
      rew_ks.append(rews[i])
      for j in range(len(data[i])):
        temp_phi += returngrad(curtheta, data[i][j][0], data[i][j][1], featvec) 
      phi_ks.append(temp_phi)
      
    phi_ks = np.array(phi_ks)
    squaredthing = phi_ks * phi_ks
      
    numer = np.zeros(len(theta))
    for i in range(len(data)):
      numer += squaredthing[i,:]*rew_ks[i]
    
    numer /= len(data)
    denom = np.mean(squaredthing, axis = 0)
    B = numer / (denom + finfo.eps)

    gest = np.zeros(len(theta)) 
    for i in range(len(theta)):
      for j in range(len(data)):
        gest[i] += phi_ks[j][i] * (rew_ks[j] - B[i])
  
    ans =  gest/len(data)
    print np.mean(np.array(rews))
    return ans

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

  val1 = np.dot(theta, featvec[(cs, 1)]);
  val2 = np.dot(theta, featvec[(cs, 2)]); 
  t = np.amax([val1, val2]);
  val1 -= t; val2 -= t;
  val1 = np.exp(val1);
  val2 = np.exp(val2); 

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
    cs = np.random.randint(1,6)
    for j in range(H):
      action = get_next_action(cs, theta, featvec)
      rew, ns = model[(cs, action)]
      tl.append((cs, action, rew, ns))
      cs = ns
    lst.append(tl)  
  return lst


def main():

  global finfo  
  finfo = np.finfo('float');
  npp = 10 #number of policy parameters
  theta = np.random.random(npp)
  #theta = np.array([0.5, 0.5, 0.5, 0.5])

  featvec = {(1, 2): np.array([ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]), (3, 2): np.array([ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.]), (5, 2): np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]), (3, 1): np.array([ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.]), (2, 1): np.array([ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]), (2, 2): np.array([ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]), (5, 1): np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.]), (4, 2): np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.]), (4, 1): np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.]), (1, 1): np.array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])}

  model = {(1,1): (0.0, 2), (1, 2):(2.0, 1), \
                (2,1): (0.0, 3), (2, 2):(2.0, 1), \
                (3,1): (0.0, 4), (3, 2):(2.0, 1), \
                (4,1): (0.0, 5), (4, 2):(2.0, 1), \
                (5,1): (10.0, 5), (5, 2): (2.0, 1)}

  E = 10 #number of episodes
  H = 50 # horizon length
  
  total = 200 # total number of iterations
  alpha = 0.10
  i = 1
  while (i <= total):
    newt = get_estimate(theta, E, H, model, featvec)
#   print "new estimate became", newt
#   if converged(theta, theta + alpha * newt) == True:
#     print "convergence!"
#     print "print number of ieaerations", i
#     break
    theta = theta + alpha * newt

    i += 1
  #for i in range(total):
  # newest = get_estimate(theta, E, H, model, featvec)
  # theta = theta + alpha * newest
  # data = sampletraj(theta, 1, H, model, featvec)
  # print "average reward is ", avgrew(data)
  
  #curtheta = [ 3.84964882, -3.07000536, -0.21214966 , 0.51792267]
  #data = sampletraj(theta, 1, H, model, featvec)
  #print avgrew(data)[0]
if __name__ == "__main__":
  main()  
