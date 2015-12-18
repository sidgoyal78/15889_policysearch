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
  eps = 0.005
# print abs(theta1 - theta2)
# print theta1, theta2
# print
  return (abs(theta1 - theta2) < eps).all()

def get_estimate(theta, E, H, model, featvec):
  curtheta = np.copy(theta)
  alpha = 0.05
  noi= 200
  cc = 0
  while(cc < noi):
    data = sampletraj(curtheta, E, H, model, featvec)
    
    newtheta = np.copy(curtheta)
    rews = avgrew(data)
    reqdrew = np.mean(np.array(rews))
    print reqdrew
    for i in range(len(data)):
      for j in range(len(data[i])):
        newtheta = newtheta + alpha * returngrad(curtheta, data[i][j][0], data[i][j][1], featvec)*rews[i]
    #if converged(curtheta, newtheta) == True:
    # print "convergence!!!"
    # return curtheta
    #else:
    # curtheta = np.copy(newtheta)
    curtheta = np.copy(newtheta)

    cc += 1
  return curtheta

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
  
  val1 = np.exp(np.dot(theta, featvec[(cs,1)]))
  val2 = np.exp(np.dot(theta, featvec[(cs,2)]))
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
  
  
  npp = 10 #number of policy parameters
  theta = np.random.random(npp)
  
  featvec = {(1, 2): np.array([ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]), (3, 2): np.array([ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.]), (5, 2): np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]), (3, 1): np.array([ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.]), (2, 1): np.array([ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]), (2, 2): np.array([ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]), (5, 1): np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.]), (4, 2): np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.]), (4, 1): np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.]), (1, 1): np.array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])}

  model = {(1,1): (0.0, 2), (1, 2):(2.0, 1), \
                (2,1): (0.0, 3), (2, 2):(2.0, 1), \
                (3,1): (0.0, 4), (3, 2):(2.0, 1), \
                (4,1): (0.0, 5), (4, 2):(2.0, 1), \
                (5,1): (10.0, 5), (5, 2): (2.0, 1)}

  E = 10 #number of episodes
  H = 50 # horizon length
  
  total = 100 # total number of iterations
  get_estimate(theta, E, H, model, featvec)
  #for i in range(total):
  # newest = get_estimate(theta, E, H, model, featvec)
  # theta = theta + alpha * newest
  # data = sampletraj(theta, 1, H, model, featvec)
  # print "average reward is ", avgrew(data)
  
  #curtheta = [ 3.84964882, -3.07000536, -0.21214966 , 0.51792267]
  #data = sampletraj(curtheta, 1, H, model, featvec)
  #print avgrew(data)[0]
if __name__ == "__main__":
  main()  
