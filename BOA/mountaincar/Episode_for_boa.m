function [ avgreward,steps ] = Episode_for_boa( maxsteps, theta, statelist,actionlist,grafic )
%MountainCarEpisode do one episode of the mountain car with sarsa learning
% maxstepts: the maximum number of steps per episode
% Q: the current QTable
% alpha: the current learning rate
% gamma: the current discount factor

% statelist: the list of states
% actionlist: the list of actions

% Mountain Car Problem 
% 
% Modified the code of: Jose Antonio Martin H. <jamartinh@fdi.ucm.es>
% 



initial_position = -0.5;
initial_speed    =  0.0;

x            = [initial_position,initial_speed];
steps        = 0;
total_reward = 0;


% convert the continous state variables to an index of the statelist
s   = DiscretizeState(x,statelist);
% selects an action based on the softmax policy
a   = choose_next_action_mountaincar(theta, x);

for i=1:maxsteps    
        
    % convert the index of the action into an action value
    action = actionlist(a);
    
    %do the selected action and get the next car state    
    xp  = DoAction( action , x );    
    
    % observe the reward at state xp and the final state flag
    [r,f]   = GetReward(xp);
    total_reward = total_reward + r;
    
    % convert the continous state variables in [xp] to an index of the statelist    
    sp  = DiscretizeState(xp,statelist);
    
    % select action prime
    %ap = e_greedy_selection(Q,sp,epsilon);
    ap   = choose_next_action_mountaincar(theta, xp);
    
    
    % Update the Qtable, that is,  learn from the experience
    %Q = UpdateSARSA( s, a, r, sp, ap, Q , alpha, gamma );
    
    
    %update the current variables
    s = sp;
    a = ap;
    x = xp;
    %x  = statelist(sp,:);
    
    %increment the step counter.
    steps=steps+1;
    
   
    % Plot of the mountain car problem
    if (grafic==true)      
        
       MountainCarPlot(x,action,steps);    
       %MountainCarPlotSingle(x,action,steps);    
    end
    
    % if the car reachs the goal breaks the episode
    if (f==true)
        break
    end
   
end

avgreward = total_reward;

