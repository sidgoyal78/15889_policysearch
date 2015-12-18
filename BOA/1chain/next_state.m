
function lst =  next_state(cs, action)

    transition = [2, 1; 3, 1; 4, 1; 5, 1; 5, 1];
    rewards = [0.0, 2.0; 0.0, 2.0; 0.0, 2.0; 0.0, 2.0; 10.0, 2.0];
    lst = [rewards(cs, action), transition(cs, action)]; 

end
