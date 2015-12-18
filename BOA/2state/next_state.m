
function lst =  next_state(cs, action)
    if cs == 1
        if action == 1
            % lst has form reward, next state
            lst = [1, 1];
        else
            lst = [0, 2];
        end
    else
        if action == 1
            lst = [2, 2];
        else
            lst = [0, 1];
        end
    end
end