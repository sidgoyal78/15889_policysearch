function a = evaluate_and_estimate(theta, statelist, actionlist, E, T, flag)
    totalavgreward = 0.0;
    for i=1:E
        
        [p q] = Episode_for_boa(T, theta, statelist, actionlist, flag);       
        totalavgreward = totalavgreward + p;
        
    end
    a = totalavgreward / E;
end


