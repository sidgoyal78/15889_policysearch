function a = choose_next_action(theta, cur)

    features = eye(10);

    exponent = features([cur * 2 - 1, cur * 2], :) * theta;
    expmax = max(exponent); expb = exp(exponent - expmax); 
    prob = expb ./ sum(expb); 
    a = find(mnrnd(1, prob)); 

end
