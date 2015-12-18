function val = funtomax(theta)
    global etamax listofeval
  %  kyun = theta
   % kyahai = etamax
    new_mu = get_next_mu(theta);
    new_sig = get_next_sigma(theta)^0.5;
    if new_sig == 0.0
        val = 0.0;
        return
    end
    temp = (new_mu - etamax)/(new_sig);
    val_EI = (new_mu - etamax )* normcdf(temp) + new_sig * normpdf(temp);  
    
    val_PI  = new_sig/(etamax - new_mu);
    
    val_PIalt = (new_mu - etamax) / new_sig;
    
    dim = size(theta, 1);
    kucb = sqrt(1 * 2 * log(size(listofeval, 1)^ (2 + dim/2)  * 2* (pi^2) / (3*0.05)));
    val_UCB = new_mu + kucb * new_sig;
    val = -1 * val_EI;
end

function v = get_next_mu(theta)
    global Kmatinv  y_m kvect1
    kvect1 = getKvector(theta, 1);
    
    v =  +(kvect1*Kmatinv*y_m);
end

function v = get_next_sigma(theta)
    global kvect1 Kmatinv
    %kvect2 = getKvector(theta, 2);
    
    v = 1.0 - kvect1*Kmatinv*kvect1';
end

function v = getKvector(theta, flag)
    global listofthetas
    a= [];
    n = size(listofthetas,1);
    for i = 1:n
        if flag == 1
            a = [a; kernelf(theta', listofthetas(i,:))];
        else
            a = [a; kernelf(listofthetas(i,:), theta')];
        end
    end
    v = a';
end

    