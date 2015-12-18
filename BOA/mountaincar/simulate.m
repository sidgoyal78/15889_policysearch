global listofeval listofthetas Kmatinv y_m kvect1 etamax

fvalue = [];

D = 6;
%bounds = ones(D, 2);
%bounds(:, 1) = 0;
%bounds(:, 2) = 1;

statelist = BuildStateList();
actionlist = BuildActionList();

E = 1;
T = 1000;
opts.LBounds = -100; opts.UBounds = 100;

theta = unifrnd(0,1,D,1)';
%theta = [0 1.0];
starter = unifrnd(0, 1, D, 1)';

listofthetas = [theta];
b  = evaluate_and_estimate(listofthetas(1,:)',statelist, actionlist, E, T, false);
listofeval = [b];

etamax = max(listofeval);
y_m = listofeval;

n = size(listofeval,1);
Knewmatold = ones(n);
Kmatinv = ones(n);

[xatmin, sfval] = cmaes('funtomax', theta, [], opts);

%Problem.f = 'funtomax';
%[fmin, xatmin, hist] = Direct(Problem, bounds);

newtheta = xatmin';
sfvallist = [sfval];
count = 50;

epsilon = 0.0001;
for i = 1:count
    theta = newtheta;
    fval = evaluate_and_estimate(theta', statelist, actionlist, E, T, false);
    listofeval = [listofeval; fval];
    listofthetas = [listofthetas;  theta]
    etamax = max(listofeval)
    iterationnumber = i
    y_m = listofeval;
    n = size(listofeval, 1);
    Knewmatnew = zeros(n);
    Knewmatnew(1:n-1, 1:n-1) = Knewmatold(1:n-1, 1:n-1);
 
    for p = 1:n
        Knewmatnew(p, n) = kernelf(listofthetas(p,:), listofthetas(n, :));
        Knewmatnew(n,p) = kernelf(listofthetas(n, :), listofthetas(p,:));
    end

    Kmatinv = inv(Knewmatnew + epsilon*eye(n));
    Knewmatold = Knewmatnew;
    
    [xatmin,sfval] = cmaes('funtomax', starter,[], opts);
   % Problem.f = 'funtomax';
   % [fmin, xatmin, hist] = Direct(Problem, bounds);
    newtheta = xatmin';
    sfvallist = [sfvallist; sfval];
end
[value, index] = max(listofeval);
%oyee = newtheta
%n = evaluate_and_estimate(newtheta',statelist, actionlist, 1, T, true)
noyee = listofthetas(index,:)
b  = evaluate_and_estimate(listofthetas(index, :)',statelist, actionlist, 1, T, true)




