global listofeval listofthetas Kmatinv y_m kvect1 etamax


D = 10;
%bounds = ones(D, 2);
%bounds(:, 1) = 0;
%bounds(:, 2) = 1;

E = 10;
T = 50;
opts.LBounds = -10; opts.UBounds = 10;

theta = unifrnd(0,1,D,1)';
%theta = [0 1.0];

listofthetas = [theta];
b  = evaluate_and_estimate(listofthetas(1,:)',E, T);
listofeval = [b];

etamax = max(listofeval)
y_m = listofeval;

n = size(listofeval,1);
Knewmatold = ones(n);
Kmatinv = ones(n);

[xatmin, sfval] = cmaes('funtomax', theta, [], opts)

%Problem.f = 'funtomax';
%[fmin, xatmin, hist] = Direct(Problem, bounds);

newtheta = xatmin';
sfvallist = [sfval];
count = 50;

epsilon = 0.0001;
for i = 1:count
    theta = newtheta;
    fval = evaluate_and_estimate(theta', E, T);
    listofeval = [listofeval; fval];
    listofthetas = [listofthetas;  theta]
    etamax = max(listofeval)
    
    y_m = listofeval;
    n = size(listofeval, 1);
    Knewmatnew = zeros(n);
    Knewmatnew(1:n-1, 1:n-1) = Knewmatold(1:n-1, 1:n-1);
   % for p = 1: n-1s
   %     for q = 1:n-1
   %         Knewmatnew(p,q) = Knewmatold(p,q);
   %     end
   % end
    
    for p = 1:n
        Knewmatnew(p, n) = kernelf(listofthetas(p,:), listofthetas(n, :));
        Knewmatnew(n,p) = kernelf(listofthetas(n, :), listofthetas(p,:));
    end
   % abc = Knewmatnew
    
    Kmatinv = inv(Knewmatnew + epsilon*eye(n));
    Knewmatold = Knewmatnew;
    
    [xatmin,sfval] = cmaes('funtomax', 0.5 * ones(10, 1),[], opts)
   % Problem.f = 'funtomax';
   % [fmin, xatmin, hist] = Direct(Problem, bounds);
    newtheta = xatmin'
    sfvallist = [sfvallist; sfval];
end
b  = evaluate_and_estimate(newtheta',E, T)




