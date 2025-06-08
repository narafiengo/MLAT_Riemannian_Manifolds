function [x, cost,info,options, current_point] = LevenbergMarquardt(problem, x, options)
localdefaults.verbosity = 2;
localdefaults.maxtime = inf;
localdefaults.miniter = 0;
localdefaults.maxiter = 1000;
localdefaults.eta = 0.2;
localdefaults.beta = 1.5;
localdefaults.mu_min = 0.1;
localdefaults.flagnz = true;
localdefaults.tolgradnorm = 1e-8;
current_point = cell(1,1);
current_point{1} = x;
localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
M = problem.M;
if ~exist('options', 'var') || isempty(options)
    options = struct();
end
options = mergeOptions(localdefaults, options);

%if no initial point x is given by the user, generate one at random.
if ~exist('x', 'var') || isempty(x)
    x = M.rand();
end
M = problem.M;
cost = problem.cost;
F = problem.F;
J = problem.J; %must be riemannian jacobian, gradient must be riemannian
hess = problem.hess; %need to porovide the riemannian GN hessian approx
mu_bar = options.mu_min;
mu = mu_bar;
k = 0;
info = struct('iter', [], 'cost', [], 'gradnorm', [], 'mu', [], 'lambda', [], 'rho', [], 'stepsizenorm', []);
fprintf('%5s   %20s  %15s  %15s %15s\n', 'iter', 'cost', 'grad. norm', 'rho', 'lambda');
while k < options.maxiter
    Fx = F(x);
    fx = cost(x);
    normFx = norm(Fx); %euclidean norm
    lambda = mu*normFx^2; %damping parameter
    if isfield(problem, 'grad')
        grad_fx = problem.grad;
        grad_fx = grad_fx(x);
    else
        eGrad = problem.egrad;
        grad_fx = M.egrad2rgrad(x, eGrad(x));
    end
    HxLM = @(s) hess(x,s) + lambda*s;
    s = lm_tCG_solver(HxLM, -grad_fx, M, x, options.tolgradnorm, 100);
    theta_0 = normFx^2;
    inner_grad_s = M.inner(x,grad_fx,s);
    model_diff = (-2*inner_grad_s - norm(J(x,s))^2 - lambda*M.norm(x,s)^2);
    x_new = M.retr(x,s);
    Fx_new = F(x_new);
    theta_new = norm(Fx_new)^2;
    rho = (theta_0-theta_new) / model_diff;


    if rho >= options.eta
        x = x_new;
        mu_bar = mu;
        if options.flagnz
            mu = mu_bar;
        else
            mu  = max(options.mu_min, mu_bar/options.beta);

        end
    else
        mu = mu*options.beta;
    end
info.rho(end+1) = rho;
info.iter(end+1) = k;
info.cost(end+1) = fx;
info.gradnorm(end+1) = M.norm(x,grad_fx);
info.mu(end+1) =mu;
info.lambda(k+1) =lambda;
info.stepsizenorm(k+1) = M.norm(x, s);
current_point{end+1} = x;
fprintf('%5d   %+.16e   %12e %12e %12e\n',k, fx, info.gradnorm(k+1), rho, lambda);

if info.gradnorm(end) < options.tolgradnorm
    break;
end

k = k+1;
end
end