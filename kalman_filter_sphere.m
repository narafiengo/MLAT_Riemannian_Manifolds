%%   In this script the iterative refinement procedure is tested on 
%%   of selected path trajectory on the unit sphere
%%   The script is structured as follows: 
%%   In part 1: we define the necessary functions to apply optimization algorithms on the sphere
%%   In part 2: we generate the exact path in order to generate our measurements
%%   In part 3: we generate the noisy measurements and apply GN and LM to generate the raw estimates
%%   In part 4: the iterative refinement scheme gn_kalman_smoother is implemented and applied to refine path estimates
%%              where 1 is used to indicate where we are working with GN and 0 for LM
%%   In part 5: the MSE and gain are measured


%% PART 1:    

%Euclidean gradient
function gradient = gradientSphere(x, p, r, manifold)
K = length(p);
gradient = 0;
for i = 1: K
    pval = p{i};
    chordalDist = norm(x(:)-pval(:), 'fro');
    gradientChordal = 1/chordalDist .* (x(:) - pval(:));
    gradientDist = 1/sqrt(1-(chordalDist/2)^2)*gradientChordal;
    gradient = gradient + (manifold.dist(x,pval)-r{i})*gradientDist; 
end
gradient = gradient./K;
end 

%Euclidean GN approximation
function hessian = hessian_GN(x,p,manifold, v)
K = length(p);
hessian = 0;
for i = 1 : K
    pval = p{i};
    dist = manifold.dist(x, pval);
    chordalDist = norm(x(:)-pval(:), 'fro');
    A = sqrt(1-(chordalDist/2)^2);
    gradientDist = 1/(A*chordalDist)*(x(:)-pval(:))./(sqrt(K)); 
    gradient_prd = manifold.inner(1,gradientDist, v);
    hessian = hessian + gradient_prd*gradientDist;
end
end 

%Riemannian GN approximation (needed as LM implementation on works with Riemannian version)
function hessian = hessian_GNRiemannian(x,p,manifold, v)
K = length(p);
hessian = 0;
for i = 1 : K
    pval = p{i};
    dist = manifold.dist(x, pval);
    chordalDist = norm(x(:)-pval(:), 'fro');
    A = sqrt(1-(chordalDist/2)^2);
    gradientDist = 1/(A*chordalDist)*(x(:)-pval(:))./(sqrt(K)); 
    %convert euclidean gradient to riemannian
    gradientDist = manifold.egrad2rgrad(x, gradientDist);
    gradient_prd = manifold.inner(x,gradientDist, v);
    hessian = hessian + gradient_prd*gradientDist;
end
end 

function Jx = Jfunction(x, pvalues, rvalues, manifold, v)
K = length(pvalues);
for i = 1 : K
    pval = pvalues{i};
    dist = manifold.dist(x, pval);
    chordalDist = norm(x(:)-pval(:), 'fro');
    A = sqrt(1-(chordalDist/2)^2);
    gradientDist = 1/(A*chordalDist)*(x(:)-pval(:))./(sqrt(K)); 
    gradientDistR = manifold.egrad2rgrad(x, gradientDist);
    gradient_prd = manifold.inner(x,gradientDistR, v);
    Jx(i) = gradient_prd;
end
end

function B = tangent_basis_sphere(x)
    if abs(x(1))<0.9
        v = [1; 0; 0];
    else
        v = [0; 1; 0];
    end
    b1 = v-dot(v, x)*x;
    b1 = b1/norm(b1);
    b2 = cross(x, b1);
    b2 = b2/norm(b2);
    B{1} = b1;
    B{2} =b2;
end

function Fx = Ffunction(x,pvalues, rvalues, manifold)
K = length(pvalues);
Fx = [];
for i = 1 : K
    Fx = [Fx; manifold.dist(x, pvalues{i}) - rvalues{i}];
end
Fx = (1/sqrt(K)).*Fx;
end

%% PART 2: 
n = 3;
manifold = spherefactory(n);
N = 500;
t_vals = linspace(0, pi/2, N);
traj = zeros(3, N);
x0 = manifold.rand();
v = manifold.randvec(x0);
for i = 1: N
    traj2(:,i) = manifold.exp(x0 , v*t_vals(i));
end
plot3(traj2(1,:),traj2(2,:), traj2(3,:), 'b')

% the trajectory we used for the results (which was generated using previous commands)
traj2 = load('trajectory2.mat').traj2;
traj = load('trajectory.mat').traj;
plot3(traj2(1,:),traj2(2,:), traj2(3,:), 'b')
xlim([-1,1])
ylim([-1,1])
zlim([-1,1])
%%

pvalues = cell(1, 3);
midTrajectory = traj2(:,250);
startTrajectory = traj(:,15);
premidTrajectory = traj(:,25);
% random vectors to generate anchors
%v1 = manifold.randvec(midTrajectory);
%v2 = manifold.randvec(midTrajectory);

%or choose independent vectors
B = tangent_basis_sphere(midTrajectory);
v1 = B{1};
v2 = B{2};
v1 = v1./manifold.norm(midTrajectory,v1);
v2 = v2./manifold.norm(midTrajectory,v2);

v3 = manifold.randvec(midTrajectory);
v3 = v3./manifold.norm(midTrajectory,v3);
v4 = manifold.randvec(midTrajectory);
v4 = v4./manifold.norm(midTrajectory,v4);
%% PART 3:
% generate anchors
pvalues{1} = manifold.exp(midTrajectory, 0.5*v1);
pvalues{2} = manifold.exp(midTrajectory, -1.5*v2);
pvalues{3} = manifold.exp(midTrajectory, 1.5*v3);
pvalues{4} = manifold.exp(midTrajectory, 1*v4);

rvalues = cell(1, 3);
GNtrajectorynoiseFilter = [];
errorFilter = [];
errorNoFilter = [];
GNtrajectorynoise = [];

problemGN.M = manifold;
problemLM.M = manifold;
options.tolgradnorm = 10^-8;
%x0 = manifold.rand();
x0 = [-0.589819; -0.805304; 0.0599929];
GNtrajectorynoise(:,1) = x0;
x0GNpure = x0;
x0LMpure = x0;

N = 500;
for i = 2: N+1
    eps = randn();
    rvalues{1} = manifold.dist(pvalues{1}, traj2(:,i-1));
    rvalues{2} = manifold.dist(pvalues{2}, traj2(:,i-1));
    rvalues{3} = manifold.dist(pvalues{3}, traj2(:,i-1));
    rvalues{4} = manifold.dist(pvalues{4}, traj2(:,i-1));
    rvaluesnoise{1} = rvalues{1} + (0.05)*eps;
    rvaluesnoise{2} = rvalues{2}+ (0.05)*eps;
    rvaluesnoise{3} = rvalues{3}+(0.05)*eps;
    rvaluesnoise{4} = rvalues{4}+(0.05)*eps;
    measurements(i-1,1) =  rvaluesnoise{1};
    measurements(i-1,2) =  rvaluesnoise{2};
    measurements(i-1,3) =  rvaluesnoise{3};
    measurements(i-1,4) =  rvaluesnoise{4};

    %compute the raw estimates for GN and LM 
    problemGN.cost = @(x) objectiveFunction(x, pvalues, rvaluesnoise,manifold);
    options.statsfun = statsfunhelper('current_point', @(x) x);
    problemGN.egrad = @(x) gradientSphere(x, pvalues, rvaluesnoise, manifold);
    problemGN.ehess = @(x, v) hessian_GN(x, pvalues, manifold, v);
    [solGNRnnoisePure, cost, info, options] = trustregions(problemGN, x0GNpure, options);
    GNtrajectorynoisePure(:,i-1) = solGNRnnoisePure;
    x0GNpure = solGNRnnoisePure; 
    numberIterRawGN(i-1) = length([info.iter]);
    
    problemLM.cost = @(x) objectiveFunction(x, pvalues, rvaluesnoise,manifold);
    problemLM.egrad = @(x) gradientSphere(x, pvalues, rvaluesnoise, manifold);
    problemLM.F = @(x) Ffunction(x, pvalues, rvaluesnoise, manifold);
    problemLM.J = @(x, v) Jfunction(x,pvalues, rvaluesnoise, manifold, v);
    problemLM.hess = @(x, v) hessian_GNRiemannian(x,pvalues,manifold, v);
    [solLM, cost, infoLM, options, current_points] = LevenbergMarquardt(problemLM,x0LMpure);
    numberIterRawLM(i-1) = length([infoLM.iter]);
    LMtrajectorynoisePure(:,i-1) = solLM;
    x0LMpure = solLM; 
end
%% PART 4:
% this is important otherwise it uses the last x0 of the gn and LM
x0GNpure = x0;
x0LMpure = x0;

% GN estimates (1 to indicate GN method used)
B_ran = tangent_basis_sphere(x0GNpure);
B_ran = [B_ran{1}, B_ran{2}];    
[x_filtered, P, numberIterGN] = gn_kalman_smoother(measurements, pvalues, manifold, x0GNpure, 1, B_ran);

% LM estimates (0 to indicated LM method used)
B_ran = tangent_basis_sphere(x0GNpure);
B_ran = [B_ran{1}, B_ran{2}];
[x_filteredLM, P, numberIterLM] = gn_kalman_smoother(measurements, pvalues, manifold, x0LMpure, 0, B_ran);

%% PART 5:

for i = 1:500
    errorValNoise(i) = manifold.dist(traj2(:,i),GNtrajectorynoisePure(:,i));
    errorValFilt(i) = manifold.dist(traj2(:,i), x_filtered(:,i+1));
end

MSENoise = mean(errorValNoise.^2);
MSEFilt = mean(errorValFilt.^2);
gain = MSENoise/MSEFilt;

%%
function [x_filtered, P, numberIter] = gn_kalman_smoother(measurements, anchors, manifold, x0, gnLM,B_ran)
    %measurements= rvalues for N = 100: N x K matrix (K anchors, N time steps)
    %anchors: Kx3 cell array of anchor positions
    
    N = size(measurements,1);
    x_filtered = zeros(3,N);
    P = zeros(2,2,N); %covariance in tangent space
    
    x_filtered(:,1) = x0;
    P(:,:,1) =(1.7)^2.*eye(2);
    
    for k = 2:N+1
        rvalues{1} = measurements(k-1,1); 
        rvalues{2} = measurements(k-1,2);
        rvalues{3} = measurements(k-1,3);
        rvalues{4} = measurements(k-1,4);
      
        if gnLM == 1
            problemGN.M = manifold;
            problemGN.cost = @(x) objectiveFunction(x, anchors, rvalues,manifold);
            options.statsfun = statsfunhelper('current_point', @(x) x);
            problemGN.egrad = @(x) gradientSphere(x, anchors, rvalues, manifold);
            problemGN.ehess = @(x, v) hessian_GN(x, anchors, manifold, v);
            [x_gn, cost, info, options] = trustregions(problemGN, x_filtered(:,k-1));
        else
            problemLM.M = manifold;    
            problemLM.cost = @(x) objectiveFunction(x, anchors, rvalues,manifold);
            problemLM.egrad = @(x) gradientSphere(x, anchors, rvalues, manifold);
            problemLM.F = @(x) Ffunction(x, anchors, rvalues, manifold);
            problemLM.J = @(x, v) Jfunction(x,anchors, rvalues, manifold, v);
            problemLM.hess = @(x, v) hessian_GNRiemannian(x,anchors,manifold, v);
            [x_gn, cost, info, options, current_points] = LevenbergMarquardt(problemLM,x_filtered(:,k-1));
        end 
        
        numberIter(k-1) = length([info.iter]);
        
        %random walk to generate estimate
        xi = randn(2,1);
        velocityk = B_ran*xi;
        xprior = manifold.exp(x_filtered(:,k-1),0.01.*velocityk);
        [R_gn, B] = compute_gn_covariance(xprior, anchors, manifold);
        Q = 0.01^2 * eye(2);
        P(:,:,k-1) = P(:,:,k-1)+ Q;
        %update step
        [x_filtered(:,k), P(:,:,k)] = update_step(xprior, P(:,:,k-1), x_gn, R_gn, manifold, B);
        %compute basis at the new filtered estimate (needed to compute vector for random walk)
        B_ran = tangent_basis_sphere(x_filtered(:,k));
        B_ran = [B_ran{1}, B_ran{2}];
    end
end

function [x_new, P_new] = update_step(x, P, z, R, manifold, B)
    %x: current state (3x1)
    %P: current covariance (2x2)
    %z: GN measurement (3x1)
    %R: GN covariance (2x2 in tangent space)
    
    %innovation step
    y = manifold.log(x, z);
    y_tangent = [B{1}'*y; B{2}'*y];
    
    H = eye(2);
    %update and kalman gain
    S = H*P*H' + R;
    K = P*H'/S;
    
    %tangent space
    dx_tangent = K*y_tangent;
    
    %map back to manifold using tangent basis
    dx = B{1}*dx_tangent(1) + B{2}*dx_tangent(2);
    %this was added by me the 1.5
    x_new = manifold.exp(x, dx);
    %joseph formula to update
    I = eye(2);
    P_new = (I-K*H)*P*(I-K*H)' + K*R*K';
end

function [R_gn, B] = compute_gn_covariance(x, anchors, manifold)
    J = zeros(length(anchors),2);
    B = tangent_basis_sphere(x);
    
    for i = 1:length(anchors)
        d = manifold.dist(x, anchors{i});
        grad = -manifold.log(x, anchors{i})./d;
        J(i,:) = [B{1}'*grad, B{2}'*grad];
    end
    sigma2 = 0.05; %measurement noise --> 0.05
    R_gn = sigma2^2 * inv(J'*J);
end


