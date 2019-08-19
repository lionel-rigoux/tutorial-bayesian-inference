function [q, logEvidence] = invert_variational_stochastic (data, prior)
% Bayesian logistic regression using stochasting gradient inference
% -------------------------------------------------------------------------
% This script implements a so called "blackbox" variational inference.
% This approach relies on the Free Energy approximation (aka ELBO). Rather 
% trying to direclty approximate the expected energy (eg with the Laplace
% approximation), we first derive the gradient of the ELBO wrt the variational 
% parameters (mu, sigma) and approximate this gradient via sampling.
%
% See Ranganath, Gerrish & Blei (2013) Black Box Variational Inference
% -------------------------------------------------------------------------

% Assuming that the posterior takes a Gaussian form: 
% q(\theta) = N(mu, Sigma)
% the inference reduces to an optimization problem: finding the moments 
% mu and Sigma which maximize the Free energy.

%% =========================================================================
% Free energy optimisation
% =========================================================================

% Meta parameters
% -------------------------------------------------------------------------
% number of random draws for the gradient estimation
batchSize = 1e3;

% convergence criteria
epsilon = 0.001;
maxIter = 1e3;

% learning rate scaling
eta = 1;

% Stochastic gradient ascent
% -------------------------------------------------------------------------
% initialisation
q = prior;

% follow gradient until convergence
for t = 1 : maxIter
    
    % draw a random batch of parameters from the variational distribution
    z = q.mu + sqrt(q.sigma) * randn(1,batchSize);

    % compute scores for all elements of the batch
    for i = 1 : batchSize
        h(:,i) = grad_score(q, z(i));
        f(:,i) = h(:,i) * (log_joint(data, prior, z(i)) - score(q, z(i)));
    end

    % approximate gradient as expectation 
    for d = 1 : 2 
        % control variate
        c = cov(h(d,:),f(d,:));
        a(d) = c(1,2)/var(h(d,:));
        
        % approximate gradient
        gL(d, t) = mean(f(d,:) - a(d)*h(d,:));
    end
    
    % adjust learning rate (adaGrad)
    rho = eta./sqrt(sum(gL.^2,2));

    % update variational moments
    delta = rho .* gL(:,t);
    q.mu = q.mu + delta(1);
    q.sigma = (sqrt(q.sigma) + delta(2))^2;

    % check for convergence
    if norm(delta) < epsilon
        break
    end
    
end

%% =========================================================================
% model evidence
% =========================================================================

% Using the Jensen's inequality, we can write the lower bound to the log 
% model evidence as:
%
% $$ log p(y) >= E_q[log p(y,\theta) - log q(theta)] $$
%
% We can then use the law of large number to approximate the expectation
% via sampling.

for i = 1 : 1e5
    theta = q.mu + sqrt(q.sigma) * randn();
    Li = log_joint(data, prior, theta) - score(q, theta);
end
logEvidence = mean(Li);

end

%% =========================================================================
% Subfunction
% =========================================================================

% scoring function: log q(\theta) = N(mu,sigma)
function s = score (q, z_i)
    s = - 0.5*log(2*pi*q.sigma) - 0.5 * ((z_i-q.mu).^2)/q.sigma;
end

% gradient of the scoring function
function g = grad_score(q, z_i)
    % d/d\theta
    g(1,1) = (z_i-q.mu)/q.sigma;
    % d/d\Sigma
    sq_sigma = sqrt(q.sigma);
    g(2,1) = ((z_i-q.mu)^2)/sq_sigma^3- 1/sq_sigma;
end
