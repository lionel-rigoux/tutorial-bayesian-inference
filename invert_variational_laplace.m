function [posterior, logEvidence] = variational_laplace (data, prior)
% Bayesian logistic regression using variational Laplace
% -------------------------------------------------------------------------
% This script implements a simple variational inference scheme to compute 
% the (Gaussian approximation of the) posterior distribution and the 
% (Free energy approximation of the) model evidence for our logisitic
% model. Using the Laplace approximation to the expected log joint, all
% values except the posterior mean can be derived analytically
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% 1a) the posterior mean is the MAP. This is usually solved using a
% regularized Gauss-Newton scheme. For the sake of simplicity, we prefer
% here the built-in Matlab optimization function.

posterior.mu = fminsearch(@(theta) - log_joint(data, prior, theta), 0);

% -------------------------------------------------------------------------
% 1b) The posterior variance has an analytical solution:
%
% $$ Sigma^* = - [d^2/d\theta^2 log p(y,\theta)]^1 $$
%

% Second order derivative of the log prior
d2dtheta2_logPrior = - 1 / prior.sigma;

% Second order derivative of the log likelihood. This is the most difficult
% term to derive, and can be approximated via automatic numeric
% differentiation if necessary. Here, we used known identities of the
% log-sigmoid derivatives to simplify the expression.
sx = sigmoid (data.x, posterior.mu);
d2dtheta2_logLikelihood = sum(- data.x.^2 .* sx .* (1-sx));
  
% Second order derivative of the log joint
posterior.sigma = - inv (d2dtheta2_logPrior + d2dtheta2_logLikelihood);
               
% -------------------------------------------------------------------------
% 2) The (log) model evidence can be approximated by the Free energy, which
% it itself approximated via the Laplace approximation at the optimal q.

logEvidence = free_energy_laplace (data, prior, posterior);
               
end

%% =========================================================================
% Free energy:
% $$ F = E[\log p(y,\theta)]_q + H[q]
% =========================================================================
function F = free_energy_laplace (data, prior, q)
  F = ...
      Eq_log_joint_laplace (data, prior, q) ...
    + entropy (q);
end

%% =========================================================================
% Evaluates the expectation of the log joint distribution, ie:
%
% $$ E[\log p(y, \theta)]_q = \int \log p(y,\theta) q(\theta) d\theta $$
%
% using the Laplace approximation:
%
% $$ E[\log p(y, \theta)]_q \approx 
%               \log p(y, \theta^*) 
%             + 1/2 tr[Sigma d^2/d\theta^2 \log p(y, \theta)]
% $$
%
% Here, we used the fact that:
%
% $$ Sigma^* = - [d^2/d\theta^2 \log p(y, \theta)]^-1 $$
%
% to simplify the second term of the appoximation.
% =========================================================================
function elj = Eq_log_joint_laplace (data, prior, q)
   elj = log_joint (data, prior, q.mu) - 0.5;
end




         
    