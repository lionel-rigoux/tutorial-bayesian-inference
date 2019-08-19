function [posterior, logEvidence] = invert_variational_numeric(data, prior)
% Bayesian logistic regression using variational inference (with sampling)
% -------------------------------------------------------------------------
% This script implements a simple variational inference scheme to compute 
% the (Gaussian approximation of the) posterior distribution and the 
% (Free energy approximation of the) model evidence for our logisitic
% model. In this case, we do ot use the Laplace approximation and compute
% the expected log-joint via sampling.
% No one is using this in practice, this code is only intended for
% educative purpose!
% -------------------------------------------------------------------------

% Assuming that the posterior takes a Gaussian form: 
% q(\theta) = N(mu, Sigma)
% the inference reduces to an optimization problem: finding the moments 
% mu and Sigma which maximize the Free energy.

% Starting point of the search. For the sake of simplicity, we quickstart
% the algorithm with a good guess.

% Maximize the Free Energy
[muSigma, mF] = fminsearch ( ...
    @(x) - free_energy(data, prior, struct('mu', x(1), 'sigma', x(2)^2)), ...
    [2; 1], ... starting point of the search
    struct('TolFun', 1e-2, 'TolX', 1e-2) ... no need to be more precise than the sampling
  );

% Wrapping up
posterior.mu = muSigma(1);
posterior.sigma = muSigma(2)^2;
logEvidence = - mF;

end

%% =========================================================================
% Free energy:
% $$ F = E[\log p(y,\theta)]_q + H[q]
% =========================================================================
function F = free_energy (data, prior, q)
  F = ...
      Eq_log_joint (data, prior, q) ...
    + entropy (q);
end

%% =========================================================================
% Computes the expectation of the log joint distribution, ie:
%
% $$ E[\log p(y, \theta)]_q = \int \log p(y,\theta) q(\theta) d\theta $$
%
% This is done via a sampling approach as follows:
%   - for t = 1 : N   
%       - sample \theta_t from q(\theta)
%       - compute \log p(y, \theta_t)
% According to the low of large numbers, the mean of the \log p(y, \theta_t)
% will converge, with increasing N, to the value of the expectation.
% =========================================================================
function elj = Eq_log_joint(data, prior, q)
    % initialisation
    % ---------------------------------------------------------------------
    % number of samples
    N = 1e4;
    % memory pre-allocation
    lj = nan(1, N);
    
    % Sampling procedure
    % ---------------------------------------------------------------------
    parfor t = 1 : N
        % draw theta from q(\theta) = N(mu, Sigma)
        theta = q.mu + sqrt(q.sigma) * randn();
        % compute the corresponding log joint
        lj(t) =  log_joint (data, prior, theta);
    end
    
    % Apply the law of large numbers
    % ---------------------------------------------------------------------
    elj = mean (lj);
end


