function [posterior, logEvidence, theta]= monte_carlo (data, prior)
% Bayesian logistic regression using MCMC/sampling (Metropolis Hasting)
% -------------------------------------------------------------------------
% This script implements a Metropolis-Hastings algorithm to generate samples
% from the posterior of our logisitic problem.
% We also compute the so-called harmonic estimator of the model evidence 
% using the posterior samples.
% -------------------------------------------------------------------------

%% Metropolis-Hastings algorithm
% ========================================================================
% Here, we use a Markov Chain to generate samples and an accept/reject rule
% based on the joint density to collect samples from the posterior.

% Initialisation
% ---------------------------------------------------------------------
% number of samples
N = 1e6;
% variance of the proposal distribution
proposal_sigma = 0.15;
% starting values
theta = 0;
old = log_joint(data, prior, theta);

% Main loop
% ---------------------------------------------------------------------
for t = 2 : N  
   
    % propose a new sample with a random step (Markov-Chain jump)
    proposal = theta(t-1) + sqrt (proposal_sigma) * randn();

    % compute the (log) joint probability of the proposal
    new = log_joint (data, prior, proposal);

    % do we get warmer?
    accept_prob = exp (new - old);
    accept =  accept_prob > rand(1) ;
    if accept % if yes, confirm the jump
        theta(t) = proposal;
        old = new;
    
    else % otherwise, stay in place
        theta(t) = theta(t-1); 
    end        
end

%% Posterior characterization
% ========================================================================
% Before we can use the law of large numbers, we need to clean up the
% samples to ensure they are memory less (no effect of the starting point) 
% and independent (no autocorrelation).

% If we were doing things properly, we would have run multiple chains and
% would compute convergence scores, eg. Gelman Rubin Disagnostic...

% Remove "burn in" phase. See "Geweke diagnostic"
% ---------------------------------------------------------------------
theta(1:100) = [];

% De-correlation
% ---------------------------------------------------------------------
% Compute autocorrelation for increasing lag
for lag = 1 : 100 
    AR(lag) = corr(theta(lag+1:end)', theta(1:end-lag)');
end
% find minimum lag to have negligible autocorrelation
optlag = find(AR<.05, 1);
% decimate the samples accordingly
theta = theta(1:optlag:end);

% Posterior moments
% ---------------------------------------------------------------------
% We can now use the law of large numbers to approximate the sufficient
% statistics of the posterior distribution. 
posterior.mu = mean (theta);
posterior.sigma = var (theta);

%% Model evidence
% ========================================================================
% If one can approximate the model evidence using samples from the prior,
% it is better to do it using samples from the posterior because it better
% explores where the likelihood is high. Here, we apply the so-called Harmonic
% estimator which uses samples from the posterior: \theta_t ~ p(\theta|y)
%
% $$ p(y) \approx  N / sum [1 / p(y|\theta_t)] $$
%
% Note that this estimator tend to overestimate the evidence and is quite
% insensitive to the prior:
% See https://radfordneal.wordpress.com/2008/08/17/the-harmonic-mean-of-the-likelihood-worst-monte-carlo-method-ever/
% Better (but slightly more complicated) estimators exists,like the one in
% Chib & Jeliazkov for the Metropolis-Hastings output.

for t = 1 : numel (theta)
   ll(t) = log_likelihood (data, theta(t));
end

logEvidence = log (numel (ll)) - logsumexp (-ll);

end

% ========================================================================
% Returns log(sum(exp(a))) while avoiding numerical underflow.
function s = logsumexp(a)
    ma = max(a);
    s = ma + log(sum(exp(a - ma)));
end
    