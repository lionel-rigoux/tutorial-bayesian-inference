function [posterior, logEvidence] = invert_VBAtoolbox(data, prior)
% Bayesian logistic regression using the VBA toolbox (variational laplace)
% -------------------------------------------------------------------------
% This script is a minimal demo showing how to run the inference only given
% the specification of the model prediction, letting the toolbox do the all
% the work.
% -------------------------------------------------------------------------

%% =========================================================================
% Model definition
% =========================================================================
% Note that in the toolbox, parameters of static models are called "phi"

% mapping between input and response
function [gx] = g_logistic(~,param,input,~)
    gx = VBA_sigmoid(param * input);
end

% number of parameters
dim.n_phi = 1;

% indicate we are fitting binary data
options.sources.type = 1;
  
%% =========================================================================
% Inference
% =========================================================================

% specify the prior 
options.priors.muPhi = prior.mu;
options.priors.SigmaPhi = prior.sigma;
options.tolFun = 1e-4;
options.GNtolFun = 1e-4;

% call the inversion routine
[post, out] = VBA_NLStateSpaceModel (data.y, data.x, [], @g_logistic, dim, options);


%% =========================================================================
% Wrapping up
%  =========================================================================

% rename for consitency with the other demos
posterior.mu = post.muPhi;
posterior.sigma = post.SigmaPhi;
logEvidence = out.F;

end
