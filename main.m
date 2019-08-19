function [posterior, logEvidence] = main ()

% Simulate data according to our logisitic model with a known parameter
% -------------------------------------------------------------------------
true_theta = 3;
data = simulate_data (true_theta);

% Define a prior
% -------------------------------------------------------------------------
prior.mu = 0;
prior.sigma = 5;

% Solve the inference problem...
% -------------------------------------------------------------------------

% - using sampling (MCMC)
tic
[posterior(1), logEvidence(1), samples] = invert_monte_carlo (data, prior);
toc

% - using the variational-laplace approch
tic
[posterior(2), logEvidence(2)] = invert_variational_laplace (data, prior);
toc

% - using a dummy variational without laplace
tic
[posterior(3), logEvidence(3)] = invert_variational_numeric (data, prior);
toc

% - using variational inference with stochastic gradient ("blackbox")
tic
[posterior(4), logEvidence(4)] = invert_variational_stochastic (data, prior);
toc

% - using the VBA toolobx (variational laplace)
tic
[posterior(5), logEvidence(5)] = invert_VBAtoolbox (data, prior);
toc

% Plot the results
% -------------------------------------------------------------------------

x = linspace(- 6, 6, 200);
figure();
hold on
plot_gaussian(x, prior, 'r');
plot_likelihood(x, data, 'b');
plot_joint(x, data, prior, 'm');
plot_samples(samples, [.3 .6 .6]);
plot_gaussian(x, posterior(2), 'g');
plot_gaussian(x, posterior(4), 'g--');
legend({'prior','likelihood', 'joint', 'MCMC', 'variational Laplace', 'blackbox'});

end

% plotting helpers

function h = plot_gaussian(x, p, color)
    h = plot(x, normpdf (x, p.mu, sqrt(p.sigma)), color);
end

function plot_samples(samples, color)
    histogram (samples, ...
        'Normalization', 'pdf', ...
        'FaceColor', color, ...
        'EdgeColor', color)
end

function h = plot_likelihood(x, data, color)
    for t = 1 : numel (x)
        p(t) = exp(log_likelihood(data, x(t)));
    end
    p = p / sum (p * (x(2)-x(1)));
    h = plot(x, p, color);
end

function h = plot_joint(x, data, prior, color)
    for t = 1 : numel (x)
        p(t) = exp(log_joint(data, prior, x(t)));
    end
    p = p / sum (p * (x(2)-x(1)));
    h = plot(x, p, color);
end



