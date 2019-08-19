function z = log_joint(data, prior, theta)

    % compute log joint
    z = - 0.5*log(2*pi*prior.sigma) - 0.5 * ((theta-prior.mu).^2)/prior.sigma ... % Gaussian prior
        + log_likelihood(data, theta);
end