function ll = log_likelihood(data, theta)

   % model prediction
    sx = sigmoid(data.x, theta);
    % avoid numerical overflow
    sx = max(min(sx,1-eps), eps);
    % compute log joint
    ll = sum( ... % aggregate over obsevations
            data.y.*log(sx) + (1-data.y).*log(1-sx) ... %binomial log-likelihood
        );
end