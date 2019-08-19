function data = simulate_data (theta)
% Simulate binary responses from a logisitc model
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------

% true parameter of the model: the slope of the sigmoid mapping
data.theta = theta;

% experimental manipulation (eg: stimulus intensity)
data.x = linspace(-5,5,100);

% predictions of the model
data.sx = sigmoid (data.x, data.theta);

% generate binary responses from model predictions
data.y = +(rand(1, numel(data.x)) < data.sx) ;


