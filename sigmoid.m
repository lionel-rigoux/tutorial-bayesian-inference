function z = sigmoid (x, theta)
    z = 1 ./ (1 + exp(- theta * x));
end