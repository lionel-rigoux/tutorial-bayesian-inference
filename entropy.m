function H = entropy (q)
% Entropy of the univariatate Gaussian
% -------------------------------------------------------------------------
% $$ H[q] = E[- \log q(\theta)]_q = 1/2 log(2e\pi\Sigma) $$
% -------------------------------------------------------------------------
    H = 0.5 * (log (2 * pi * q.sigma) + 1);
end