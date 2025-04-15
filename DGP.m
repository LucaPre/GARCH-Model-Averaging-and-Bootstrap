% Generates data (and variances) of DGP number i of size T with degree of
% freedom nu
function [y,sigmasq] = DGP(i, T, nu)
    y = zeros(T,1);
    sigmasq = 0.15*ones(T,1); % Preallocation of scalings
    for t = 3:(T)
        u = y(t-1)-y(t-2); % Past error
       
        % Update variance according to DGP number
        if i == 1
            sigmasq(t) = 0.1 + 0.045 * u^2 + (0.5 + 0.45 * (u / sqrt(sigmasq(t - 1)) < -0.5)) * sigmasq(t - 1); % DGP1
        elseif i == 2
            sigmasq(t) = 0.1 + 0.045 * u^2 + 0.2 * sigmasq(t - 1) + (u < 0) * (0.5 + 0.15 * u^2 + 0.35 * sigmasq(t - 1)); % DGP2
        elseif i == 3
            sigmasq(t) = (0.1 + 0.2 * (abs(u) - 0.5 * u) + 0.2 * sqrt(sigmasq(t - 1)))^2; % DGP3
        elseif i == 4
            y(2) = 0.1;
            u = y(t-1)-y(t-2);
            sigmasq(t) = 0.1 * (u^2)^0.3 * sigmasq(t - 1)^0.3; % DGP4
        end

        y(t) = y(t-1) + trnd(nu) * sqrt(sigmasq(t)); % Generate yt as random walk with random t errors 
    end
