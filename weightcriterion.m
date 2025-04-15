% Computes the criterion to minimize to estimate weights 

function Q = weightcriterion(theta,sigmas,y,K,lambda,yboot)
nu=min(200,(theta(1))); % Degree of freedom
W=theta(2:end); % Weights
sigmasqhat=sigmas*W'*(nu-2)/nu; % Variances as weighted average of fitted GARCH variances
z=(yboot(2:end)-y(1:end-1))./sqrt(sigmasqhat(2:end)); % Standardization
LogLik=log(gamma((nu+1)/2))-((nu+1)/2)*log(1+z.^2/nu)-log(sqrt(sigmasqhat(2:end)))-0.5*log(nu)-log(gamma(nu/2))-0.5*log(pi); % Log-Likelihood

Q=-sum(LogLik)+lambda*K*W'; % Criterion as sum of negative log likelihood and weighted punishment for overly parameterized models

if nu<=2
    Q=1e+50+randn(1); % Enforce degree of freedom larger than 2
end