% Computes log-likelihood (and more) of GARCH model

function [Q, sigma_sq, sumLik, sigma_sq_h1, params] = tGARCHLikelihood(theta,y,yboot)
nu=min(200,(theta(1))); % Degree of freedom 
% GARCH specific parameters
a0=(theta(2));
a=(theta(3));
b=(theta(4));
params=[a0 a b];

T=size(y,1); % Sample size
sigmasq_initial=var(y(2:T)-y(1:T-1))*(nu-2)/nu; % Initialize variance as unconditional variance 

LogLik=zeros(T-1,1);
sigma_sq=zeros(T,1);
sigma_sq(2)=sigmasq_initial;
for t=1:1
    LogLik(t)=logtdens(yboot(t+1),y(t),sigmasq_initial,nu); % Log Likelihood of first period
end

for t=3:T
    u=y(t-1)-y(t-2); % Error of past period
    sigma_sq(t)=a0+a*u.^2+b*sigma_sq(t-1); % Update our variance
    LogLik(t-1)=logtdens(yboot(t),y(t-1),sigma_sq(t),nu); % Evaluate period t log likelihood
end

Q=-sum(LogLik); % Criterion to minimize (negative of log-likelihood)
sumLik=sum(LogLik);

% One period variance forecast
t=t+1;
u=y(t-1)-y(t-2);
sigma_sq_h1=a0+a*u.^2+b*sigma_sq(t-1);


% Enforce degree of freedom larger than 2
if nu<=2
Q=1e+50+randn(1);
end

% Enforce parameters to be larger than 0
if min(params)<0
    Q=1e+50+randn(1);
end

% Rule out infinity or not defined expressions
if Q==Inf
    Q=1e+50+randn(1);
end

if isnan(Q)==1
    Q=1e+50+randn(1);
end