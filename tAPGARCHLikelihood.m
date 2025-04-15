% Code equivalent to tGARCHLikelihood except for parameter definition and variance process
% and in lines with comments

function [Q, sigma_sq, sumLik, sigma_sq_h1, params] = tAPGARCHLikelihood(theta,y,yboot)
a0=(theta(2));
nu=min(200,(theta(1)));
a=(theta(3));
b=(theta(4));
c=theta(5);
delta=(theta(6));
params=[a0 a b c delta];

T=size(y,1);
sigmasq_initial=var(y(2:T)-y(1:T-1))*(nu-2)/nu;
LogLik=zeros(T-1,1);
sigma_sq=zeros(T,1);
sigma_sq(2)=sigmasq_initial;
for t=1:1
    LogLik(t)=logtdens(yboot(t+1),y(t),sigmasq_initial,nu);
end
for t=3:T
    u=(y(t-1)-y(t-2));
    sigma_sq(t)=(a0+a*(abs(u)+c*u)^delta+b*sqrt(sigma_sq(t-1))^delta)^(2/delta);
    LogLik(t-1)=logtdens(yboot(t),y(t-1),sigma_sq(t),nu);
end
Q=-sum(LogLik);
sumLik=sum(LogLik);
t=t+1;
u=(y(t-1)-y(t-2));
sigma_sq_h1=(a0+a*(abs(u)+c*u)^delta+b*sqrt(sigma_sq(t-1))^delta)^(2/delta);

if abs(c)>=1
    Q=1e+50+randn(1); % Enforce c to be smaller than 1 in absolute for stability
end

if nu<=2
Q=1e+50+randn(1);
end

if Q==Inf
    Q=1e+50+randn(1);
end

if isnan(Q)==1
    Q=1e+50+randn(1);
end

if min([a0 a b delta])<0
       Q=1e+50+randn(1);
end 
