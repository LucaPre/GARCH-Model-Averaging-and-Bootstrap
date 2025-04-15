clear
clc
MC=500+1; % Number of Monte Carlo simulations
coverage=ones(MC,4); % Preallocate coverage rates of confidence intervals
Tsimul=500; % Sample size
B=100; % Number of Bootstrap simulations
nu=10; % Degree of freedom to generate data
lambda=0; % Penalty for large models for weighting estimator
DGPnum=3; % DGP number (here APGARCH)
nummod=6; % Number of models used for estimation
x0s=cell(1,nummod); % Preallocate starting values
burn=100; % Number of burn in periods
rng(666)
seeds=randi([1 2^32-1],MC,1); % Seeds for replication
for m=1:MC
    if m==1
        T=10000; % For first simulation use sample size 10000
    else
        T=Tsimul;
    end
rng(seeds(m)) 
[y,sigmasqtrue]=DGP(DGPnum,T+burn,nu); % Generate data
y=y(burn:end-1); % Use data after burn in before forecast period
fut=sigmasqtrue(end); 
fut=nu/(nu-2)*fut; % Save future variance that we want to estimate

sigmasfit=zeros(T,nummod); % Preallocate fitted variances
options=optimset('MaxFunEvals',10000);
forecasts=zeros(nummod,1); % Preallocate Forecasts
K=zeros(nummod,1)'; % Preallocation K
thetahats=cell(1,nummod); % Preallocate estimated coefficients
AIC=zeros(nummod,1); % Preallocate Akaike information criteria

% Repeat the following for each model in the model set
for i=1:nummod
[fixedFunction,x0]=maxset(i,nu,y,y); % Set up starting value and function for optimization for model i
if m>1
    x0=x0s{i}; % Use estimates from first large simulation as starting values
end
x0(1)=nu;
[thetahat,fval,exitflag]=fminsearch(fixedFunction,x0,options); % Minimize the negative likelihood
if m==1
    exitflag=0; % Use global solver for first simulation
end

% Use global optimizer in case there is a convergence problem of the
% minimizer
if exitflag==0
opts = optimoptions(@fmincon,'Display','off');
gs=GlobalSearch;
problem = createOptimProblem('fmincon','x0',thetahat,'objective',fixedFunction,'options',opts);
[thetahat, fval,exitflag] = run(gs,problem); 
end

if m==1
x0s{i}=thetahat; % Update starting values as estimate from first large simulation
end

thetahats{i}=thetahat; % Save estimates of model i
[~, sigma_sq, LogLik, sigma_sq_h1]=fixedFunction(thetahat);
K(i)=length(thetahat)-1;
AIC(i)=-2*LogLik+2*(-1+length(thetahat)); % Save AIC
nuhat=min(200,thetahat(1));
sigmasfit(:,i)=sigma_sq*nuhat/(nuhat-2); % Save fitted variances
forecasts(i)=sigma_sq_h1*nuhat/(nuhat-2); % Save forecasts
end

if m>1 % Leave the next steps out for first simulation which only exists to get good starting values

% Weight estimator 
x0=[nu 1/nummod*ones(1,nummod)]; % Starting value for weight estimates
fixedFunction = @(x) weightcriterion(x,sigmasfit,y,K,lambda,y);  
thetahat=fmincon(fixedFunction,x0,[],[],[0 ones(1,nummod)],1,[-Inf zeros(1,nummod)],[Inf ones(1,nummod)]);  % Optimization subject to weights adding to 1 and being between 0 and 1
What=thetahat(2:end); % Estimated weights
sigmasqhat=What*forecasts; % Forecast as weighted forecast of all models
nuhat=min(200,thetahat(1)); % Estimated degree of freedom

aicsel=find(AIC==min(AIC)); % AIC selected model
aicsel=aicsel(1);

seedsboot=randi([1 2^32-1],B,1);
dist=WBoot(B,nuhat,sigmasfit*(nuhat-2)/nuhat,y,What,lambda,seedsboot,thetahats); % Bootstrap density using the weighting estimator and our suggested method
distparam=BootParam(B,thetahats{aicsel},sigmasfit,y,seedsboot,aicsel); % Recursive Bootstrap with AIC selected model

% 90% Intervals for weight-bootstrap
lower=quantile(dist,0.05);
upper=quantile(dist,0.95);
if  fut<lower(1)
    coverage(m,1)=0;
end
if  fut>upper(1)
    coverage(m,1)=0;
end

% 95% Intervals for weight-bootstrap
lower=quantile(dist,0.025);
upper=quantile(dist,0.975);
if  fut<lower(1)
    coverage(m,2)=0;
end
if  fut>upper(1)
    coverage(m,2)=0;
end

% 90% Intervals for recursive bootstrap
lower=quantile(distparam,0.05);
upper=quantile(distparam,0.95);
if  fut<lower(1)
    coverage(m,3)=0;
end
if  fut>upper(1)
    coverage(m,3)=0;
end

% 95% Intervals for recursive bootstrap
lower=quantile(distparam,0.025);
upper=quantile(distparam,0.975);
if  fut<lower(1)
    coverage(m,4)=0;
end
if  fut>upper(1)
    coverage(m,4)=0;
end


clc
fprintf('%1.0f Simulations \n',m)
fprintf('Coverage rate of 0.9 interval is %1.4f for Weight-Bootstrap \n',mean(coverage(2:m,1)))
fprintf('Coverage rate of 0.9 interval is %1.4f for Recursive Bootstrap \n',mean(coverage(2:m,3)))
fprintf('Coverage rate of 0.95 interval is %1.4f for Weight-Bootstrap \n',mean(coverage(2:m,2)))
fprintf('Coverage rate of 0.95 interval is %1.4f for Recursive Bootstrap \n',mean(coverage(2:m,4)))
What
end

end