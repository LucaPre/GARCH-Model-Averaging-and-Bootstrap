clear
clc
MC=1000+1; % Number of Monte Carlo simulations
IS=zeros(MC,5); % Preallocate IS distances
nu=10; % Degree of freedom to generate data
DGPnum=1; % Choose which DGP to generate data from
nummod=4; % Choose how many models to use
x0s=cell(1,nummod); % Preallocate starting values
h=5; % Forecast horizon
Tsimul=500; % Sample size
ForecastDist=zeros(MC,h,5); % Preallocate IS distance for Forecast
burn=100; % Number of burn in periods
hsimul=1000; % Number of simulations for h-step forecast h>1
rng(666)
seeds=randi([1 2^32-1],MC,1); % Seeds for replication

for m=1:MC

    if m==1
        T=10000; % For first simulation use sample size 10000
    else
        T=Tsimul;
    end
rng(seeds(m))
[y,sigmasqtrue]=DGP(DGPnum,T+h+burn,nu); % Generate data
sigmafut=sigmasqtrue(end-h+1:end)*nu/(nu-2); % Extract true future variances
y=y(burn+1:end-h); % Use data after burn in before forecast period
sigmasqtrue=sigmasqtrue(burn+2:end-h)*nu/(nu-2); % Save true variances

sigmasfit=zeros(T,nummod); % Preallocate fitted variances
options=optimset('MaxFunEvals',10000);
ICs=zeros(nummod,2); % Preallocate information criteria
K=zeros(nummod,1)'; % Preallocation K
modelforecasts=zeros(h,nummod); % Preallocate Forecasts
nuhats=zeros(1,nummod); % Preallocate estimated degree of freedoms

% Repeat the following steps for each GARCH-type model
for i=1:nummod
[fixedFunction,x0]=maxset(i,nu,y,y);
if m>1
    x0=x0s{i}; % Use estimates from first large simulation as starting values
end

x0(1)=nu; % Starting value for degree of freedom as true degree of freedom
[thetahat,fval,exitflag]=fminsearch(fixedFunction,x0,options); % Minimize the negative likelihood
thetahatfminsearch=thetahat;
fvalfminsearch=fval;

if m==1
    exitflag=0; % Use global solver for first simulation
end

% Use global optimizer in case there is a convergence problem of the
% minimizer
while exitflag==0
opts = optimoptions(@fmincon,'Display','off');
gs=GlobalSearch;
problem = createOptimProblem('fmincon','x0',thetahat,'objective',fixedFunction,'options',opts);
[thetahat, fval,exitflag] = run(gs,problem); 
if fval>fvalfminsearch
    thetahat=thetahatfminsearch; % Check if global optimizer gives better result than the local one
end
end

if m==1
x0s{i}=thetahat; % Update starting values as estimate from first large simulation
end

nuhats(i)=min(200,(thetahat(1))); % Estimated degree of freedom
[~, sigma_sq, LogLik, ~, params]=fixedFunction(thetahat);

ICs(i,1)=-2*LogLik+log(T)*(-1+length(thetahat)); % BIC information criterion
ICs(i,2)=-2*LogLik+2*(-1+length(thetahat)); % AIC information criterion
K(i)=length(thetahat)-1; % Update K
sigmasfit(:,i)=sigma_sq*nuhats(i)/(nuhats(i)-2); % Save fitted variances

% simulated h step forecasts
[~,~,sigmafunction]=maxset(i,nuhats(i),y,y); % Return function for variance process
sigmasimul=zeros(hsimul,h); 
rng(seeds(m))
% Repeat simulations hsimul times
for ii=1:hsimul
    explosive=1;
    while explosive==1
    u=y(end)-y(end-1);
    sigmalag=sigma_sq(end);
    for j=1:h
sigmasimul(ii,j)=sigmafunction(sigmalag,u,params); % Simulate Sigma one step ahead
sigmalag=sigmasimul(ii,j); % Update Sigma
u=trnd(min(200,exp(thetahat(1))),1)*sqrt(sigmalag); % Simulate error
    end

    % Check for explosiveness of variance
    if isnan(mean(sigmasimul(ii,:)))==1
        explosive=1;
    elseif mean(sigmasimul(ii,:))==Inf
        explosive=1;
    else
        explosive=0;
    end
    end
end

modelforecasts(:,i)=mean(sigmasimul)'*nuhats(i)/(nuhats(i)-2); % Estimate the forecast as mean of simulations

% Use median instead of mean for Egarch if individual simulations become explosive and dominate the mean
if i==4
modelforecasts(:,i)=mean(sigmasimul)*nuhats(i)/(nuhats(i)-2);
if max(modelforecasts(:,i)./modelforecasts(:,i-1))>2
modelforecasts(:,i)=median(sigmasimul)*nuhats(i)/(nuhats(i)-2); 
end
if min(modelforecasts(:,i)./modelforecasts(:,i-1))<0.5
modelforecasts(:,i)=median(sigmasimul)*nuhats(i)/(nuhats(i)-2);
end
end

end

x0=[nu 1/nummod*ones(1,nummod)]; % Starting value at equal weighting 

% Weight estimator Lambda=0
fixedFunction = @(x) weightcriterion(x,sigmasfit,y,K,0,y);  
thetahat=fmincon(fixedFunction,x0,[],[],[0 ones(1,nummod)],1,[-Inf zeros(1,nummod)],[Inf ones(1,nummod)]); % Optimization subject to weights adding to 1 and being between 0 and 1
What=thetahat(2:end); % Estimated weight
sigmasqhat=(What*sigmasfit(2:end,:)')'; % Estimated variances

% Weight estimator Lambda=1
fixedFunction = @(x) weightcriterion(x,sigmasfit,y,K,1,y);  
thetahat=fmincon(fixedFunction,x0,[],[],[0 ones(1,nummod)],1,[-Inf zeros(1,nummod)],[Inf ones(1,nummod)]);
WhatAIC=thetahat(2:end);
sigmasqhataic=(WhatAIC*sigmasfit(2:end,:)')';

% Weight estimator Lambda=0.5log(T)
fixedFunction = @(x) weightcriterion(x,sigmasfit,y,K,0.5*log(T),y);  
thetahat=fmincon(fixedFunction,x0,[],[],[0 ones(1,nummod)],1,[-Inf zeros(1,nummod)],[Inf ones(1,nummod)]);
WhatBIC=thetahat(2:end);
sigmasqhatbic=(WhatBIC*sigmasfit(2:end,:)')';

% AIC and BIC selected models
aicsel=find(ICs(:,2)==min(ICs(:,2))); % Find minimum of AIC
bicsel=find(ICs(:,1)==min(ICs(:,1))); % Find minimum of BIC
sigmaaic=sigmasfit(2:end,aicsel); % Variances of AIC minimizing model
sigmabic=sigmasfit(2:end,bicsel); % Variances of BIC minimizing model

% Save IS distances for weighting estimator with different lambdas and AIC
% and BIC model
IS1=mean(sigmasqtrue./(sigmasqhat)-log(sigmasqtrue./(sigmasqhat))-1);
IS2=mean(sigmasqtrue./(sigmasqhataic)-log(sigmasqtrue./(sigmasqhataic))-1);
IS3=mean(sigmasqtrue./(sigmasqhatbic)-log(sigmasqtrue./(sigmasqhatbic))-1);
IS4=mean(sigmasqtrue./(sigmaaic)-log(sigmasqtrue./(sigmaaic))-1);
IS5=mean(sigmasqtrue./(sigmabic)-log(sigmasqtrue./(sigmabic))-1);
IS(m,:)=[IS1 IS2 IS3 IS4 IS5];

% Save IS distances for Forecast for weighting estimator with different lambdas and AIC
% and BIC model
ForecastDist(m,:,1)=sigmafut./(What*modelforecasts')'-log(sigmafut./(What*modelforecasts')')-1;
ForecastDist(m,:,2)=sigmafut./(WhatAIC*modelforecasts')'-log(sigmafut./(WhatAIC*modelforecasts')')-1;
ForecastDist(m,:,3)=sigmafut./(WhatBIC*modelforecasts')'-log(sigmafut./(WhatBIC*modelforecasts')')-1;
ForecastDist(m,:,4)=sigmafut./(modelforecasts(:,aicsel))-log(sigmafut./(modelforecasts(:,aicsel)))-1;
ForecastDist(m,:,5)=sigmafut./(modelforecasts(:,bicsel))-log(sigmafut./(modelforecasts(:,bicsel)))-1;

clc
fprintf('%1.0f Simulations \n',m)
% Mean of IS distance of volatility vector relative to benchmark
[mean(IS(2:m,2)) mean(IS(2:m,3)) mean(IS(2:m,4)) mean(IS(2:m,5))]./mean(IS(2:m,1))

% Mean of IS distance of volatility forecasts relative to benchmark
[mean(ForecastDist(2:m,:,2)); mean(ForecastDist(2:m,:,3)); mean(ForecastDist(2:m,:,4)); mean(ForecastDist(2:m,:,5))]./mean(ForecastDist(2:m,:,1))

end
