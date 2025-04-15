% Computes empirical distribution of B 1-step ahead forecasts according to
% our suggested bootstrap algorithm 

function dist = WBoot(B,nuhat,sigmas,y,What,lambda,seeds,initial)
nummod=width(sigmas); % Number of models
T=length(sigmas); % Time Series length
dist=zeros(B,1); 
parfor b=1:B 
rng(seeds(b))   % Fix seed for replication
yboot=zeros(T,1);
yboot(1)=y(1);
for t=2:T
   yboot(t)=y(t-1)+sqrt(What*sigmas(t,:)')*trnd(nuhat); % Generate yboot at period t with estimated nu and variance at time t centered around the real data at t-1
end

% Repeat estimation procedure (identical to main script except in lines
% with comments)
sigmasfit=zeros(T,nummod);
options=optimset('MaxFunEvals',10000);
forecasts=zeros(nummod,1);
K=zeros(nummod,1)';
for i=1:nummod
fixedFunction=maxset(i,nuhat,y,yboot); % get function to maximize likelihood but use bootstrap observations conditional on real observations at t-1
x0=initial{i}; % Starting value as estimate of using the real data
[thetahat,fval,exitflag]=fminsearch(fixedFunction,x0,options);

[~, sigma_sq, LogLik, sigma_sq_h1]=fixedFunction(thetahat);
K(i)=length(thetahat)-1;
nuhati=min(200,thetahat(1));
sigmasfit(:,i)=sigma_sq*nuhati/(nuhati-2);
forecasts(i)=sigma_sq_h1*nuhati/(nuhati-2);
end

x0=[nuhat What];
fixedFunction = @(x) weightcriterion(x,sigmasfit,y,K,lambda,yboot); % Function to maximize to get weight but use bootstrap observations conditional on real observations at t-1 
thetahat=fmincon(fixedFunction,x0,[],[],[0 ones(1,nummod)],1,[-Inf zeros(1,nummod)],[Inf ones(1,nummod)]);
dist(b)=thetahat(2:end)*forecasts; % Save variance that is estimated applying the weight estimator to bootstrapped data


end