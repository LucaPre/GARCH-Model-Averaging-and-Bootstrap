% Performs Bootstrap for 1-step ahead volatility forecast for 
% GARCH-type model indicated by "ind" similar to 
% Lorenzo Pascual, Juan Romo, Esther Ruiz, Bootstrap prediction for returns
% and volatilities in GARCH models, Computational Statistics & Data Analysis,
% Volume 50, Issue 9, 2006, Pages 2293-2312
% Difference is that the errors are generated using the t distribution instead
% of empirical residuals (which is valid as long as the true DGP is using t
% errors as well)

function sigmadist = BootParam(B,thetahat,sigmas,y,seeds,ind)
nuhat=min(200,thetahat(1));
sigmadist=zeros(B,1);
T=length(y);

% Get functions and parameters of the GARCH model that you want to use 
[fixedFunction,~,sigmafunction]=maxset(ind,nuhat,y,y); 
[~,~,~,~,params]=fixedFunction(thetahat);
parfor b=1:B 
    rng(seeds(b))
    yboot=zeros(T,1);
    yboot(1)=y(1);
    sigmasq=sigmas(:,ind)*(nuhat-2)/nuhat;
    for t=2:T
        if t>2
        sigmasq(t)=sigmafunction(sigmasq(t-1),yboot(t-1)-yboot(t-2),params); % Generate sigma according to function of the selected model
        end
        yboot(t)=yboot(t-1)+sqrt(sigmasq(t))*trnd(nuhat); % Generate bootstrap data recursively
    end

    options=optimset('MaxFunEvals',10000);
    fixedFunction = maxset(ind,nuhat,yboot,yboot);
    thetahatboot=fminsearch(fixedFunction,thetahat,options); % Estimate GARCH parameters with bootstrap data

    fixedFunction = maxset(ind,nuhat,y,y);
    [~, ~, ~, sigma_sq_h1] = fixedFunction(thetahatboot); % Get sigma forecast with actual data and newly estimated parameters
    nuhatboot=min(200,thetahatboot(1)); 
    sigmadist(b)=sigma_sq_h1*nuhatboot/(nuhatboot-2); % Save variance forecast with bootstrap data
end