% Helper function to fixate functions for minimization, get starting values
% centered around constant variance process and extract the volatility process for a model i

function [fixedFunction,x0,sigmafunction]=maxset(i,nu,y,yboot)

if i==1
    fixedFunction = @(x) tGARCHLikelihood(x,y,yboot); 
    x0=[nu 0.15 0 0];
    sigmafunction=@(sigmalag,u,params) params(1)+params(2)*u^2+params(3)*sigmalag;
end

if i==2
    fixedFunction = @(x) tARCHLikelihood(x,y,yboot); 
    x0=[nu 0.15 0];
    sigmafunction=@(sigmalag,u,params) params(1)+params(2)*u^2;
end

if i==3
    fixedFunction = @(x) tGJRLikelihood(x,y,yboot); 
    x0=([nu 0.15  0 0  0]);
    sigmafunction=@(sigmalag,u,params) params(1)+(params(2)+params(4)*(u<0))*u^2+params(3)*sigmalag;
end

if i==4
    fixedFunction = @(x) tEGARCHLikelihood(x,y,yboot); 
    x0=[nu log(0.15) 0 0 0];
    sigmafunction=@(sigmalag,u,params) exp(params(1)+params(2)*u+params(4)*abs(u)+params(3)*log(sigmalag));
end

if i==5
    fixedFunction = @(x) tTSLikelihood(x,y,yboot); 
    x0=[nu sqrt(0.15)   0   0];
    sigmafunction=@(sigmalag,u,params) (params(1)+params(2)*abs(u)+params(3)*sqrt(sigmalag))^2;
end

if i==6
    fixedFunction = @(x) tAPGARCHLikelihood(x,y,yboot); 
    x0=[(nu) (sqrt(0.15)) 0 0 0 1];
    sigmafunction=@(sigmalag,u,params) (params(1)+params(2)*(abs(u)+params(4)*u)^params(5)+params(3)*sqrt(sigmalag)^params(5))^(2/params(5));
end


