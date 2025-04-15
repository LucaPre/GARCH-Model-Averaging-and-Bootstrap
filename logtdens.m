% Computes log density of t distribution for mean, scale sigmasq and degree
% of freedom nu
function logf = logtdens(x,mean,sigmasq,nu)
z = (x-mean)./sqrt(sigmasq); % Standardization
logf=log(gamma((nu+1)/2))-((nu+1)/2)*log(1+z.^2/nu)-log(sqrt(sigmasq))-0.5*log(nu)-log(gamma(nu/2))-0.5*log(pi); % formular for log t density