
clear ; close all; clc


fprintf('Visualizing example dataset for outlier detection.\n\n');


load('data1.mat');

%  Visualize the example dataset
plot(X(:, 1), X(:, 2), 'bx');
axis([0 30 0 30]);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

fprintf('Program paused. Press enter to continue.\n');
pause



%
fprintf('Visualizing Gaussian fit.\n\n');

%  Estimate my and sigma2
fprintf('call  estimate Gaussian .\n\n');
[mu sigma2] = estimateGaussian(X);
fprintf('call  estimate Gaussian .\n\n');
%  Returns the density of the multivariate normal at each data point (row) 
%  of X
p = multivariateGaussian(X, mu, sigma2);

%  Visualize the fit
visualizeFit(X,  mu, sigma2);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

fprintf('Program paused. Press enter to continue.\n');
pause;



pval = multivariateGaussian(Xval, mu, sigma2);

[epsilon F1] = selectThreshold(yval, pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);

%  Find the outliers in the training set and plot the
outliers = find(p < epsilon);

%  Draw a red circle around those outliers
hold on
plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;


load('ex8data2.mat');


[mu sigma2] = estimateGaussian(X);

%  Training set 
p = multivariateGaussian(X, mu, sigma2);

%  Cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2);

%  Find the best threshold
[epsilon F1] = selectThreshold(yval, pval);

fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);

fprintf('# Outliers found: %d\n\n', sum(p < epsilon));



function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)

X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            

J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

z=X*Theta';
for i = 1:num_movies
  for j = 1:num_users
    if R(i,j)!=0;
      T(i,j)=(z(i,j))
    else
      T(i,j)=0; 
     end
  end
end
R ;
Y ;

%=====================================
T ; 


J=0.5*sum(sum((T-Y).^2));
J=J+(sum(sum(Theta).^2)*(lambda/2) + sum(sum(X).^2))*(lambda/2) ;
Theta ;
error=((T-Y));
%error=(error*Theta);
error=(error*Theta);
X_grad=sum(error)+(lambda*X);
%error2=X'*error;
error2=X'*error;
Theta_grad=sum(error2)+(lambda*Theta) ;


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
function p = multivariateGaussian(X, mu, Sigma2)

k = length(mu);

if (size(Sigma2, 2) == 1) || (size(Sigma2, 1) == 1)
    Sigma2 = diag(Sigma2);
end

X = bsxfun(@minus, X, mu(:)');
p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * ...
    exp(-0.5 * sum(bsxfun(@times, X * pinv(Sigma2), X), 2));

end
function [bestEpsilon bestF1] = selectThreshold(yval, pval)


bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    










    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end

function [Ynorm, Ymean] = normalizeRatings(Y, R)

[m, n] = size(Y);
Ymean = zeros(m, 1);
Ynorm = zeros(size(Y));
for i = 1:m
    idx = find(R(i, :) == 1);
    Ymean(i) = mean(Y(i, idx));
    Ynorm(i, idx) = Y(i, idx) - Ymean(i);
end

end

function visualizeFit(X, mu, sigma2)


[X1,X2] = meshgrid(0:.5:35); 
Z = multivariateGaussian([X1(:) X2(:)],mu,sigma2);
Z = reshape(Z,size(X1));

plot(X(:, 1), X(:, 2),'bx');
hold on;

if (sum(isinf(Z)) == 0)
    contour(X1, X2, Z, 10.^(-20:3:0)');
end
hold off;

end
function [mu sigma2] = estimateGaussian(X)

[m, n] = size(X)


mu = zeros(n, 1);
sigma2 = zeros(n, 1);

fprintf('meanfit.\n\n');
mu=(mean(X));
fprintf('meanfit.\n\n');
size(mu)
fprintf('varaiance.\n\n');
sigma2=var(X);

size(sigma2)




% =============================================================


end
