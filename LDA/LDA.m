function [ w ] = LDA( class1, class2 )
%Performs Linear Discriminant Analysis for a binary classification setting.
%   Input: examples from both classes
%   Output: optimal projection vector w* (eigenvector corresponding to
%   largest eigenvalue of $S_W^{-1}S_B$)

%% Compute class means

mu1 = mean(class1)';
mu2 = mean(class2)';

%% Compute "within scatter" matrix Sw.

S1 = cov(class1);
S2 = cov(class2);
Sw = S1 + S2;

%% Compute optimal projection using the closed-form solution we talked about
% in class. $S_w$ will have full rank as long as the data has been reduced
% to $N-c$ dimensions, which is the trick empoyed in the "Fisherfaces" paper.
% For numerical accuracy purposes, we employ the Moore-Penrose 
% pseudoinverse, since $S_w$ is nearly singular.

w = pinv(Sw) * (mu1 - mu2);

end

