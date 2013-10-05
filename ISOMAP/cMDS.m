function [ X, retEigvals ] = cMDS( D, dim )
%cMDS Classical Multi-Dimensional Scaling (MDS)
%   Parameters: 
%   D, a symmetric matrix of distances (not necessarily Euclidean)
%   dim: the target dimensionality to compute the low-rank approximation with.
%   Returns: X = (U * S^(1/2))[n x dim], where n is the number of rows    
%   and dim signifies the number of first columns of the matrix product to
%   return.
%   eigVals the first 5 eigenvalues returned by eig

N = size(D, 1);
H = eye(N) - 1/N * ones(N, 1) * ones(N,1)'; % centering matrix
B = (-1/2)*H*D*H;
[U, S, ~] = svd(B);

% Because the distances are not necessarily Euclidean, we need to project 
% B onto the cone of p.s.d matrices. To do that, we need to make the 
% negative eigenvalues of B zero.

eigvals = diag(S);
eigvals(eigvals < 0) = 0;
S = diag(sqrt(eigvals));
[U, S] = sort_eigs(U, S);
% We now take the first "dim" columns of (U * S^(1/2).
M = U * S; 
X =  M(:, 1:dim);
retEigvals = eigvals(1: 5)';
end

