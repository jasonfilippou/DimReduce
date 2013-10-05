function [ subMat ] = get_submatrix( sparseMat, points )
%GET_SUBMATRIX Return a submatrix of the given sparse matrix by
% including only the cells (i, j), where i and j are all possible
% 2-tuples in "points".

n = size(points, 2); % points is a row vector
subMat = zeros(n);
for ind1 = 1:n
    for ind2 = 1:n
          subMat(ind1, ind2) = sparseMat(points(ind1), points(ind2));        
    end
end

subMat = sparse(subMat);
