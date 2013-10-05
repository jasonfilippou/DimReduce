function [ G ] = build_KNN_Graph( data, k )
%BUILD_KNN_GRAPH Find the k-nearest neighbors of every data point 
% and build the knn graph G.
%   The knn graph is represented as an NxN weighted sparse matrix.
%   This is the format expected by MATLAB graph functions such as
%   graphshortestpath.

% The first column of D should always be 0.
[IDX, D] = knnsearch(data, data, 'k', k, 'distance', 'euclidean');

% Initialize the weight matrix with zeroes

 G = zeros(size(data,1)); 
 
 % For every datapoint
 for i = 1:size(data,1)
     % For every neighbor other than ourselves
     for j = 2:k
         % Set the arc weight to be the Euclidean distance calculated.
         G(i, IDX(i, j)) = D(i, j);
     end
 end
G = sparse(G); % Compact storage, needed by graphshortestpath and other functions.

end

