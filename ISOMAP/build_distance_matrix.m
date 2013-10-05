function [ D ] = build_distance_matrix(n, G )
%BUILD_DISTANCE_MATRIX Builds a distance matrix between points on the graph
% G by measuring their distances as that of the shortest path between them
% on G. Warning: This function is slower than using graphallshortestpath.

D = zeros(n);
for i = 1:n
    for j = 1:n
        %fprintf('%d, %d\n', i, j); % For tracing our progress.
        if j > i
            [D(i, j), ~, ~] = graphshortestpath(G, i, j, 'Directed', false);
        else
            D(i, j) = D(j, i); % Just a reference copy to speed things up,
                                % since the matrix is symmetric.
        end
    end
end
%fprintf('%s\n', 'Done!'); % For tracing our progress.
end

