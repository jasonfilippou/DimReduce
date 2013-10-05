clear;
close all;
clc;
load face_data;
data = images'; 
n = size(data, 1);
fprintf('Data loaded!\n');
for k = 4:10
    fprintf('Building KNN graph for k =%d.\n', k); 
    G = build_KNN_Graph(data, k);
    
    % We now need to consider all possible connected components for this
    % graph, and run steps 2 and 3 of IsoMAP for all different connected
    % components. This is done because we need to avoid infs in the final
    % distance matrix, or else cMDS (which depends on the eigen-decomposition 
    % of the centered version of D) will fail.
    [S, C] = graphconncomp(G, 'Directed', false);
    assert(size(C, 2) == n, 'Assignments of components should be done for every point');
    fprintf('Found %d connected components for k = %d.\n', S, k);
    projectedData = zeros(n, 3);
    eigValAvg = zeros(1, 5);
    for i = 1:S
        fprintf('Building distance matrix for connected component %d and k =%d.\n',i, k);
        pointsInComponent = find(C == i);
        
        % We require that the points in a component are at least 5.
        % Otherwise, we discard them as noise and move on to projecting
        % the next component. The amount of minimum elements required by
        % each component has a direct consequence on the width of the
        % eigenvalue spectrum that we plot; a component with less than 5
        % points can't produce 5 eigenvalues in the eigendecomposition of 
        % its centered distance matrix!
        if size(pointsInComponent, 2) < 5
            continue;
        end
        
        component = get_submatrix(G, pointsInComponent);
        fprintf('%d points in component %d for k =%d\n', size(component, 2), i, k);
        % D = build_distance_matrix(size(component, 2), component); % SLOW
        D = graphallshortestpaths(component, 'Directed', false); % FAST
        D = D.^2;
        fprintf('Running cMDS for connected component %d and k = %d.\n', i, k);
        [projectedData(pointsInComponent, :), eigVals] = cMDS(D, 3);
        eigValAvg = eigValAvg + eigVals;
    end;
    eigValAvg = eigValAvg ./S;
    % projectedData should hold only the points that we actually projected.
    projectedData = projectedData(projectedData(:, 3) ~= 0, :);
    h = figure; bar(eigValAvg); 
    title(sprintf('Eigenvalue spectrum averaged over %d components and k=%d.', S, k));
    hold off;
    writeEPS(h, sprintf('figures/faceEigSpectrum_k=%d', k));
    fprintf('Made plot for k =%d.\n', k);
end
fprintf('Done!\n');
