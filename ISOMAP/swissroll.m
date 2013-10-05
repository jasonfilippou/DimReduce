clear;
close all;
clc;
swissroll_data = gen_swissroll(2500);
n = size(swissroll_data, 1);
h = figure;scatter3(swissroll_data(:, 1),swissroll_data(:,2), swissroll_data(:, 3), 10, swissroll_data(:, 1));
axis equal;
title('Swissroll data in 3D');
hold off;
input('wait');
fprintf('Data loaded (%d examples) and plotted.\n', n);
%matlabpool(4);
for k = 4:10
    fprintf('Building KNN graph for k =%d.\n', k); 
    G = build_KNN_Graph(swissroll_data, k);
    
    % We now need to consider all possible connected components for this
    % graph, and run steps 2 and 3 of IsoMAP for all different connected
    % components. This is done because we need to avoid infs in the final
    % distance matrix, or else cMDS (which depends on SVD of the centered
    % version of D) will fail.
    [S, C] = graphconncomp(G, 'Directed', false);
    assert(size(C, 2) == n, 'Assignments of components should be done for every point');
    fprintf('Found %d connected components for k = %d.\n', S, k);
    projectedData = zeros(n, 3);
    eigValAvg = zeros(1, 5);
    for i = 1:S
        fprintf('Building distance matrix for connected component %d and k =%d.\n',i, k);
        pointsInComponent = find(C == i);
        
        % TODO: We require that the points in a component are at least 5.
        % Otherwise, we discard them as noise and move on to projecting
        % the next component. This could be used to show why dense sampling
        % is very important in Isomap.
        if size(pointsInComponent, 2) < 5
            continue;
        end
        
        component = get_submatrix(G, pointsInComponent);
        fprintf('%d points in component %d for k =%d\n', size(component, 2), i, k);
        % D = build_distance_matrix(size(component, 2), component); % SLOW
        D = graphallshortestpaths(component, 'Directed', false); % FAST
        D = D.^2;
        % Save the freshly created distance matrix on disk.
        %save(strcat('distanceMat_custom_data_k=', strcat(num2str(k), '.mat')), 'D');
        fprintf('Running cMDS for connected component %d and k = %d.\n', i, k);
        [projectedData(pointsInComponent, :), eigVals] = cMDS(D, 3);
        eigValAvg = eigValAvg + eigVals;
        %figure;scatter3(X(:, 1),X(:,2), X(:,3), 10, cmap); 
        %title(strcat('"Unfolded" data for k = ', num2str(k)));
        %hold off; 
    end;
    eigValAvg = eigValAvg ./S;
    % projectedData should hold only the points that we actually projected.
    projectedData = projectedData(projectedData(:, 3) ~= 0, :); % logical indexing fine here
    %thetatokeep = theta(find(projectedData(:, 3) ~= 0)); % but not here
    save_var_to_file1(sprintf('projectedData_k=%d.mat', k), projectedData);
    %save_var_to_file2(sprintf('thetatokeep_k=%d.mat', k), thetatokeep);
    save_var_to_file3(sprintf('eigVals_k=%d.mat', k), eigValAvg);
    fprintf('Saved projection for k =%d.\n', k);
    %cmap2 = jet(size(projectedData, 1));
    % Won't plot in multi-threaded mode, but whatever.
    h = figure; scatter(projectedData(:, 1), projectedData(:, 2), 10, projectedData(:, 1));
    axis equal;
    title(strcat('Isomap projection for k =', num2str(k)));
    hold off;
    writeEPS(h, sprintf('figures/swissroll_embedding_k=%d', k));
    h = figure; bar(eigValAvg); 
    title(sprintf('Eigenvalue spectrum averaged over %d components and k=%d.', S, k));
    hold off;
    writeEPS(h, sprintf('figures/swissroll_barplot_k=%d', k));
    fprintf('Made plots for k =%d.\n', k);
end
%matlabpool close;
fprintf('Done!\n');
