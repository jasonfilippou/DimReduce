clear;
close all;
clc;
%[swissroll] = gen_swissroll(2000);
%n = size(swissroll, 1);
%figure;scatter3(swissroll(:, 1),swissroll(:,2), swissroll(:, 3), 10, swissroll(:, 1));
load data;
figure;scatter(data(:, 1),data(:,2), 10, data(:, 1));
n = size(data, 1);
axis equal;
title('Original data');
hold off;
fprintf('Data loaded (%d examples) and plotted.\n', n);
%% For various values of k, build the KNN graph and run Isomap.
%matlabpool(3);
for k =3:10
    fprintf('Building KNN graph for k =%d.\n', k); 
    G = build_KNN_Graph(data, k);
    [S, ~] = graphconncomp(G, 'Directed', false);
    fprintf('For k = %d, number of components = %d.\n', k, S);
    fprintf('Building distance matrix for k =%d.\n', k); 
    %D = build_distance_matrix(n, G); % SLOW
    D = graphallshortestpaths(G, 'Directed', false); % FAST
    D = D.^2;
    
    fprintf('Running cMDS for k =%d.\n', k); 
    [projectedData, eigVals] = cMDS(D, 2);
    % Why does it never plot in multi-threaded mode...
    figure; scatter(projectedData(:,1), projectedData(:, 2), 10, projectedData(:, 1));
    axis equal;
    title(sprintf('projection for k=%d', k));
    hold off;
    figure; bar(eigVals); 
    title(sprintf('Eigenvalue spectrum, for k =%d', k)); 
    hold off;
end
fprintf('Done!\n');
