function [ data ] = make_own_data()
% make_own_data() build data in such a way that you will not need
% multiple connected components and you will be able to debug Isomap.
    X = linspace(1, 10, 90);
    numlines = 10;
    data = zeros(numlines * length(X), 2);
    startFrom = 1; % Data point to start generating
    yoffset = 0;
    for n = 1:numlines
        data(startFrom:startFrom + length(X) - 1, 1) = X;
        data(startFrom:startFrom + length(X) - 1, 2) = yoffset;
        startFrom = startFrom + length(X);
        %yoffset = yoffset + 0.2;
        yoffset = yoffset + 0.3;
    end    
end