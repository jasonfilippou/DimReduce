function [ swissroll] = gen_swissroll(N)
xmin = pi;
xmax = 3*pi;
ymin = 0;
ymax = 35;
X = xmin + (xmax - xmin) * rand(N, 1);
Y = ymin + (ymax - ymin) * rand(N, 1);
swissroll = [X.*cos(X), Y, X.*sin(X)];
end