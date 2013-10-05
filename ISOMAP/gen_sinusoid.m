function [ sinusoid] = gen_sinusoid(N)
X = 10 + 10.*rand(N, 1);
Y = 10 + 10.*rand(N, 1);
sinusoid = [X, Y, sin(X)];
end