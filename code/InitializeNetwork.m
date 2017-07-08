function [W, b] = InitializeNetwork(layers)
nLayers = length(layers);
W = cell(nLayers - 1, 1);
b = cell(nLayers - 1, 1);

for i = 1:nLayers - 1
    lo = - sqrt(6) / (sqrt(layers(i) + layers(i+1)));
    hi = sqrt(6) / (sqrt(layers(i) + layers(i+1)));
    W{i} = lo + (hi - lo)*rand(layers(i), layers(i+1));
    b{i} = lo + (hi - lo)*rand(1, layers(i+1));
end
end
