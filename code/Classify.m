function [outputs] = Classify(W, b, data)
% [predictions] = Classify(W, b, data) should accept the network parameters 'W'
% and 'b' as well as an DxN matrix of data sample, where D is the number of
% data samples, and N is the dimensionality of the input data. This function
% should return a vector of size DxC of network softmax output probabilities.

display('In function Classify.m')

nPts = size(data, 1);
nDim = size(data, 2);
nLayers = size(W, 1) + 1;
nCategories = size(W{nLayers - 1}, 2);
outputs = zeros(nPts, nCategories);

% [output, act_h, act_a] = Forward(W, b, X)
for i = 1:nPts
    [predY, , ] = Forward(W, b, data(i,:));
    sizePredY = size(predY);
    outputs(i, :) = predY;
end
end
