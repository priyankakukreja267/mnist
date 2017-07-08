function [accuracy, loss] = ComputeAccuracyAndLoss(W, b, data, labels)
% [accuracy, loss] = ComputeAccuracyAndLoss(W, b, X, Y) computes the networks
% classification accuracy and cross entropy loss with respect to the data samples
% and ground truth labels provided in 'data' and labels'. The function should return
% the overall accuracy and the average cross-entropy loss.

% display('In function ComputeAccuracyAndLoss');
nPts = size(data, 1);
nDim = size(data, 2);
correct = 0;
loss = 0;

for i = 1:nPts
    v = Forward(W, b, data(i, :)');
    v = v';
    labels(i,:);
    [temp ,predY] = max(v);
    actualY = find(labels(i, :));
    if predY == actualY
        correct = correct + 1;
    end
    loss = loss - log(sum(v.*labels(i,:)));
end

accuracy = double(correct) / double(nPts);
loss = loss / nPts;
end
