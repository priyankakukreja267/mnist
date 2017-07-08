function checkGradient(W, b, grad_W, grad_b, X, Y)
% call by putting breakpoint at the end of 'Backward.m', by using:
% function checkGradient(W, b, grad_W, grad_b, X, Y)


% for all wts:
%     find error E_lo with wt - 10^-4
%     find error E_hi with wt + 10^-4
%     
%     if grad_W of that wt = E_hi - E_lo
%         gradient is correct
%     else
%         gradient is wrong
%     end
% end    
%     

layers = [1024, 400, 26];

nLayers = length(layers);

W = cell(nLayers - 1, 1);
b = cell(nLayers - 1, 1);

% layers(1) = 1024
% layers(2) = 400
% layers(3) = 26

for i = 1:nLayers - 1
    W{i} = rand(layers(i), layers(i+1));
    b{i} = rand(1, layers(i+1));
end

load('../data/nist26_train.mat', 'train_data', 'train_labels')
X = train_data;
Y = train_labels;

nWtMatrix = size(W,1);
nBMatrix = size(b,1);
nData = size(X, 1);
% nData = 1;
for i = 2:nWtMatrix
    nNodesLo = size(W{i}, 1);
    nNodesHi = size(W{i}, 2);
    
    for k = 1:nNodesHi
        for j = 1:nNodesLo
            wActual = W{i}(j, k);
            
            W{i}(j, k) = wActual + 0.001;
            for d = 1:1
                [acc, lossHi] = ComputeAccuracyAndLoss(W, b, X(d, :), Y(d, :));
            end

            W{i}(j, k) = wActual - 0.001;
            for d = 1:1
                [acc, lossLo] = ComputeAccuracyAndLoss(W, b, X(d, :), Y(d, :));
            end
            
            computedLoss = (lossHi - lossLo) / 0.002
            
            [output, act_h, act_a] = Forward(W, b, X(d, :));
            [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a);
            gradient_at_that_point = grad_W{i}(j, k)
            
            W{i}(j,k) = wActual;
        end
    end
end


for i = 1:nBMatrix
    nNodesLo = size(b{i}, 1);
    nNodesHi = size(b{i}, 2);
    
    for j = 1:nNodesHi
        bActual = b{i}(j);

        b{i}(j) = bActual + 0.001;
        lossHi = 0;
        for d = 1:1
            [acc, lossHi] = ComputeAccuracyAndLoss(W, b, X(d, :), Y(d, :));
        end

        b{i}(j) = bActual - 0.001;
        lossLo = 0;
        for d = 1:1
            [acc, lossLo] = ComputeAccuracyAndLoss(W, b, X(d, :), Y(d, :));
        end

        computedLoss = (lossHi - lossLo) / 0.002
        gradient_at_that_point_b = grad_b{i}(j)
        
        b{i}(j) = bActual;
    end
end
