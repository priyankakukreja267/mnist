function [W, b] = Train(W, b, train_data, train_label, learning_rate)
% [W, b] = Train(W, b, train_data, train_label, learning_rate) trains the network
% for one epoch on the input training data 'train_data' and 'train_label'. This
% function should returned the updated network parameters 'W' and 'b' after
% performing backprop on every data sample.


% This loop template simply prints the loop status in a non-verbose way.
% Feel free to use it or discard it

% train_data: D X N
% train_label: D X 1

nData = size(train_data, 1);
finalGradW = cell(size(W, 1), 1);
finalGradB = cell(size(b, 1), 1);

isBatch = 0;
batchSize = 12;

if isBatch == 1 % Doing Batch gradient update    
%   BATCH GRADIENT UPDATE
    display('Using Batch Update rule');
    
    % Initialise the finalGradient with all zeros
    for j = 1:size(finalGradW, 1)
        finalGradW{j} = zeros(size(W{j}));
        finalGradB{j} = zeros(size(b{j}));
    end

    % Run once on entire dataset
    for t = 1:nData
        [output, act_h, act_a] = Forward(W, b, train_data(t,:)); % output is same as act_h{nLayers - 1}
        
        [grad_W, grad_b] = Backward(W, b, train_data(t,:), train_label(t, :), act_h, act_a);
        for j = 1:size(grad_W, 1)
            finalGradW{j} = finalGradW{j} + grad_W{j};
            finalGradB{j} = finalGradB{j} + grad_b{j};
        end
    end

    % Update Parameters
    for j = 1:size(grad_W, 1)
        finalGradW{j} = finalGradW{j}/nData;
        finalGradB{j} = finalGradB{j}/nData;
    end
    [W, b] = UpdateParameters(W, b, finalGradW, finalGradB, learning_rate);

else
%   STOCHASTIC GRADIENT UPDATE with MINI BATCH of size 12.
    display('Using Stochastic Update rule with a batch size of 12');

    for t = 1:batchSize:nData
        % Initialise the finalGradient with all zeros
        for j = 1:size(finalGradW, 1)
            finalGradW{j} = zeros(size(W{j}));
            finalGradB{j} = zeros(size(b{j}));
        end

        cap = min(nData - t + 1, batchSize); % max number of records to consider
        for batch = 0:cap-1
            [output, act_h, act_a] = Forward(W, b, train_data(t + batch,:)');
            [grad_W, grad_b] = Backward(W, b, train_data(t + batch,:)', train_label(t + batch, :)', act_h, act_a);
            finalGradW{1} = finalGradW{1} + grad_W{1};
            finalGradW{2} = finalGradW{2} + grad_W{2};
            finalGradB{1} = finalGradB{1} + grad_b{1};
            finalGradB{2} = finalGradB{2} + grad_b{2};
        end

        % Update Parameters
        finalGradW{1} = finalGradW{1}/batchSize;
        finalGradW{2} = finalGradW{2}/batchSize;
        finalGradB{1} = finalGradB{1}/batchSize;
        finalGradB{2} = finalGradB{2}/batchSize;
        [W, b] = UpdateParameters(W, b, finalGradW, finalGradB, learning_rate);
    end
end

end