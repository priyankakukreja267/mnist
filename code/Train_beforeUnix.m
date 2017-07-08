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
        
        t
        output
        [valY, predY] = max(output)
        actualY = find(train_label(t,:))
        
        [grad_W, grad_b] = Backward(W, b, train_data(t,:), train_label(t, :), act_h, act_a);
        for j = 1:size(grad_W)
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
%   STOCHASTIC GRADIENT UPDATE
    display('Using Stochastic Update rule');
    for t = 1:nData
        [output, act_h, act_a] = Forward(W, b, train_data(t,:));

%         t
%         output
%         [valY, predY] = max(output)
%         actualY = find(train_label(t,:))
        [grad_W, grad_b] = Backward(W, b, train_data(t,:), train_label(t, :), act_h, act_a);
        
        [W, b] = UpdateParameters(W, b, grad_W, grad_b, learning_rate);
%         [accuracy, loss] = ComputeAccuracyAndLoss(W, b, train_data, train_label);
%         sprintf('\tAfter iteration-%d: Accuracy = %f, Loss = %f\n', t, accuracy, loss)
    end
end

end



% function [outputs] = Classify(W, b, data)
% function [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a)
% function [W, b] = UpdateParameters(W, b, grad_W, grad_b, learning_rate)
% function [output, act_h, act_a] = Forward(W, b, X)
% function [W, b] = InitializeNetwork(layers)
% function [accuracy, loss] = ComputeAccuracyAndLoss(W, b, data, labels)

% for i = 1:size(train_data,1)
% 
% 
%     if mod(i, 100) == 0
%         fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
%         fprintf('Done %.2f %%', i/size(train_data,1)*100)
%     end
% end
% fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
