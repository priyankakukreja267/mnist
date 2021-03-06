function [W, b] = UpdateParameters(W, b, grad_W, grad_b, learning_rate)
% [W, b] = UpdateParameters(W, b, grad_W, grad_b, learning_rate) computes and returns the
% new network parameters 'W' and 'b' with respect to the old parameters, the
% gradient updates 'grad_W' and 'grad_b', and the learning rate.

lenW = size(W, 1);
for i = 1:lenW
    W{i} = W{i} - (learning_rate * grad_W{i});
    b{i} = b{i} - (learning_rate * grad_b{i});
end

% W{1} = W{1} - (learning_rate * grad_W{1});
% W{2} = W{2} - (learning_rate * grad_W{2});
% b{1} = b{1} - (learning_rate * grad_b{1});
% b{2} = b{2} - (learning_rate * grad_b{2});
end
