function [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a)
% [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a) computes the gradient
% updates to the deep network parameters and returns them in cell arrays
% 'grad_W' and 'grad_b'. This function takes as input:
%   - 'W' and 'b' the network parameters
%   - 'X' and 'Y' the single input data sample and ground truth output vector,
%     of sizes Nx1 and Cx1 respectively
%   - 'act_h' and 'act_a' the network layer pre and post activations when forward
%     forward propogating the input smaple 'X'

X = X';
Y = Y';

nOutputNodes = size(W{2}, 2);
nHiddenNodes = size(W{2}, 1);
nInputNodes = size(W{1}, 1);

nLayers = size(W, 1) + 1;
grad_W = cell(nLayers - 1, 1);
grad_b = cell(nLayers - 1, 1);

f = act_h{2};
fDotY = sum(f .* Y); % fDotY is the dot product of f vector and y vector
M1 = -Y / fDotY;
M2 = diag(f) - (f' * f);
M12 = repmat((M1 * M2)', 1, nHiddenNodes);
M3 = repmat(act_h{1}, nOutputNodes, 1);
grad_W{2} = (M12 .* M3)';
grad_b{2} = M1 * M2 * ones(nOutputNodes, nOutputNodes);

M4 = W{2}';
M5 = diag(act_h{1} .* (1 - act_h{1}) );
M6 = repmat(X, nHiddenNodes, 1);
M1245 = M1 * M2 * M4 * M5;
grad_W{1} = (repmat(M1245', 1, nInputNodes) .* M6)';
grad_b{1} = M1245 * ones(nHiddenNodes, nHiddenNodes);
end
