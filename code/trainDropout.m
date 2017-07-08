% Accuracy should start at 45%, and end at 90% for train and 80% for valid
function trainDropout()
display('Running dropout of 0.5')
num_epoch = 30; % DEFAULT

classes = 26;
layers = [32*32, 400, classes];
learning_rate = 0.01;

load('../data/nist26_train.mat', 'train_data', 'train_labels')
load('../data/nist26_test.mat', 'test_data', 'test_labels')
load('../data/nist26_valid.mat', 'valid_data', 'valid_labels')

% To display the kth picture in the dataset
% imshow(reshape(train_data(k, :), [32, 32]))
[W, b] = InitializeNetwork(layers);
initWtsW = W;
initWtsB = b;

train_acc = zeros(num_epoch, 1);
train_loss = zeros(num_epoch, 1);
valid_acc = zeros(num_epoch, 1);
valid_loss = zeros(num_epoch, 1);
test_acc = zeros(num_epoch, 1);
test_loss = zeros(num_epoch, 1);

nData = size(train_data, 1);
nInputs = size(train_data, 2);

for j = 1:num_epoch
    dataAns = [train_data, train_labels];
    data = dataAns(randperm(length(train_data)), :);
    labels = data(:, nInputs + 1 : end);
    data = data(:, 1 : nInputs);
    [W, b] = Train(W, b, data, labels, learning_rate);
    [train_acc(j), train_loss(j)] = ComputeAccuracyAndLoss(W, b, data, labels);
    [valid_acc(j), valid_loss(j)] = ComputeAccuracyAndLoss(W, b, valid_data, valid_labels);
    [test_acc(j), test_loss(j)] = ComputeAccuracyAndLoss(W, b, test_data, test_labels);

    sprintf('Epoch %d (train, valid, test) - accuracy: %.5f, %.5f, %.5f \t loss: %.5f, %.5f, %.5f \n', ...
        j, train_acc(j), test_acc(j), valid_acc(j), train_loss(j), test_loss(j), valid_loss(j))
end

epochs = [1:num_epoch];
plot(epochs, train_acc, '-ro', epochs, valid_acc, '-.b');
legend('Train accuracy', 'Validation Accuracy');

plot(epochs, train_loss, '-ro', epochs, valid_loss, '-.b');
legend('Train loss', 'Validation loss');

plot(epochs, test_loss, 'r');
plot(epochs, test_acc, 'r');

end

function [W, b] = Train(W, b, train_data, train_label, learning_rate)
% This loop template simply prints the loop status in a non-verbose way.
% Feel free to use it or discard it

% train_data: D X N
% train_label: D X 1

nData = size(train_data, 1);
finalGradW = cell(size(W, 1), 1);
finalGradB = cell(size(b, 1), 1);
batchSize = 12;

% STOCHASTIC GRADIENT UPDATE with MINI BATCH of size 12.
for t = 1:batchSize:nData

    %  Drop some columns from W{1}, and same elements from b{1}
    %  Drop same rows from W{2}, and same elements from b{2}

    selectedNodes = randperm(400, 200);
    W{1}(:, selectedNodes) = 0.0;
    b{1}(selectedNodes) = 0.0;
    W{2}(selectedNodes, :) = 0.0;

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

function [output, act_h, act_a] = Forward(W, b, X)
X = X';

nLayers = size(W, 1) + 1;
nHidden = nLayers - 2;
nInputs = size(W{1}, 1);
% pre-act: LC
% post-act:sigmoid of LC

act_a = cell(nLayers-1, 1);
act_h = cell(nLayers-1, 1);

post_act = X; % 1XN

for i = 1:nLayers-2 % Do only for non-output layers, starting from 2nd layer
    wts = W{i};  % wts is a #Li X #L(i+1)

    % Compute LC of inputs
    pre_act = b{i} + (post_act * wts); % returns a 1 X #L(i+1) data points: one for each of the node in Layer (i+1)
    
    % Compute Sigmoid of LC
    post_act = sigmoid(pre_act);
    
    % Append to pre-act    
    act_a{i} = pre_act;
    
    % Append to post-act
    act_h{i} = post_act;
end

% Find output using softmax
pre_act = b{nLayers - 1} + (post_act * W{nLayers - 1});
post_act = exp(pre_act);
post_act = post_act / sum(post_act);

act_a{nLayers - 1} = pre_act;
act_h{nLayers - 1} = post_act;
output = act_h{nLayers - 1}';
end
 
function out = sigmoid(in)
    out = 1 ./ (1.0 + exp(-in));
end

function [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a)
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
