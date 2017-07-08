% Randomly generated data
train_data = randi(10, [1000 10]); 
train_label = randi(3, [1000 1]);
nData = size(train_data, 1);
nInputs = size(train_data, 2) % 4
nCategories = length(unique(train_label)) % 5
layers = [nInputs, 7, nCategories] % [10 7 5] Gives the number of nodes in each layer

m = zeros(nData, nCategories);
for i = 1:nData
    m(i, train_label(i)) = 1;
end
train_label = m;

[W, b] = InitializeNetwork(layers)

% train_data: D X N
% train_label: D X 1

nEpoch = 30;
for i = 1:nEpoch
    fprintf('Starting Epoch - %d\n', i);
    learning_rate = 0.01;
    [W, b] = Train(W, b, train_data, train_label, learning_rate);
    [acc, loss] = ComputeAccuracyAndLoss(W, b, train_data, train_label);
    sprintf('At Epoch - %d: Accuracy = %f, Loss = %f\n', i, acc, loss);
end
