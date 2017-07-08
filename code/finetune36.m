% Accuracy should start at 45%, and end at 90% for train and 80% for valid
num_epoch = 20; % DEFAULT

classes = 36;
layers = [32*32, 800, classes];
learning_rate = 0.01;

load('../data/nist36_train.mat', 'train_data', 'train_labels')
load('../data/nist36_test.mat', 'test_data', 'test_labels')
load('../data/nist36_valid.mat', 'valid_data', 'valid_labels')

% To display the kth picture in the dataset
% imshow(reshape(train_data(k, :), [32, 32]))
% [W, b] = InitializeNetwork(layers);

load('../data/nist26_model_60iters.mat', 'W');
load('../data/nist26_model_60iters.mat', 'b');


W{1} = W{1}';
W{2} = horzcat(W{2}', rand(800, 10));
b{1} = b{1}';
b{2} = horzcat(b{2}', rand(1, 10));

W = W';
b = b';

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

    sprintf('Epoch %d (train, valid, test) - accuracy: %.5f, %.5f, %.5f \t loss: %.5f, %.5f, %.5f \n', j, train_acc(j), test_acc(j), valid_acc(j), train_loss(j), test_loss(j), valid_loss(j))
end
save('nist36_model.mat', 'W', 'b')

epochs = [1:num_epoch];
plot(epochs, train_acc, '-ro', epochs, valid_acc, '-.b');
legend('Train accuracy', 'Validation Accuracy');

plot(epochs, train_loss, '-ro', epochs, valid_loss, '-.b');
legend('Train loss', 'Validation loss');

plot(epochs, test_loss, 'r');
plot(epochs, test_acc, 'r');

save('nist36_model.mat', 'W', 'b')
