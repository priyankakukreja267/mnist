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


train_acc = zeros(num_epoch, 1);
train_loss = zeros(num_epoch, 1);
valid_acc = zeros(num_epoch, 1);
valid_loss = zeros(num_epoch, 1);

%%%%%% REMOVE LATER
% train_data = train_data(1:50, :);
% train_labels = train_labels(1:50, :);

for j = 1:num_epoch
    display('time for training')
    tic;
    [W, b] = Train(W, b, train_data, train_labels, learning_rate);
    toc;
    
    display('time for computeAcc')
    tic;
    [train_acc(j), train_loss(j)] = ComputeAccuracyAndLoss(W, b, train_data, train_labels);
    toc;

    %     [valid_acc(j), valid_loss(j)] = ComputeAccuracyAndLoss(W, b, valid_data, valid_labels);

    sprintf('Epoch %d - accuracy: %.5f, %.5f \t loss: %.5f, %.5f \n', j, train_acc(j), valid_acc(j), train_loss(j), valid_loss(j))
end

epochs = [1:num_epoch];
plot(epochs, train_acc, '-ro', epochs, valid_acc, '-.b');
legend('Train accuracy', 'Validation Accuracy');

plot(epochs, train_loss, '-ro', epochs, valid_loss, '-.b');
legend('Train loss', 'Validation loss');

save('nist26_model.mat', 'W', 'b')
