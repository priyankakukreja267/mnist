load('nist26_model.mat', 'W', 'b')

finalWtsW = W;
finalWtsB = b;

initWtsW = W;
collage = zeros(32, 32, 1, 800);
for i = 1:800
    m = reshape(initWtsW{1}(:, i), [32,32]);
    collage(:,:,1,i) = m - min(min(m));
end
h = montage(collage, 'Size', [20 40])

% finalWtsW = W;
collage = zeros(32, 32, 1, 800);
for i = 1:800
    m = reshape(finalWtsW{1}(:, i), [32,32]);
    collage(:,:,1,i) = m - min(min(m));
end
h = montage(collage, 'Size', [20 40])



%%%%%% CONFUSION MATRIX
% Find the output prediction for each of the data point: as a 1X26 vector
predY = zeros(nData, 1);
actualY = zeros(nData, 1);
for i = 1:nData
    [output, act_h, act_a] = Forward(W, b, data(i, :)');
    [temp, predY(i, 1)] = max(output);
    actualY(i, 1) = find(labels(i, :));
end

% Visualize as a matrix
C = confusionmat(actualY, predY);
sizeC = size(C)
imshow(C)
% upscale the image
% paste in report

for i = 1:36
    temp(i,i) = 0;
end
