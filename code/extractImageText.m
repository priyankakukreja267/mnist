function [text] = extractImageText(fname)
% [text] = extractImageText(fname) loads the image specified by the path 'fname'
% and returns the text contained in the image as a string.

load('nist36_model.mat', 'W', 'b')

% Get bounding boxes of letters in the image
[lines, bw] = findLetters(fname);

text = [];
th = 35; % threshold for inserting a whitespace
minLetterArea = 100; % threshold for considering a given rectangle as letter, otherwise it is a noise point

for l = 1:size(lines, 2)
    for c = 1:size(lines{l}, 1)
        maxX = size(bw, 2);
        maxY = size(bw, 1);

        x1 = lines{l}(c, 1);
        y1 = lines{l}(c, 2);
        x2 = lines{l}(c, 3);
        y2 = lines{l}(c, 4);
        if x1 < 1 | y1 < 1 | x2 < 1 | y2 < 1 | x1 > maxX | y1 > maxY | x2 > maxX | y2 > maxY
            continue;
        end
        img = bw(y1:y2, x1:x2);
        imgSize = size(img);

        if (imgSize(1) * imgSize(2) < minLetterArea)
            continue;
        end
        
        img = padarray(img, [30 30], 1, 'both');
%         imshow(img, 'InitialMagnification','fit');
        img = imresize(img, [32, 32]);
%         imshow(img, 'InitialMagnification','fit');
        
        % Get rid of holes created after resizing
        invImg = ~img;
        invImg = imfill(invImg, 'holes');
        invImg = bwdist(invImg) <= 1;
        img = ~invImg;

        
%         imshow(img, 'InitialMagnification','fit');

        imgVector = reshape(img, 1, 1024);

        % Use the found W and b to get the prediction
        [output, act_h, act_a] = Forward(W, b, imgVector');

        % Convert prediction to letter
        [tempY, predY] = max(output);
        if predY <= 26
            letter = char(predY + 64);
        else
            letter = char(predY - 27 + 48);
        end
%         sprintf('predY = %d, letter = %s', predY, letter);
        
        % Insert character into the array
        text = [text letter];

        % Check if space is needed? insert : don't insert
        if c < length(lines{l})
            % dist = x1 of next box - x2 of this box
%             lines{l}(c:c+1, :)
            dist = lines{l}(c+1, 1) - lines{l}(c, 3);
            if dist > th
                text = [text, ' '];
            end
        end
    end
    if l < length(lines)
        text = [text, '\n'];
    end
end
end
