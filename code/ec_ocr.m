function ec_ocr()
    ims = cell(1,4);
    ims{1} = '../images/02_letters.jpg'; 
    ims{2} = '../images/01_list.jpg';
    ims{3} = '../images/03_haiku.jpg'; 
    ims{4} = '../images/04_deep.jpg';

    for i = 1:length(ims)
        [text] = extractImageText(ims{i});
        sprintf(text)
    end
end

function [text] = extractImageText(fname)
load('nist36_model.mat', 'W', 'b')

% Get bounding boxes of letters in the image
[lines, bw] = findLetters(fname);

text = [];
th = 35; % threshold for inserting a whitespace
minLetterArea = 140; % threshold for considering a given rectangle as letter, otherwise it is a noise point

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
        sprintf('predY = %d, letter = %s', predY, letter);
        
        text = [text letter];

        if c < length(lines{l})
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

function [lines, bw] = findLetters(im)

% Read in the image
img = imread(im);

% Convert to grayscale
if (size(img, 3) == 3)
    img = rgb2gray(img);
end

% Threshold it
level = graythresh(img);
bw = im2bw(img, level);

% Invert it
bw = ~bw;

% Fill holes in image
% bw = imfill(bw, 'holes');
% bw = bwdist(bw) <= 1;

% Put a bounding box on connected components
[cc] = bwconncomp(bw);
props = regionprops(cc); %, 'BoundingBox');
boxes = [props.BoundingBox];
boxes = reshape(boxes, 4, [])';

% Sort the boxes, first line wise, then column wise
boxes = [boxes(:,1), boxes(:,2), boxes(:,1)+boxes(:,3), boxes(:,2)+boxes(:,4)];

% iBox = (x1, y1, x2, y2)
i = 1;
lines = cell(1,1);
maxBoxHt = max(boxes(:,4) - boxes(:,2));
while size(boxes, 1) > 0
    minY1 = min(boxes(:,2));
    selectedBoxes = boxes( (boxes(:,2) >= minY1) & (boxes(:,2) < minY1 + maxBoxHt), :);
    boxes( (boxes(:,2) >= minY1) & (boxes(:,2) <= minY1 + maxBoxHt), :) = [];
    lines{i} = sortrows(selectedBoxes, 1);
    i = i+1;
end

bw = ~bw;
end
