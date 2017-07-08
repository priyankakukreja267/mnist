function [lines, bw] = findLetters(im)

% [lines, BW] = findLetters(im) processes the input RGB image and returns a cell
% array 'lines' of located characters in the image, as well as a binary
% representation of the input image. The cell array 'lines' should contain one
% matrix entry for each line of text that appears in the image. Each matrix entry
% should have size Lx4, where L represents the number of letters in that line.
% Each row of the matrix should contain 4 numbers [x1, y1, x2, y2] representing
% the top-left and bottom-right position of each box. The boxes in one line should
% be sorted by x1 value.

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
    % Find minimum y1   x(x(:,1) < 3, :) = []    a = x(x(:,1) < 3, :)
    minY1 = min(boxes(:,2));
%     maxBoxHt = max(boxes(:,4) - boxes(:,2));
    selectedBoxes = boxes( (boxes(:,2) >= minY1) & (boxes(:,2) < minY1 + maxBoxHt), :);
    boxes( (boxes(:,2) >= minY1) & (boxes(:,2) <= minY1 + maxBoxHt), :) = [];
    lines{i} = sortrows(selectedBoxes, 1);
    i = i+1;
end

bw = ~bw;
end
