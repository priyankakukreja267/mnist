ims = cell(1,4);
ims{1} = '../images/02_letters.jpg'; 
ims{2} = '../images/01_list.jpg';
ims{3} = '../images/03_haiku.jpg'; 
ims{4} = '../images/04_deep.jpg';

for i = 1:length(ims)
    [lines, bw] = findOCRLetters(ims{i});
    imshow(ims{i});
    hold on
    for j = 1:length(lines)
        boxes = lines{j};
        boxes = [boxes(:,1), boxes(:,2), boxes(:,3)-boxes(:,1), boxes(:,4)-boxes(:,2)];
        for k = 1:size(boxes, 1)
            rectangle('Position', boxes(k, :),'EdgeColor', 'r', 'LineWidth', 1);
        end
    end
end

function [lines, bw] = findOCRLetters(im)
img = imread(im);

% Convert to grayscale
if (size(img, 3) == 3)
    img = rgb2gray(img);
end

% Threshold it
level = graythresh(img);
bwImg = im2bw(img, level);

% Invert it
bwImg = ~bwImg;

% Fill holes in image
ifill = imfill(bwImg, 'holes');
ifill = bwdist(ifill) <= 5;

% IMPORTANT
% fill just the holes smaller than a certain size (say 100 pixels)
% filled = imfill(bwImg, 'holes');
% holes = filled & ~bwImg;
% bigholes = bwareaopen(holes, 30);
% smallholes = holes & ~bigholes;
% ifill = bwImg | smallholes;

imshow(ifill);

% Get the number of connected components in the binary image
[iLabel num] = bwlabel(ifill);
iprops = regionprops(iLabel);
ibox = [iprops.BoundingBox];
ibox = reshape(ibox, [4 num]);

imshow(img);
hold on;
for i = 1:num
    rectangle('Position', ibox(:,i), 'EdgeColor', 'r', 'LineWidth', 1);
end
end
