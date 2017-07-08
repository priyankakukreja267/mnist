ims = cell(1,4);
ims{1} = '../images/02_letters.jpg'; 
ims{2} = '../images/01_list.jpg';
ims{3} = '../images/03_haiku.jpg'; 
ims{4} = '../images/04_deep.jpg';

for i = 1:length(ims)
    [lines, bw] = findLetters(ims{i});
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
