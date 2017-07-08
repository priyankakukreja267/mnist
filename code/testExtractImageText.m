ims = cell(1,4);
ims{1} = '../images/02_letters.jpg'; 
ims{2} = '../images/01_list.jpg';
ims{3} = '../images/03_haiku.jpg'; 
ims{4} = '../images/04_deep.jpg';

for i = 1:length(ims)
    [text] = extractImageText(ims{i});
    sprintf(text)
end
