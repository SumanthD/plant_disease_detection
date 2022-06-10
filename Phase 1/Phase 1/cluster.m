close all
clear all
clc

[filename,pathname] = uigetfile({'*.*';'*.bmp';'*.tif';'*.gif';'*.png'},'Pick a Disease Affected Leaf');
J = imread([pathname,filename]);
figure, imshow(J);
title('Disease Affected Leaf');
%J = histeq(I);
cform = makecform('srgb2lab');
lab_he = applycform(J,cform);
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);
nColors = 5;
[cluster_idx cluster_center] = kmeans(ab,nColors,'Distance','sqEuclidean', 'Replicates',5);
pixel_labels = reshape(cluster_idx,nrows,ncols);
segmented_images = cell(1,5);

rgb_label = repmat(pixel_labels,[1,1,3]);

for k = 1:nColors
colors = J;
colors(rgb_label ~= k) = 0;
segmented_images{k} = colors;
end

figure, subplot();
imshow(segmented_images{1});
title('Cluster 1'); 
figure,subplot();
imshow(segmented_images{2});
title('Cluster 2');
figure,subplot();
imshow(segmented_images{3});
title('Cluster 3');
figure,subplot();
imshow(segmented_images{4});
title('Cluster 4');
figure,subplot();
imshow(segmented_images{5});
title('Cluster 5');

x = inputdlg('Enter the cluster no. containing the disease affected leaf part only:');
i = str2double(x);

seg_img = segmented_images{i};
if ndims(seg_img) == 5
    img = rgb2gray(seg_img);
end