
function [H, world_pts_x, world_pts_y] = part1(image_list, match_list, K, pointsw)
clear all
close all

% imagepath='/home/dinis/Desktop/Cadeiras/PIV/Dinis/Project/datasets/4/';
imagepath='/home/dinis/Desktop/Cadeiras/PIV/Dinis/Project/examples/sintetico/';
files = dir(imagepath);
imagelist = files(3:end);

for i=1:length(imagelist)
    img_list{i} = imread(fullfile(imagepath, imagelist(i).name));
end

% Initialize all the transforms to the identity matrix. 
numImages = numel(img_list);
tforms(numImages, numImages) = projective2d(eye(3));

% Initialize variable to hold image sizes.
imageSize = zeros(numImages,2);

% Iterate over remaining image pairs
for i = 1:numImages
    % Read the first image from the image set.
    I = imread(fullfile(imagepath, imagelist(i).name));

    % Initialize features for I(1)
    grayImage = rgb2gray(I);
    points = detectSURFFeatures(grayImage);
    [features, points] = extractFeatures(grayImage, points);
    
    % Save image size.
    imageSize(i,:) = size(grayImage);
    
    % Store points and features for I(i).
    pointsPrevious = points;
    featuresPrevious = features;
    
    for j = 1:numImages
        % If i = j, it's equal to the identity matrix
        if i ~= j
            % Read I(j).
            I = imread(fullfile(imagepath, imagelist(j).name));

            % Convert image to grayscale.
            grayImage = rgb2gray(I);    

            % Save image size.
            imageSize(i,:) = size(grayImage);

            % Detect and extract SURF/SIFT features for I(j).
            points = detectSURFFeatures(grayImage);    
            [features, points] = extractFeatures(grayImage, points);

            % Find correspondences between I(j) and I(i).
            indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);

            matchedPoints = points(indexPairs(:,1), :);
            matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);        

            % Estimate the transformation between I(n) and I(n-1).
            % This is MSANSAC instead of RANSAC -> 'same' thing,
            % faster/easier no extra code required, only a line of code
            tforms(i,j) = estimateGeometricTransform(matchedPointsPrev, matchedPoints,...
                'projective', 'Confidence', 99.9, 'MaxNumTrials', 100);
            
            % Do referencing, (doesn't apply in this case) T(j) * T(j-1) * ... * T(1)
%             tforms(j).T = tforms(j).T * tforms(j-1).T; 
        end
    end
end

% This doesn't give the best solution, only the Homographies between all
% images, there are images that cannot be mapped into another, so this can
% bring explosive results ->  Present best homography for plotting
tform_best(numImages) = projective2d(eye(3));
for i = 2:length(tforms)           
   tform_best(i) = tforms(i,i-1);
   tform_best(i).T = tform_best(i).T * tform_best(i-1).T; 
end

% Get limits of each transformed image, save those limits
for i = 1:length(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tform_best(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

% Get max size index
maxImageSize = max(imageSize);

% Find the minimum and maximum output limits 
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama = zeros([height width 3], 'like', I);


blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');  

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the panorama.
for i = 1:numImages
    
    I = imread(fullfile(imagepath, imagelist(i).name));   
   
    % Transform I into the panorama.
    warpedImage = imwarp(I, tform_best(i), 'OutputView', panoramaView);
                  
    % Generate a binary mask.    
    mask = imwarp(true(size(I,1),size(I,2)), tform_best(i), 'OutputView', panoramaView);
    
    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);
end

figure
imshow(panorama)
grid on

% Save the Homographies into the wanted variable
for i=1:length(tforms)
   for j=1:length(tforms)
       H{i,j} = tforms(i,j).T;
   end
end

% Get the world points
% obtained from [x y z]*H
figure
for i=1:length(img_list)
    im1 = img_list{i};

    corners=[1          1           1; 
            size(im1,2) 1           1;
            size(im1,2) size(im1,1) 1;
            1           size(im1,1) 1;
            1           1           1]; % this last line is only to complete the square :)

    pts12=corners*tform_best(i).T;

    world_pts_x(i,:) = (pts12(:,1)./pts12(:,3))';
    world_pts_y(i,:) = (pts12(:,2)./pts12(:,3))';

    plot(world_pts_x(i,:), world_pts_y(i,:));
    hold on;

end

% the code above is saving the last value of the world, that is the same of
% the first, we remove that now and save what we want
world_pts_x(:,end) = [];
world_pts_y(:,end) = [];
axis ij

end