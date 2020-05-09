function [score] = homework1( image_name, K )
% This is a simple example to help you test your implementation using
% an image. Please feel free to use the attached images, or your own
% images.
%
% An example of running this script is
%   homework1('taj.jpg', 4);
%
% You are not supposed to edit this file. Your job is implementing k-means
% and k-medoids in the other files, mykmeans.m and mykmedoids.m. Have fun!

	score = 0;
	
	image = imread(image_name);
%     image = imread('football.bmp');
	rows = size(image, 1);
	cols = size(image, 2);
	pixels = zeros(rows*cols, 3);

	for i=1:rows
		for j=1:cols
			pixels((j-1)*rows+i, 1:3) = image(i,j,:);
		end
	end

	tic;
	[class1, centroid1] = mykmeans(pixels, K);
	kmeansTime = toc
	
	tic;
	[class2, centroid2] = mykmedoids(pixels, K);
	kmedoidsTime = toc

	converted_image1 = zeros(rows, cols, 3);
	converted_image2 = zeros(rows, cols, 3);
	for i=1:rows
		for j=1:cols
			converted_image1(i, j, 1:3) = centroid1(class1((j-1)*rows+i),:);
			converted_image2(i, j, 1:3) = centroid2(class2((j-1)*rows+i),:);
		end
	end

	converted_image1 = converted_image1 / 255;
	converted_image2 = converted_image2 / 255;
	
	subplot(1,3,1);
	h = imshow(image_name, 'InitialMag',100, 'Border','tight');
	title('Original')
	
	subplot(1,3,2);
	h = imshow(converted_image1, 'InitialMag',100, 'Border','tight');
	title('K-means')
	
	subplot(1,3,3);
	h = imshow(converted_image2, 'InitialMag',100, 'Border','tight');
	title('K-medoids')
end
