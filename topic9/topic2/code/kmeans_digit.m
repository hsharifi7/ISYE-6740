clear; 
close all; 

load usps_all; 

pixelno = size(data, 1); 
digitno = size(data, 2); 
classno = size(data, 3); 

H = 16; 
W = 16; 
figure; 
show_image([data(:,:,1), data(:,:,10)]', H, W); 
title('digit 1 and 0'); 
% figure; 
% show_image(data(:,:,10)', H, W); 
% title('digit 2'); 

x0 = reshape(data(:,:,[1,10]), [pixelno, digitno*2]); 
x = double(x0); 
y = [ones(1,digitno), 2*ones(1,digitno)]; 

% number of data points to work with; 
m = size(x, 2); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% k-means algorithm; 
% greed algorithm trying to minimize the objective function; 
% A highly matricized version of kmeans.

%% run kmeans; 
cno = 2; 

% randomly initialize centroids with data points; 
c = x(:,randsample(size(x,2),cno));

iterno = 100; 
for iter = 1:iterno
  fprintf('--iteration %d\n', iter); 
  
  % norm2 of the centroids; 
  c2 = sum(c.^2, 1);  
  
  % for each data point, computer max_j -2 * x' * c_j + c_j^2; 
  tmpdiff = bsxfun(@minus, 2*x'*c, c2); 
  [val, labels] = max(tmpdiff, [], 2); 
  
  % update data assignment matrix; 
  P = sparse(1:m, labels, 1, m, cno, m); 
  count = sum(P, 1); 
   
  % recompute centroids; 
  c = bsxfun(@rdivide, x*P, count); 
end

for i = 1:cno
  figure; 
  show_image(x0(:,P(:,i)==1)', H, W); 
  title(['cluster ', int2str(i)]); 
end
