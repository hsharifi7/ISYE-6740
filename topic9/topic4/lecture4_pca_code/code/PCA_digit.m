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

Anew = x'; 
stdA = std(Anew, 1, 1); 
Anew = Anew * diag(1./stdA); 
Anew = Anew'; 

% PCA
mu=sum(Anew,2)./m;
xc = bsxfun(@minus, Anew, mu); 

C = xc * xc' ./ m; 

k = 2; 
[W, S] = eigs(C, k); 
diag(S);
%single(W' * W)

dim1 = W(:,1)' * xc ./ sqrt(S(1,1));
dim2 = W(:,2)' * xc ./ sqrt(S(2,2));

figure; 
hold on; 
plot(dim1(y==1), dim2(y==1), 'r.'); 
plot(dim1(y==2), dim2(y==2), 'b.'); 
hold off; 
