clear; 

%%
% generate the data; 

% generate random angles between [0, 2pi]
n = 800; 
rangle = 2 * pi * rand(n, 1); 

% generate random radius for the first circle; 
e = 0.2;
rr = 1.9 + e * rand(n, 1); 

rx = rr .* sin(rangle); 
ry = rr .* cos(rangle); 

x = rx; 
y = ry; 

% generate random radius for the second circle; 
rr2 = 1.2 + e * rand(n, 1); 

rx2 = rr2 .* sin(rangle); 
ry2 = rr2 .* cos(rangle); 

x = [x; rx2]; 
y = [y; ry2]; 

rx3 = 1.4 + (1.9 - 1.4) * rand(10, 1); 
ry3 = e * rand(10, 1); 

% % uncomment this to comment the two rings; 
x = [x; rx3]; 
y = [y; ry3]; 

%%
data = [x, y]; 

figure; 
plot(x, y, 'k.'); 
hold on; 
title('original data'); 

keyboard; 

% run kmeans on the original coordinates; 
K = 2; 
idx = kmeans([x, y], K, 'Replicates', 10); 

figure; 
plot(x(idx==1), y(idx==1), 'r.'); 
hold on; 
plot(x(idx==2), y(idx==2), 'b.'); 
title('K-means'); 

keyboard;

%%
distmat = squareform(pdist(data)).^2; 

% A(A<0.99) = 0; 
A = double(distmat < 0.1);  

% figure; 
% spy(A); 
% hold on; 
% title('adjacency matrix'); 

%%

D = diag(sum(A,2)); 
L = D - A; 

[V, S] = eig(L); 
K = 2; 
idx = kmeans(V(:,1:2), K, 'Replicates', 10); 

figure; 
plot(x(idx==1), y(idx==1), 'r.'); 
hold on; 
plot(x(idx==2), y(idx==2), 'b.'); 
title('Spectral Clustering'); 








