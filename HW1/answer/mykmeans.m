function [ class, centroid ] = mykmeans( pixels, K )
%
% Your goal of this assignment is implementing your own K-means.
%
% Input:
%     pixels: data set. Each row contains one data point. For image
%     dataset, it contains 3 columns, each column corresponding to Red,
%     Green, and Blue component.
%
%     K: the number of desired clusters. Too high value of K may result in
%     empty cluster error. Then, you need to reduce it.
%
% Output:
%     class: the class assignment of each data point in pixels. The
%     assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
%     of class should be either 1, 2, 3, 4, or 5. The output should be a
%     column vector with size(pixels, 1) elements.
%
%     centroid: the location of K centroids in your result. With images,
%     each centroid corresponds to the representative color of each
%     cluster. The output should be a matrix with K rows and
%     3 columns. The range of values should be [0, 255].
%     
%
% You may run the following line, then you can see what should be done.
% For submission, you need to code your own implementation without using
% the kmeans matlab function directly. That is, you need to comment it out.
for k=1:K
c1(k,:)=quantile(pixels,(0.5/K+(k-1)/K));
end
c0=c1-10;

i=1;
while (norm(c1 - c0, 'fro') > 1e-6)
    fprintf(1, '--iteration %d \n', i);
    
    % record previous c; 
    c0 = c1; 
    
    % assign data points to current cluster; 
    for j = 1:length(pixels) % loop through data points; 
        tmp_distance = zeros(1, K); 
        for k = 1:K % through centers; 
            tmp_distance(k) = sum((pixels(j,:) - c1(k,:)).^2); % norm(x(:,j) - c(:,k)); 
        end
        [~,K_index] = min(tmp_distance); % ~ ignores the first argument; 
        P(:, j) = zeros(K, 1); 
        P(K_index, j) = 1; 
    end
        
    % adjust the cluster centers according to current assignment;     
%     cstr = {'r.', 'b.', 'g.', 'r+', 'b+', 'g+'};
    obj = 0;
    obj2=0;
    for k = 1:K
        idx = find(P(k, :)>0); 
        no_of_points = length(idx);  
        if (no_of_points == 0) 
            % a center has never been assigned a data point; 
            % re-initialize the center; 
            c1(k,:) = quantile(pixels,0.5);  
        else
            % equivalent to sum(x(:,idx), 2) ./ no_of_points;            
            c1(k,:) = P(k,:) * pixels ./ no_of_points;         
        end
        obj = obj + sum(sum((pixels(idx,:) - repmat(c0(k,:),no_of_points,1)).^2));
        obj2 = obj2 + sum(sum((pixels(idx,:) - repmat(c1(k,:),no_of_points,1)).^2));
    end
    
    if obj2-obj>0
        c1=c0;
    end
    i = i + 1;     
end   
P1=(sum(P.*(1:K)'))';
    
    class=P1;
    centroid=c1;
end

