function [ class, centroid ] = mykmedoids( pixels, K)
%
% Your goal of this assignment is implementing your own K-medoids.
% Please refer to the instructions carefully, and we encourage you to
% consult with other resources about this algorithm on the web.
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

c1=datasample(pixels,K);
c0=c1-10;

z=1;
while (norm(c1 - c0, 'fro') > 1e-6)
    fprintf(1, '--iteration %d \n', z);
    
    % record previous c; 
    c0 = c1; 
    
    % assign data points to current cluster; 
    for j = 1:length(pixels) % loop through data points; 
        tmp_distance = zeros(1, K); 
        for k = 1:K % through centers; 
%             tmp_distance(k) = sum((pixels(j,:) - c1(k,:)).^2); % norm(x(:,j) - c(:,k)); 
            tmp_distance(k) = max(abs(pixels(j,:) - c1(k,:))); % inf distance 
        end
        [~,K_index] = min(tmp_distance); % ~ ignores the first argument; 
        P(:, j) = zeros(K, 1); 
        P(K_index, j) = 1; 
    end
        
    % adjust the cluster centers according to current assignment; 

    obj=0;
    obj2=0;
    for k = 1:K
        idx = find(P(k, :)>0); 
        no_of_points = length(idx);             
        centroid(k,:) = P(k,:) * pixels ./ no_of_points; 
        tmp_distance2 = zeros(1, K); 
        for l=1:length(idx)
%             tmp_distance2(l) = sum((pixels(idx(l),:) - centroid(k,:)).^2);
            tmp_distance2(l) = max(abs(pixels(idx(l),:) - centroid(k,:)));
        end
        [~,cntr_pt_index] = min(tmp_distance2); % ~ ignores the first argument;
        cnew(k,:)=pixels(idx(cntr_pt_index),:);
%         obj = obj + sum(sum((pixels(idx,:) - repmat(c1(k,:),no_of_points,1)).^2));
%         obj2 = obj2 + sum(sum((pixels(idx,:) - repmat(cnew(k,:),no_of_points,1)).^2));
        obj = obj + sum(max(abs(pixels(idx,:) - repmat(c1(k,:),no_of_points,1))));
        obj2 = obj2 + sum(max(abs(pixels(idx,:) - repmat(cnew(k,:),no_of_points,1))));
        clear tmp_distance2;
    end

    
    if obj2-obj<0
        c1=cnew;
    end
    
    z = z + 1;     
end   
P1=(sum(P.*(1:K)'))';
    
    class=P1;
    centroid=c1;
end