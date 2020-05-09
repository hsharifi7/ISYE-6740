% k-means clustering; 

clear; 
% close all; clc; 

% change the value to 0 to run just kmeans without comparing to brute force
% method; 
iscomparebruteforce = 0; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% k-means algorithm; 
% greed algorithm trying to minimize the objective function; 
%
dim = 2; % dimension of the data points; 
% change to larger number of data points and cno = 3 after comparison with brute force
% method
if (iscomparebruteforce == 1)
    m = 6; % number of data points; 
    % fix the seed of the random number generator, so each time generate
    % the same random points; 
    randn('seed', 1); 
    x = [...
        randn(dim, m) + repmat([4,1]', 1, m), ...
        randn(dim, m) + repmat([4,4]', 1, m), ...
        randn(dim, m) + repmat([1,2]', 1, m)]; 
    
    % number of clusters; 
    cno = 2; 
else
    m = 100; % number of data points; 
    % fix the seed of the random number generator, so each time generate
    % the same random points; 
    randn('seed', 1);     
    x = [...
        randn(dim, m) + repmat([4,1]', 1, m), ...
        randn(dim, m) + repmat([4,4]', 1, m), ...
        randn(dim, m) + repmat([1,2]', 1, m)]; 
    
    % number of clusters; 
    cno = 6;     
end
m = size(x, 2); 

%%
% randomly initialize the cluster center; since the seed for function rand
% is not fixed, every time it is a different matrix; 
rand('seed', sum(clock));
c = 6*rand(dim, cno); 
c_old = c + 10; 

figure; 
i = 1; 
% check whether the cluster centers still change; 
tic
while (norm(c - c_old, 'fro') > 1e-6)
    fprintf(1, '--iteration %d \n', i);
    
    % record previous c; 
    c_old = c; 
    
    % assign data points to current cluster; 
    for j = 1:m % loop through data points; 
        tmp_distance = zeros(1, cno); 
        for k = 1:cno % through centers; 
            tmp_distance(k) = sum((x(:,j) - c(:,k)).^2); % norm(x(:,j) - c(:,k)); 
        end
        [~,k_star] = min(tmp_distance); % ~ ignores the first argument; 
        P(j, :) = zeros(1, cno); 
        P(j, k_star) = 1; 
    end
        
    % adjust the cluster centers according to current assignment;     
    cstr = {'r.', 'b.', 'g.', 'r+', 'b+', 'g+'};
    obj = 0; 
    for k = 1:cno
        idx = find(P(:,k)>0); 
        nopoints = length(idx);  
        if (nopoints == 0) 
            % a cener has never been assigned a data point; 
            % re-initialize the center; 
            c(:,k) = rand(dim,1);  
        else
            % equivalent to sum(x(:,idx), 2) ./ nopoints;            
            c(:,k) = x * P(:,k) ./ nopoints;         
            plot(x(1,idx),x(2,idx), cstr{k});             
            hold on;
        end
        obj = obj + sum(sum((x(:,idx) - repmat(c(:,k), 1, nopoints)).^2)); 
    end
    
    plot(c(1,:),c(2,:), 'ro'); 
    hold off; 
    drawnow;
    
    M(i) = getframe(gcf);
    pause(1)
    
    i = i + 1;     
end   
toc 
% kmeans will be much faster than brute force enumeration, even after we
% have the additional pause 1 second and visualization in the codes; 

% run it several times and you will see that objective function is
% different; 
obj

% play movie; 
% movie(M, 10);
% save movie to avi files; 
% movie2avi(M, 'M.avi'); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% enumerating all possibilities is computational intensive; 
% we can only work with a small number data points; 
%
if (iscomparebruteforce == 1)
    
    % enumerate all possibility; 
    best_obj = 1e7; 
    result = []; 
    tic
    for i = 1:(m-1)
        fprintf(1, '-- case %d\n', i); 
        partition1 = nchoosek((1:m), i); 
        for j = 1:size(partition1, 1)
            obj = 0; 
            group1_idx = partition1(j,:); 
            center1 = sum(x(:,group1_idx), 2) ./ length(group1_idx); 

            obj = obj + sum(sum((x(:,group1_idx) - repmat(center1, 1, length(group1_idx))).^2)); 
            % equivalent: 
    %         for l = 1:length(group1_idx)
    %             obj = obj + sum((x(:,group1_idx(l)) - center1).^2); 
    %         end

            group2_idx = setdiff((1:m), group1_idx); 
            center2 = sum(x(:,group2_idx), 2) ./ length(group2_idx); 
            for l = 1:length(group2_idx)
                obj = obj + sum((x(:,group2_idx(l)) - center2).^2); 
            end    

            if (obj < best_obj)
                result = group1_idx; 
                best_obj = obj; 
            end
        end
    end
    toc
    % look at the objective function; the objective function of brute force
    % enumerate is smaller! kmeans only find a local minimum in this case; 
    best_obj

end

%%
ra = randn(m, 1); 
idx1 = find(ra > 0); 
idx2 = setdiff((1:m)', idx1); 

center1 = mean(x(:,idx1),2); 
center2 = mean(x(:,idx2),2); 

newobj = sum(sum((x(:,idx1) - repmat(center1, 1, length(idx1))).^2)) + ...
    sum(sum((x(:,idx2) - repmat(center2, 1, length(idx2))).^2)); 

newobj


