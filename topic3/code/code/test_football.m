clc; clear all; close all;

%% spectral clustering
n = 321;
k = 13;

% load graph
a = dlmread('play_graph.txt');

A = sparse(a(:, 1), a(:, 2), ones(size(a, 1), 1), n, n);
A = (A + A') / 2;

D = diag(1 ./ sqrt(sum(A, 1)));
L = D * A * D;

[X, V] = eigs(L, k);
X = bsxfun(@rdivide, X, sqrt(sum(X .* X, 2)));

% scatter(X(:, 1), X(:, 2))

% k-means
c_idx = kmeans(X, k, 'replicates', 20);


%% show clusters
fid = fopen('inverse_teams.txt');

idx2names = cell(1, 1);

tline = fgetl(fid);
count = 1;
while ischar(tline)
    sep = strfind(tline, '	');
    idx = str2double(tline(1:sep-1));
    team_name = tline(sep+1:end);
    idx2names{count} = team_name;
    count = count + 1;
    tline = fgetl(fid);
end

fclose(fid);

% c_idx, k
for i = 1:k
    fprintf('Cluster %i\n***************\n', i);
    idx = c_idx == i;
    content = sprintf('%s\n', idx2names{idx});
    fprintf('%s\n', content);
end

%% output files
node_file = 'nodes.csv';
edge_file = 'edges.csv';

fid = fopen(edge_file, 'w');
fprintf(fid, 'Source\tTarget\n');
for i = 1:size(a, 1)
    fprintf(fid, '%i\t%i\n', a(i, 1), a(i, 2));
end
fclose(fid);

fid = fopen(node_file, 'w');
fprintf(fid, 'Id\tLabel\tColor\n');
for i = 1:length(idx2names)
    fprintf(fid, '%i\t%s\t%i\n', i, idx2names{i}, c_idx(i));
end
fclose(fid);