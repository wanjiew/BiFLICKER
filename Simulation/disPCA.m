function result = disPCA(B, L1, L2)
%DISPCA Distributed-PCA comparison method for bipartite clustering.
%
% This is an independent implementation of Algorithm disPCA described in:
%   M.-F. Balcan, V. Kanchanapally, Y. Liang, and D. P. Woodruff,
%   "Improved Distributed Principal Component Analysis," NeurIPS 2014.
%   https://proceedings.neurips.cc/paper_files/paper/2014/hash/
%   e968f1646c1c6c35422b64c0934772a4-Abstract.html
%
% Each server sends a rank-LOCAL_RANK summary of its centered local matrix.
% The center stacks those summaries and extracts the leading L right singular
% vectors. The simulation uses LOCAL_RANK = min(10*L, matrix dimensions),
% matching the oversampling setting used in this project.

L = min(L1, L2);
K = numel(B);

if K == 0 || any(cellfun(@isempty, B))
    error('disPCA:InvalidInput', 'B must contain one nonempty matrix per server.');
end

item_counts = cellfun(@(x) size(x, 2), B);
if any(item_counts ~= item_counts(1))
    error('disPCA:DimensionMismatch', ...
        'All local matrices must have the same number of item columns.');
end

% Compute the global column mean from additive local summaries.
total_users = sum(cellfun(@(x) size(x, 1), B));
column_sums = cellfun(@(x) sum(x, 1), B, 'UniformOutput', false);
global_mean = sum(vertcat(column_sums{:}), 1) / total_users;

local_rank = min(10 * L, min(total_users, item_counts(1)));
local_summaries = cell(K, 1);

for k = 1:K
    centered = bsxfun(@minus, double(B{k}), global_mean);
    [~, S_k, V_k] = svd(centered, 'econ');
    rank_k = min(local_rank, min(size(S_k)));

    % Sigma_k V_k' summarizes the leading local components.
    local_summaries{k} = S_k(1:rank_k, 1:rank_k) * ...
        V_k(:, 1:rank_k)';
end

summary_matrix = vertcat(local_summaries{:});
[~, ~, V_global] = svd(summary_matrix, 'econ');
Psi = V_global(:, 1:L);

% Recover a compatible orthonormal left embedding from local products.
left_products = cellfun(@(x) double(x) * Psi, B, 'UniformOutput', false);
[Xi, ~, ~] = svd(vertcat(left_products{:}), 'econ');
Xi = Xi(:, 1:L);

l1 = kmeans(Xi, L1, 'MaxIter', 100, 'Replicates', 50);
l2 = kmeans(Psi, L2, 'MaxIter', 100, 'Replicates', 50);

result = struct('Xi', Xi, 'Psi', Psi, 'l1', l1, 'l2', l2);
end
