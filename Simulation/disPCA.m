%https://arxiv.org/pdf/1408.5823
function result = disPCA(B, L1, L2)
%Distributed PCA when there are m servers
% B: the data matrices on each server
% L1: number of clusters for U
% L2: number of clusters for V
% Output:
% Psi: m times L matrix, each column being a right singular vector 
% Xi: n times L matrix, each column being a left singular vector
% l2: the clustering labels of V
% l1: the clustering labels of U



L = min(L1, L2);
K = length(B);

% Dimensions
n = cellfun(@(x) size(x,1), B); % vector of n_k
indn = arrayfun(@(i) i * ones(1, n(i)), 1:numel(n), 'UniformOutput', false);
indn = [indn{:}];
P = vertcat(B{:});
Psi = distributed_pca(P, [L*10, L], sum(n), indn); %<--- we use the code from original paper
Psi = Psi{2};

u = cellfun(@(b) b * Psi, B, 'UniformOutput', false);
uall = vertcat(u{:});
Xi = select_top_K_eigvecs(uall * uall', L);

% Clustering
l1 = kmeans(Xi, L1, 'MaxIter', 100, 'Replicates', 50);
l2 = kmeans(Psi, L2, 'MaxIter', 100, 'Replicates', 50);

% Output
result = struct('Xi', Xi, 'Psi', Psi, 'l1', l1, 'l2', l2);
end

% Helper function
function V = select_top_K_eigvecs(M, K)
    try
        [V, ~] = eigs(M, K);
    catch
        [V_full, D] = eig(M);
        [~, idx] = sort(diag(D), 'descend');
        V = V_full(:, idx(1:K));
    end
end
