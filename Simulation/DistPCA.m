%https://pmc.ncbi.nlm.nih.gov/articles/PMC6836292/pdf/nihms-1003769.pdf
function result = DistPCA(B, L1, L2)
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

eigVecs = cellfun(@(b) select_top_K_eigvecs(b' * b / size(b, 1), L), B, 'UniformOutput', false);
Vsigma = 0;
for kk = 1:K
    Vsigma = Vsigma + eigVecs{kk}*eigVecs{kk}';
end
Vsigma = Vsigma/K;
Psi = select_top_K_eigvecs(Vsigma, L);

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
