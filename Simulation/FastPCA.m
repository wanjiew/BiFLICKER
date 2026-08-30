%https://arxiv.org/pdf/2108.12373
function result = FastPCA(B, L1, L2, T, alpha)
%Distributed PCA when there are m servers. We consider a star network. 
% B: the data matrices on each server
% L1: number of clusters for U
% L2: number of clusters for V
% T: number of iterations
% alpha: step size
% Output:
% Psi: m times L matrix, each column being a right singular vector 
% Xi: n times L matrix, each column being a left singular vector
% l2: the clustering labels of V
% l1: the clustering labels of U
L = min(L1, L2);
K = length(B);

% Dimensions
% n = cellfun(@(x) size(x,1), B); % vector of n_k
m = size(B{1}, 2);

% Covariance matrices
C = cellfun(@(b) b'*b/size(b,1), B, 'UniformOutput', false);

% Initialisation
Psi = randn(m, m); [Psi, ~, ~] = svds(Psi, L);
S = cellfun(@(c) create_h(Psi, c), C, 'UniformOutput', false);
S_center = 0;
Psi_node = cell(1, K);

% Updates
for t = 1:T
    Psi_update = 0;
    S_update = 0;
    for kk = 1:K
        if t == 1
            Hcurr = S{kk};
            Psi_node{kk} = Psi;
        else
            Hcurr = Hnext;
        end
        Psi_node{kk} = Psi_node{kk}/2 + K*Psi_node{kk}/2/(K+1) + Psi/2/(K+1) + alpha*S{kk};
        Psi_update = Psi_update + Psi_node{kk}/(K+1);

        Hnext = create_h(Psi_node{kk}, C{kk});
        S{kk} = S{kk}/2 + K*S{kk}/2/(K+1) + S_center/2/(K+1) + Hnext - Hcurr;
        S_update = S_update + S{kk}/(K+1);
    end


    Psi = Psi/2 + Psi/2/(K+1) + Psi_update/2 +  alpha*S_center;
    S_center = S_center/2 + S_center/2/(K+1) + S_update/2;
end

Psi = Psi./vecnorm(Psi);

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
function H = create_h(Xii, Ci)
    H = Xii*0;
    xiinorm = vecnorm(Xii); 
    xiinorm(xiinorm <= 1e-9) = 1;
    XCX = Xii' * Ci * Xii ./ (xiinorm.^2);
    L = size(Xii, 2);
    H(:, 1) = Ci * Xii(:, 1) - XCX(1,1) * Xii(:, 1);
    for ll = 2:L
        H(:, ll) = Ci * Xii(:, ll) - XCX(ll, ll) * Xii(:, ll) - Xii(:, 1:(ll - 1))*XCX(ll, 1:(ll - 1))';
    end
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
