function result = BiFLICKER(B, L1, L2, T, eta, inflation)
% B: cell array of local datasets (each B{k} is n_k x m)
% L1: number of communities in U
% L2: number of communities in V
% z1, z2: ground truth (for tracking error)
% T: number of iterations
% eta: optional step size (can be scalar or vector of length L)
% inflation: an integer that controls how many more eigenvectors to
% calculate

if nargin < 7 || isempty(eta)
    eta = []; % will be set later
end
if isempty(inflation)
    inflation = []; % will be set later
end

L = min(L1, L2);
K = length(B);

% Dimensions
n = cellfun(@(x) size(x,1), B); % vector of n_k
m = size(B{1},2);

% Extra singular vectors to calculate
if isempty(inflation)
    inflation = round(sum(n)/n(1));
end

% SVD on the largest block
[~, idx] = max(n);
[~, S1, Psi] = svds(B{idx}, L);
S1 = diag(S1);
lambda = sqrt(sum(n)/n(idx)) * S1;

if lambda(end) > sqrt(lambda(1)) && inflation > 1
    add = linspace(lambda(1), sqrt(lambda(1))/L, inflation + 1);
    lambda = [lambda; add(2:end)'];
end

% Stepsize
if isempty(eta)
    eta = 1 ./ max((lambda.^4), (4*lambda(1)^2 - lambda.^2).^2);
end
if length(eta) < L
    eta = repmat(eta, 1, L); eta = eta(1:L);
end

% Initialization
Lnew = length(lambda);

u = cellfun(@(b) b * Psi, B, 'UniformOutput', false);
Xi = vertcat(u{:});

if Lnew - L > 0
    Psi1 = randn(m, Lnew - L);
    Psi1 = Psi1 ./ vecnorm(Psi1);
    Psi = [Psi, Psi1];

    Xi1 = randn(sum(n), Lnew - L);
    Xi = [Xi, Xi1];
    Xi = Xi ./ vecnorm(Xi);
end


% Iterations
for iter = 1:T
    % Left singular vector update
    xall = 0;
    offset = 0;
    for kk = 1:K
        xall = xall + Xi(offset+1:offset+n(kk), :)' * B{kk};
        offset = offset + n(kk);
    end
    u = cellfun(@(b) b * xall', B, 'UniformOutput', false);
    uall = vertcat(u{:});

    xall2 = 0;
    offset = 0;
    for kk = 1:K
        xall2 = xall2 + uall(offset+1:offset+n(kk), :)' * B{kk};
        offset = offset + n(kk);
    end
    w = cellfun(@(b) b * xall2', B, 'UniformOutput', false);
    wall = vertcat(w{:});

    Xi = Xi - (Xi * diag(lambda.^4) - 2 * uall * diag(lambda.^2) + wall) * diag(eta);
    Xi = Xi ./ vecnorm(Xi);

    % Right singular vector update
    v = cellfun(@(b) b' * b * Psi, B, 'UniformOutput', false);
    vall = sum(cat(3, v{:}), 3);
    w2 = cellfun(@(b) b' * b * vall, B, 'UniformOutput', false);
    wall = sum(cat(3, w2{:}), 3);

    Psi = Psi - (Psi * diag(lambda.^4) - 2 * vall * diag(lambda.^2) + wall) * diag(eta);
    Psi = Psi ./ vecnorm(Psi);

    tol = (1/T)*(1e-5);
    if any(abs(iter/T - (0.1:0.1:1)) < tol)
        fprintf('%.0f%% progress finished\n', iter/T*100);
    end
end


Dxi = Xi' * Xi; Dxi = Dxi - eye(length(lambda));
while sum(sum(abs(Dxi) > 0.5)) > 0 
    idx = find(abs(Dxi) > 0.5, 1);
    [idrow, idcol] = ind2sub(size(Dxi), idx);
    idremove = max(idrow, idcol);
    lambda(idremove) = [];
    Xi(:, idremove) = [];
    Psi(:, idremove) = [];
    Dxi = Xi' * Xi; Dxi = Dxi - eye(length(lambda));
end

Dpsi = Psi' * Psi; Dpsi = Dpsi - eye(length(lambda));
while sum(sum(abs(Dpsi) > 0.5)) > 0
    idx = find(abs(Dpsi) > 0.5, 1);
    [idrow, idcol] = ind2sub(size(Dpsi), idx);
    idremove = max(idrow, idcol);
    lambda(idremove) = [];
    Xi(:, idremove) = [];
    Psi(:, idremove) = [];
    Dpsi = Psi' * Psi; Dpsi = Dpsi - eye(length(lambda));
end
if length(lambda) > L
    Xi = Xi(:, 1:L);
    Psi = Psi(:, 1:L);
end

% Clustering
l1 = kmeans(Xi, L1, 'MaxIter', 100, 'Replicates', 50);
l2 = kmeans(Psi, L2, 'MaxIter', 100, 'Replicates', 50);

% Output
result = struct('Xi', Xi, 'Psi', Psi, 'l1', l1, 'l2', l2);
end
