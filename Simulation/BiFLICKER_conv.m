function result = BiFLICKER_conv(B, L1, L2, T, eta, inflation)
% B: cell array of local datasets (each B{k} is n_k x m)
% L1: number of communities in U
% L2: number of communities in V
% z1, z2: ground truth (for tracking error)
% T: all the possible numbers of iterations
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

% SVD on first block
S1 = svds(B{1}, L);
lambda = sqrt(sum(n)/n(1)) * S1;

if lambda(end) > sqrt(lambda(1)) && inflation > 1
    add = linspace(lambda(end), sqrt(lambda(1))/L, inflation + 1);
    lambda = [lambda; add(2:end)'];
end

% Stepsize
if isempty(eta)
    eta = 1 ./ max((lambda.^4), (lambda(1)^2 - lambda.^2).^2);
end
if length(eta) < L
    eta = repmat(eta, 1, L); eta = eta(1:L);
end

% Initialization
Lnew = length(lambda);
Xi = randn(sum(n), Lnew);
Xi = Xi ./ vecnorm(Xi);

Psi = randn(m, Lnew);
Psi = Psi ./ vecnorm(Psi);

Xiall = cell(1, length(T));
Psiall = cell(1, length(T));
l1all = cell(1, length(T));
l2all = cell(1, length(T));
% Iterations
for iter = 1:max(T)
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

    if ~isempty(find(T == iter, 1))
        Xicurr = Xi; Psicurr = Psi; lambdacurr = lambda;

        Dxi = Xicurr' * Xicurr; Dxi = Dxi - eye(length(lambdacurr));
        while sum(sum(abs(Dxi) > 0.5)) > 0 
            idx = find(abs(Dxi) > 0.5, 1);
            [idrow, idcol] = ind2sub(size(Dxi), idx);
            idremove = max(idrow, idcol);
            lambdacurr(idremove) = [];
            Xicurr(:, idremove) = [];
            Psicurr(:, idremove) = [];
            Dxi = Xicurr' * Xicurr; Dxi = Dxi - eye(length(lambdacurr));
        end
        

        Dpsi = Psicurr' * Psicurr; Dpsi = Dpsi - eye(length(lambdacurr));
        while sum(sum(abs(Dpsi) > 0.5)) > 0
            idx = find(abs(Dpsi) > 0.5, 1);
            [idrow, idcol] = ind2sub(size(Dpsi), idx);
            idremove = max(idrow, idcol);
            lambdacurr(idremove) = [];
            Xicurr(:, idremove) = [];
            Psicurr(:, idremove) = [];
            Dpsi = Psicurr' * Psicurr; Dpsi = Dpsi - eye(length(lambdacurr));
        end
        if length(lambdacurr) > L
            Xicurr = Xicurr(:, 1:L);
            Psicurr = Psicurr(:, 1:L);
        end
        idx = find(T == iter);
        Xiall{idx} = Xicurr;
        Psiall{idx} = Psicurr;

        % Clustering
        l1all{idx} = kmeans(Xicurr, L1, 'MaxIter', 100, 'Replicates', 50);
        l2all{idx} = kmeans(Psicurr, L2, 'MaxIter', 100, 'Replicates', 50);

    end
end


% Output
result = struct('Xi', Xiall, 'Psi', Psiall, 'l1', l1all, 'l2', l2all);
end
