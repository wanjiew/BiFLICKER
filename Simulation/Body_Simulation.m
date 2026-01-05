Repetitions = 100;

%Record the found eigenvectors
L = min(L1, L2);
fieldNames = {'New', 'NewIn5', 'DistPCA', 'disPCA', 'FastPCA', 'Full', 'True'};
matrixList = repmat({zeros(n, Repetitions*L)}, size(fieldNames));
u = cell2struct(matrixList, fieldNames, 2);
matrixList = repmat({zeros(m, Repetitions*L)}, size(fieldNames));
v = cell2struct(matrixList, fieldNames, 2);

%Record the Frobenius errors
err.u = zeros(Repetitions, size(fieldNames, 2) - 1);
err.v = zeros(Repetitions, size(fieldNames, 2) - 1);

%Record the found labels
fieldNames = {'New', 'NewIn5', 'DistPCA', 'disPCA', 'FastPCA', 'Full', 'True'};
matrixList = repmat({zeros(n, Repetitions)}, size(fieldNames));
l1est = cell2struct(matrixList, fieldNames, 2);
matrixList = repmat({zeros(m, Repetitions)}, size(fieldNames));
l2est = cell2struct(matrixList, fieldNames, 2);

%Record community estimation errors
err.l1 = zeros(Repetitions, size(fieldNames, 2) - 1);
err.l2 = zeros(Repetitions, size(fieldNames, 2) - 1);

for rep = 1:Repetitions
    % Labels for U
    l1 = randi(L1, n, 1);  % sample(1:L1, n, replace = TRUE)
    Z1 = zeros(n, L1);
    for kk = 1:L1
        Z1(l1 == kk, kk) = 1;
    end

    % Labels for V
    l2 = randi(L2, m, 1);
    Z2 = zeros(m, L2);
    for kk = 1:L2
        Z2(l2 == kk, kk) = 1;
    end

    % Mean matrix
    Omega = Z1 * G * Z2';
    [Xi_true, ~, Psi_true] = svds(Omega, L);
    u.True(:,(1:L)+(rep-1)*L) = Xi_true;
    v.True(:,(1:L)+(rep-1)*L) = Psi_true;
    l1est.True(:, rep) = l1;
    l2est.True(:, rep) = l2;
    
    % Generate connection matrix A
    A = rand(n, m);
    A = Omega - A;
    A = double(A > 0);  % 0-1 matrix based on probability threshold

    % % Show average connection density
    % density = sum(A, 'all') / (n * m);
    % fprintf('Density: %.4f\n', density);

    % Divide data into K local servers (equal-sized)
    ksub = n / K;
    B = cell(1, K);
    for kk = 1:K
        indk = (1:ksub) + (kk - 1) * ksub;
        B{kk} = A(indk, :);
    end

    % SVD with full information
    [Xi_full, ~, Psi_full] = svds(A, L);
    u.Full(:,(1:L)+(rep-1)*L) = Xi_full;
    v.Full(:,(1:L)+(rep-1)*L) = Psi_full;
    
    l1full = kmeans(Xi_full, L1, 'MaxIter', 100, 'Replicates', 50);
    l2full = kmeans(Psi_full, L2, 'MaxIter', 100, 'Replicates', 50);
    l1est.Full(:, rep) = l1full;
    l2est.Full(:, rep) = l2full;

    err.l1(rep, 6) = Community_Error(l1, l1full);
    err.l2(rep, 6) = Community_Error(l2, l2full);


    % New BiFLICKER algorithm without inflation
    T = 500*(K/10);
    result_new = BiFLICKER(B, L1, L2, T, [], 1);
    u.New(:,(1:size(result_new.Xi, 2))+(rep-1)*L) = result_new.Xi;
    v.New(:,(1:size(result_new.Psi, 2))+(rep-1)*L) = result_new.Psi;
    l1est.New(:, rep) = result_new.l1;
    l2est.New(:, rep) = result_new.l2;

    err.u(rep, 1) = norm(result_new.Xi*result_new.Xi' - Xi_full*Xi_full', 'fro');
    err.v(rep, 1) = norm(result_new.Psi*result_new.Psi' - Psi_full*Psi_full', 'fro');

    err.l1(rep, 1) = Community_Error(l1, result_new.l1);
    err.l2(rep, 1) = Community_Error(l2, result_new.l2);

    % New BiFLICKER algorithm inflation as 10
    T = 500*(K/10);
    result_newinf = BiFLICKER(B, L1, L2, T, [], 10);
    u.NewIn5(:,(1:size(result_newinf.Xi, 2))+(rep-1)*L) = result_newinf.Xi;
    v.NewIn5(:,(1:size(result_newinf.Psi, 2))+(rep-1)*L) = result_newinf.Psi;
    l1est.NewIn5(:, rep) = result_newinf.l1;
    l2est.NewIn5(:, rep) = result_newinf.l2;

    err.u(rep, 2) = norm(result_newinf.Xi*result_newinf.Xi' - Xi_full*Xi_full', 'fro');
    err.v(rep, 2) = norm(result_newinf.Psi*result_newinf.Psi' - Psi_full*Psi_full', 'fro');

    err.l1(rep, 2) = Community_Error(l1, result_newinf.l1);
    err.l2(rep, 2) = Community_Error(l2, result_newinf.l2);

    % DistPCA: construct local U together
    result_distPCA = DistPCA(B, L1, L2);
    u.DistPCA(:,(1:L)+(rep-1)*L) = result_distPCA.Xi;
    v.DistPCA(:,(1:L)+(rep-1)*L) = result_distPCA.Psi;
    l1est.DistPCA(:, rep) = result_distPCA.l1;
    l2est.DistPCA(:, rep) = result_distPCA.l2;

    err.u(rep, 3) = norm(result_distPCA.Xi*result_distPCA.Xi' - Xi_full*Xi_full', 'fro');
    err.v(rep, 3) = norm(result_distPCA.Psi*result_distPCA.Psi' - Psi_full*Psi_full', 'fro');

    err.l1(rep, 3) = Community_Error(l1, result_distPCA.l1);
    err.l2(rep, 3) = Community_Error(l2, result_distPCA.l2);

    % disPCA: put Sigma*U together
    result_disPCA = disPCA(B, L1, L2);
    u.disPCA(:,(1:L)+(rep-1)*L) = result_disPCA.Xi;
    v.disPCA(:,(1:L)+(rep-1)*L) = result_disPCA.Psi;
    l1est.disPCA(:, rep) = result_disPCA.l1;
    l2est.disPCA(:, rep) = result_disPCA.l2;

    err.u(rep, 4) = norm(result_disPCA.Xi*result_disPCA.Xi' - Xi_full*Xi_full', 'fro');
    err.v(rep, 4) = norm(result_disPCA.Psi*result_disPCA.Psi' - Psi_full*Psi_full', 'fro');

    err.l1(rep, 4) = Community_Error(l1, result_disPCA.l1);
    err.l2(rep, 4) = Community_Error(l2, result_disPCA.l2);

    % FastPCA: online updates
    alpha = 1/sqrt(sum(sum(A)));
    result_fastPCA = FastPCA(B, L1, L2, 1000, alpha);
    u.FastPCA(:,(1:L)+(rep-1)*L) = result_fastPCA.Xi;
    v.FastPCA(:,(1:L)+(rep-1)*L) = result_fastPCA.Psi;
    l1est.FastPCA(:, rep) = result_fastPCA.l1;
    l2est.FastPCA(:, rep) = result_fastPCA.l2;

    err.u(rep, 5) = norm(result_fastPCA.Xi*result_fastPCA.Xi' - Xi_full*Xi_full', 'fro');
    err.v(rep, 5) = norm(result_fastPCA.Psi*result_fastPCA.Psi' - Psi_full*Psi_full', 'fro');

    err.l1(rep, 5) = Community_Error(l1, result_fastPCA.l1);
    err.l2(rep, 5) = Community_Error(l2, result_fastPCA.l2);

    if mod(rep, 20) == 0
    disp(rep)
    end
end

disp(mean(err.u))
disp(mean(err.v))
disp(mean(err.l1))
disp(mean(err.l2))

function err = Community_Error(l1, l2)

if length(l1) ~= length(l2)
    error('Clustering Error Calculation: length cannot match!')
end
n = length(l1);
confmat = crosstab(l1, l2);
pairs = matchpairs(confmat, 0, 'max');
err = 1 - sum(confmat(sub2ind(size(confmat), pairs(:,1), pairs(:,2))))/n;

end