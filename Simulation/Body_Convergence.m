Repetitions = 100;

%Record the found eigenvectors
L = min(L1, L2);
T = 5:5:300;

%Record of the left and right singular vectors
u = cell(1, Repetitions);
v = cell(1, Repetitions);
ufull = cell(1, Repetitions);
vfull = cell(1, Repetitions);

%Record the Frobenius errors
err.u = zeros(Repetitions, length(T));
err.v = zeros(Repetitions, length(T));

%Record the found labels
l1est = cell(1, Repetitions);
l2est = cell(1, Repetitions);
l1full = zeros(n, Repetitions);
l2full = zeros(m, Repetitions);

%Record community estimation errors
err.l1 = zeros(Repetitions, length(T)+1);
err.l2 = zeros(Repetitions, length(T)+1);

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
    ufull{rep} = Xi_full;
    vfull{rep} = Psi_full;
    
    l1full(:,rep) = kmeans(Xi_full, L1, 'MaxIter', 100, 'Replicates', 50);
    l2full(:,rep) = kmeans(Psi_full, L2, 'MaxIter', 100, 'Replicates', 50);

    err.l1(rep, length(T)+1) = Community_Error(l1, l1full(:,rep));
    err.l2(rep, length(T)+1) = Community_Error(l2, l2full(:,rep));


    % New BiFLICKER algorithm inflation as 10
    result = BiFLICKER_conv(B, L1, L2, T, [], 10);
    u{rep} = result.Xi;
    v{rep} = result.Psi;
    l1est{rep} = result.l1;
    l2est{rep} = result.l2;

    err.u(rep, :) = arrayfun(@(b) norm(b.Xi * b.Xi' - Xi_full*Xi_full', 'fro'), result, 'UniformOutput', true);
    err.v(rep, :) = arrayfun(@(b) norm(b.Psi * b.Psi' - Psi_full*Psi_full', 'fro'), result, 'UniformOutput', true);

    err.l1(rep, 1:length(T)) = arrayfun(@(b) Community_Error(l1, b.l1), result, 'UniformOutput', true); 
    err.l2(rep, 1:length(T)) = arrayfun(@(b) Community_Error(l2, b.l2), result, 'UniformOutput', true); 

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