% Simulation Settings
m = 500;  % Number of nodes in U and V
L1 = 2; L2 = 3;     % Number of communities for U and V
nseq = [2000, 4000, 8000]; % Number of nodes in U
Kseq = [5, 10, 20]; % Number of servers
L = min(L1, L2);

for n = nseq
    for K = Kseq
            % Connection probability matrix
            G = zeros(L1, L2);
            G(1,:) = [1, 0.5, 0.3];
            G(2,:) = [0.3, 0.5, 1];
            dd = -0.1;
            G = (G + dd) * 10 / m;

            %Run the repetitions
            rng("twister");
            Body_Convergence;
            filename = sprintf('Exp2_n_%d_K_%d.mat', n, K);
            save(filename)
            disp(filename)
    end
end