% Simulation Settings
m = 500;  % Number of nodes in U and V
L1 = 2; L2 = 3;     % Number of communities for U and V
nseq = [2000, 4000, 8000]; % Number of nodes in U
Kseq = [5, 10, 20]; % Number of servers
L = min(L1, L2);

% density = 0;
density = -0.2:0.1:0.3;

for n = nseq
    for K = Kseq
        for dd = density
            % Connection probability matrix
            G = zeros(L1, L2);
            G(1,:) = [1, 0.5, 0.3];
            G(2,:) = [0.3, 0.5, 1];
            G = (G + dd) * 10 / m;

            %Run the repetitions
            rng("twister");
            Body_Simulation;
            filename = sprintf('n_%d_K_%d_density_%.1f.mat', n, K, dd);
            save(filename)
            disp(filename)
        end
    end
end