%% Make 2 full figures for BiFLICKER simulations on Convergence
%  - Rows: n = 2000, 4000, 8000
%  - Cols: K = 5, 10, 20
%  - x-axis in each panel: T
%  - 2 methods: BiFLICKER (varying T) and Full (oracle)
%    displayed as: "BiFLICKER", "Oracle"

clear; clc;

% ----- Simulation settings -----
nseq = [2000, 4000, 8000];
Kseq = [5, 10, 20];

% Load one file to get T grid (assume common across files)
probe = sprintf('Exp2/Exp2_n_%d_K_%d.mat', nseq(1), Kseq(1));
S0 = load(probe, 'T');
T = S0.T(:)';                    % 1 x nT
nT = numel(T);

nN = numel(nseq);
nK = numel(Kseq);

% Colors for K lines
KColors = containers.Map('KeyType','double','ValueType','any');
KColors(5)  = [1 0 0];          % red
KColors(10) = [0 1 1];          % cyan
KColors(20) = [0 0 1];          % blue

% ----- Containers: mean/sd over reps -----
% size: (iN, iK, iT)
mean_l1 = nan(nN, nK, nT);  sd_l1 = nan(nN, nK, nT);
mean_l2 = nan(nN, nK, nT);  sd_l2 = nan(nN, nK, nT);
mean_u  = nan(nN, nK, nT);  sd_u  = nan(nN, nK, nT);
mean_v  = nan(nN, nK, nT);  sd_v  = nan(nN, nK, nT);

baseline_l1 = nan(nN, nK);
baseline_l2 = nan(nN, nK);

%% Load all files and compute means for each error type

for iN = 1:nN
    n = nseq(iN);

    for iK = 1:nK
        K = Kseq(iK);

        filename = sprintf('Exp2/Exp2_n_%d_K_%d.mat', n, K);
        fprintf('Loading %s\n', filename);

        S = load(filename, 'err', 'T', 'Repetitions');
        % S.err.u, S.err.v: (R x nT)
        % S.err.l1, S.err.l2: (R x (nT+1))  last column = full-info baseline

        % Defensive: allow T mismatch; use the file's T length
        Tloc = S.T(:)'; 
        nTloc = numel(Tloc);
        if nTloc ~= nT
            warning('T length mismatch in %s (probe %d vs file %d). Using file T.', filename, nT, nTloc);
        end
        nUse = min(nT, nTloc);

        % BiFLICKER curves over T
        U  = S.err.u(:, 1:nUse);              % R x nUse
        V  = S.err.v(:, 1:nUse);
        L1 = S.err.l1(:,1:nUse);              % R x nUse  (last col is oracle, not used here)
        L2 = S.err.l2(:,1:nUse);

        mean_u(iN,iK,1:nUse)  = mean(U,  1);
        sd_u(  iN,iK,1:nUse)  = std(U,   0, 1);

        mean_v(iN,iK,1:nUse)  = mean(V,  1);
        sd_v(  iN,iK,1:nUse)  = std(V,   0, 1);

        mean_l1(iN,iK,1:nUse) = mean(L1, 1);
        sd_l1(  iN,iK,1:nUse) = std(L1,  0, 1);

        mean_l2(iN,iK,1:nUse) = mean(L2, 1);
        sd_l2(  iN,iK,1:nUse) = std(L2,  0, 1);

       baseline_l1(iN,iK) = mean(S.err.l1(:,end));
        baseline_l2(iN,iK) = mean(S.err.l2(:,end));

    end
end


%% Helper function to make a 1×3 figure for one metric
%  data: (iN, iK, iT, method)
function make_n_panels(meanData, sdData, baselineData, T, nseq, Kseq, KColors, yLabel, figTitle, fileName)
    nN = numel(nseq);
    nK = numel(Kseq);

    figure('Units','normalized','Position',[0.05 0.10 0.88 0.55],'Color','w');
    t = tiledlayout(1, nN, 'TileSpacing','compact', 'Padding','compact');
    title(t, figTitle, 'Interpreter','latex', 'FontSize', 36);

    % Determine y-limits robustly
    ymax = max(meanData(:) + sdData(:));
    if ~isfinite(ymax); ymax = 1; end
    ymax = ymax + 0.05;

    axLast = [];

    for iN = 1:nN
        ax = nexttile(t);
        if iN == nN
            axLast = ax; % legend anchor
        end

        hold(ax, 'on');

        for iK = 1:nK
            K = Kseq(iK);
            c = KColors(K);

            mu = squeeze(meanData(iN, iK, :))';   % 1 x nT
            sd = squeeze(sdData(iN, iK, :))';     % 1 x nT

            % Shaded band: mu ± sd
            x = T;
            y1 = mu - sd;
            y2 = mu + sd;

            fill(ax, [x, fliplr(x)], [y1, fliplr(y2)], c, ...
                 'FaceAlpha', 0.18, 'EdgeColor', 'none', ...
                 'HandleVisibility','off');

            % Mean line
            plot(ax, x, mu, '-o', 'LineWidth', 1.8, 'Color', c, ...
                 'MarkerSize', 5, ...
                 'DisplayName', sprintf('$K=%d$', K));
        end

        if ~isempty(baselineData)
            y0 = baselineData(iN, :);
            y0 = mean(y0(~isnan(y0)));   % safety
            yline(ax, y0, 'k-', ...
                'LineWidth', 3.0, ...
                'DisplayName', 'Oracle');
        end

        hold(ax, 'off');

        grid(ax, 'on'); box(ax, 'on');
        ax.LineWidth = 1.2;

        ax.FontSize = 18;
        title(ax, sprintf('\\fontsize{28}{28}\\selectfont $n = %d$', nseq(iN)), ...
              'Interpreter','latex');

        xlabel(ax, '\fontsize{28}{28}\selectfont Iterations $T$', 'Interpreter','latex');

        if iN == 1
            ylabel(ax, yLabel, 'FontSize', 28, 'Interpreter','latex');
        else
            ax.YTickLabel = [];
        end

        xlim(ax, [min(T), max(T)]);
        ylim(ax, [0, max(ymax, 1e-8)]);    
        legend(ax, ...
            'Location', 'northeast', ...
            'Interpreter', 'latex', ...
            'FontSize', 20, ...
            'Box', 'off');

    end


    % IMPORTANT: to avoid blank PDFs with transparency, export as IMAGE-based PDF
    if nargin >= 10 && ~isempty(fileName)
        set(gcf, 'Renderer', 'opengl');   % transparency-friendly
        drawnow;
        exportgraphics(gcf, fileName + ".pdf", 'ContentType', 'image', 'Resolution', 300);
    end
end
%% Make the 4 requested figures

% 1) U set community detection error (l1)
make_n_panels(mean_l1, sd_l1, baseline_l1, T, nseq, Kseq, KColors, ...
    '\fontsize{28}{28}\selectfont Error', ...
    'Community detection error on $\mathcal{L}$', ...
    'fig_Uset_community_error_T_byK');

% 2) Left singular matrix error (u)
make_n_panels(mean_u, sd_u, [], T, nseq, Kseq, KColors, ...
    '\fontsize{28}{28}\selectfont Error', ...
    'Error on the left singular matrix $U$', ...
    'fig_left_singular_error_T_byK');

