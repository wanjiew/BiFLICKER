%% Make 4 full figures for BiFLICKER simulations
%  - Rows: n = 3600, 4000, 8000
%  - Cols: K = 5, 10, 36
%  - x-axis in each panel: density
%  - 5 methods: NewIn5, DistPCA, disPCA, FastPCA, Full
%    displayed as: "New method", "DistPCA", "disPCA", "FastPCA", "Oracle"

clear; clc;

% ----- Simulation settings (must match your generator) -----
nseq    = [2000, 4000, 8000];
Kseq    = [5, 10, 20];
density = -0.2:0.1:0.3;

% Internal method order in err.*:
methodsAll = {'New','NewIn5','DistPCA','disPCA','FastPCA','Full'};
% We keep columns corresponding to: NewIn5, DistPCA, disPCA, FastPCA, Full
idxSubset          = [2, 3, 4, 5, 6];
methodInternal     = methodsAll(idxSubset);
methodDisplayNames = {'New method','DistPCA','disPCA','FastPCA','Oracle'};
methodColors = containers.Map;
methodColors('New method') = [1 0 0];   % red
methodColors('DistPCA')    = [0 0.7 0];        
methodColors('disPCA')     = [1 0.8 0];        
methodColors('FastPCA')    = [0 0 1];        
methodColors('Oracle')     = [0 0 0];   % black


nN = numel(nseq);
nK = numel(Kseq);
nD = numel(density);
nM = numel(idxSubset);  % 5 methods

% ----- Containers: (iN, iK, iD, method) -----
mean_l1 = nan(nN, nK, nD, nM);  % U-set community error
mean_l2 = nan(nN, nK, nD, nM);  % V-set community error
mean_u  = nan(nN, nK, nD, nM);  % left singular matrix error
mean_v  = nan(nN, nK, nD, nM);  % right singular matrix error

%% Load all files and compute means for each error type

for iN = 1:nN
    n = nseq(iN);

    for iK = 1:nK
        K = Kseq(iK);

        for iD = 1:nD
            dd = density(iD);

            filename = sprintf('Exp1/n_%d_K_%d_density_%.1f.mat', n, K, dd);
            fprintf('Loading %s\n', filename);

            S = load(filename, 'err', 'Repetitions');
            R = S.Repetitions;

            % Each err.* is (R × 6) in order:
            % [New, NewIn5, DistPCA, disPCA, FastPCA, Full]
            m_l1 = mean(S.err.l1, 1);  % 1 × 6
            m_l2 = mean(S.err.l2, 1);
            m_u  = mean(S.err.u,  1);
            m_v  = mean(S.err.v,  1);

            mean_l1(iN, iK, iD, :) = m_l1(idxSubset);
            mean_l2(iN, iK, iD, :) = m_l2(idxSubset);
            mean_u( iN, iK, iD, :) = m_u( idxSubset);
            mean_v( iN, iK, iD, :) = m_v( idxSubset);
        end
    end
end

%% Helper function to make a 3×3 figure for one metric
%  data: (iN, iK, iD, method)
function make_grid_figure(data, density, nseq, Kseq, methodNames, methodColors, yLabel, figTitle, fileName)
    nN = numel(nseq);
    nK = numel(Kseq);
    nM = numel(methodNames);

    axAll   = gobjects(nN, nK);   % NEW: store all axes
    axLast = [];   % will store the last axes handle (for legend)

    figure('Units','normalized','Position',[0.02 0.02 0.8 0.8]);
    t = tiledlayout(nN, nK, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, figTitle, 'Interpreter', 'latex', 'FontSize', 36);
    
    ymax = max(max(max(max(data)))) + 0.1;

    for iN = 1:nN
        for iK = 1:nK
            ax = nexttile(t);
            axAll(iN, iK) = ax;          % NEW
            if iN == nN && iK == nK-1
                axLast = ax;  % use the last axes for the legend
            end
            
            hold(ax, 'on');

            for j = 1:nM
                y    = squeeze(data(iN, iK, :, j));
                name = methodNames{j};

                % ===== Color control from input struct =====
                if isKey(methodColors, name)
                    thisColor = methodColors(name);
                else
                    thisColor = [];   % MATLAB default
                end

                if isempty(thisColor)
                    plot(ax, density, y, '-o', ...
                         'LineWidth', 1.0, ...
                         'DisplayName', name);
                else
                    plot(ax, density, y, '-o', ...
                         'LineWidth', 1.0, ...
                         'Color', thisColor, ...
                         'DisplayName', name);
                end
            end

            hold(ax, 'off');
            title(ax, sprintf('\\fontsize{60}{34}\\selectfont $n = %d$, $K = %d$', nseq(iN), Kseq(iK)), ...
                  'Interpreter', 'latex');

            % x-axis only on bottom row
            if iN == nN
                xlabel(ax, '\fontsize{36}{34}\selectfont Density shift $d$', 'Interpreter', 'latex');
            else
                ax.XTickLabel = [];
            end

            % y-axis only on first column
            if iK == 1
                ylabel(ax, yLabel, 'FontSize', 36);
            else
                ax.YTickLabel = [];
            end

            ylim(ax, [0, max(ymax, 1)]);
            xlim(ax, [-0.2, 0.3])
            grid(ax, 'on');
            box(ax, 'on');
            set(ax, 'LineWidth', 1);
        end
    end


    % === Legend in the new bottom band ===
    lg = legend(axLast, methodNames, ...
                'Orientation', 'horizontal', ...
                'Location','southoutside', ...
                'Interpreter', 'latex',  'FontSize', 48);
    
    lg.Units = 'normalized';


    set(gcf, 'Color', 'w');

    if nargin >= 9 && ~isempty(fileName)
        exportgraphics(gcf, fileName + ".pdf", 'ContentType', 'vector', 'Resolution', 300);
    end
end

%% Make the 4 requested figures

% 1) U set community detection error (l1)
make_grid_figure(...
    mean_l1, density, nseq, Kseq, methodDisplayNames, methodColors, ...
    'Error', ...
    'Community detection error on $L$', ...
    'fig_Uset_community_error');

% 2) V set community detection error (l2)
make_grid_figure(...
    mean_l2, density, nseq, Kseq, methodDisplayNames, methodColors, ...
    'Error', ...
    'Community detection error on $R$', ...
    'fig_Vset_community_error');

% 3) Left singular matrix error (u)
make_grid_figure(...
    mean_u, density, nseq, Kseq, methodDisplayNames, methodColors, ...
    'Error', ...
    'Error on the left singular matrix $U$', ...
    'fig_left_singular_error');

% 4) Right singular matrix error (v)
make_grid_figure(...
    mean_v, density, nseq, Kseq, methodDisplayNames, methodColors, ...
    'Error', ...
    'Error on the right singular matrix $V$', ...
    'fig_right_singular_error');
