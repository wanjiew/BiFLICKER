% In this file, we decompose the users into different servers
% We use python to create the overall bipartite network


% Read the adjacency matrix
% Each row is a user and each column is a movie
filename = "adj_matrix_combined.csv";
opts = detectImportOptions(filename, 'VariableNamingRule', 'preserve');
opts.VariableNamesLine = 1;
opts = setvartype(opts, 1, 'string');
T = readtable(filename, opts);
A_all = T{2:end, 2:end};
A_all = 1*(A_all > 0);
size(A_all)

% Read the movies
filename = "movie_info_combined.csv";
opts = detectImportOptions(filename, 'VariableNamingRule', 'preserve');
opts.VariableNamesLine = 1;
opts = setvartype(opts, 1, 'string');
movie_info_all = readtable(filename, opts);
size(movie_info_all)
class(movie_info_all)

% Read the user upvotes and number of movies
filename = "user_metadata.csv";
opts = detectImportOptions(filename, 'VariableNamingRule', 'preserve');
opts.VariableNamesLine = 1;
opts = setvartype(opts, 1, 'string');
Meta = readtable(filename, opts);
Meta_all = table2array(Meta(:, 2:3));

% clean the data, and keep the movies with at least 15 users
deg_movie = sum(A_all, 1);
idx_keep = find(deg_movie >= 15 & deg_movie < size(A_all, 1)*0.2);

A_all = A_all(:, idx_keep);
movie_info_all = movie_info_all(idx_keep, :);

size(A_all)
size(movie_info_all)
size(Meta_all)
%% Generate the data on each server
% Decompose the bipartite matrix into several servers
S = 5;
% Servers consider users with different number of movies
s = zeros(size(A_all, 1), 1);

s(Meta_all(:,2) >= 450) = 1;
s(Meta_all(:,2) < 450 & Meta_all(:,2) >= 300) = 2;
s(Meta_all(:,2) < 300 & Meta_all(:,2) >= 150) = 3;
s(Meta_all(:,2) < 150 & Meta_all(:,2) >= 50) = 4;
s(Meta_all(:,2) < 50) = 5;
summary(categorical(s))

A = cell(1, S);
Meta = cell(1, S);
for i = 1:S
    A{i} = A_all(s == i, :);   
    Meta{i} = Meta_all(s == i, :);
    fprintf('density: %.3f \n', sum(sum(A{i}))/size(A{i},1)/size(A{i},2))
end

%% all movie information
%language
m = size(movie_info_all, 1);

lang = categorical(movie_info_all.language);
langCounts = countcats(lang);
langCategories = categories(lang);

threshold = m * 0.05;
keepIdx = find(langCounts >= threshold);  % indices to keep
keepCats = langCategories(keepIdx);

% Set others
langGrouped = mergecats(lang, setdiff(langCategories, keepCats), 'Other');

% update back to movie_info
movie_info_all.language = langGrouped;
movie_info_all.language(isundefined(movie_info_all.language)) = 'NA';

% Genres
uniqueStrings = unique(movie_info_all.main_genre);
counts = cellfun(@(x) sum(strcmp(movie_info_all.main_genre, x)), uniqueStrings);
for i = 1:length(uniqueStrings)
    fprintf('String: %s, Count: %d\n', uniqueStrings{i}, counts(i));
end

% Regions
reg = categorical(movie_info_all.region);
regCounts = countcats(reg);
regCategories = categories(reg);
for i = 1:length(regCounts)
    fprintf('Region: %s, Count: %d\n', regCategories{i}, regCounts(i));
end

% Types
type = categorical(movie_info_all.inferred_type);
typeCounts = countcats(type);
typeCategories = categories(type);
for i = 1:length(typeCounts)
    fprintf('Type: %s, Count: %d\n', typeCategories{i}, typeCounts(i));
end

% Festivals
festival = categorical(movie_info_all.festival_class);
FestCounts = countcats(festival);
FestCategories = categories(festival);
for i = 1:length(FestCounts)
    fprintf('Festivals: %s, Count: %d\n', FestCategories{i}, FestCounts(i));
end

% Years
year = movie_info_all.year;
decade = cell(size(year, 1), 1); 
decade(isnan(year)) = {'Undefined'};
decade(year == 2023) = {'Current Year'};
decade(year < 2023 & year >= 2020) = {'After 2020'};
decade(year <= 2020 & year > 2010) = {'2010-2020'};
decade(year <= 2010 & year > 2000) = {'2000-2010'};
decade(year <= 2000 & year > 1990) = {'1990-2000'};
decade(year <= 1990) = {'Before 1990'};

decadeCounts = countcats(categorical(decade));
decadeCategories = categories(categorical(decade));
for i = 1:length(decadeCounts)
    fprintf('Decade: %s, Count: %d\n', decadeCategories{i}, decadeCounts(i));
end

%% Apply BiFLICKER to find eigenvectors%%
% Set a random seed
rng(42);

T = 5000;
inflation = 20;
result = BiFLICKER(A, 10, 10, T, [], inflation);

z1 = cellfun(@(x) x*result.Psi, A, 'UniformOutput', false);
stackz1 = vertcat(z1{:});
values = result.Xi' * stackz1;
disp(values)
eigval = diag(values);

A_all = vertcat(A{:});

%% User clustering analysis
Xi = result.Xi;

L1 = 6;
l1 = kmeans(Xi(:,1:L1), L1, 'MaxIter', 200, 'Replicates', 50);
summary(categorical(l1))

%Relabel so that Community 1 has the most users, and so on
group_counts = histcounts(l1, 0.5:1:(L1+0.5));  % bin edges: 0.5 to 6.5
[~, sorted_labels] = sort(group_counts, 'descend');  
relabel_map = zeros(1, L1);
for new_label = 1:L1
    old_label = sorted_labels(new_label);
    relabel_map(old_label) = new_label;
end
l1 = arrayfun(@(x) relabel_map(x), l1);
summary(categorical(l1))


%User grouping results in each server
tmpn = 1;
for idx = 1:length(A)
    nn = size(A{idx}, 1) - 1 + tmpn - 1;
    fprintf('User Clustering Results on Server: %s\n', idx)
    tmpl = categorical(l1(tmpn:nn));
    cats = categories(tmpl);
    counts = countcats(tmpl);
    for j = 1:numel(cats)
        fprintf('%s: %d\n', cats{j}, counts(j));
    end
    tmpn = tmpn + size(A{idx}, 1) - 1;
end

%Generate the legend according to user groups
user_groups = cell(1, L1);  % initialize legend text

for k = 1:L1
    idx = (l1 == k);
    user_groups{k} = sprintf('User Comm. %d', k);
end

%Degree
user_deg = sum(A_all, 2);

figure;
subplot(121)
histogram(user_deg)
subplot(122)
boxplot(user_deg, l1)
set(gca, 'XTickLabel', user_groups);
title('User Degrees by Groups')

%% User communities on each server

% Count class occurrences at each location
n = cellfun(@(x) size(x,1), A); 
counts = zeros(length(n), L1);
prop = zeros(length(n), L1);

nn = 0;
for kk = 1:length(n)
    for jj = 1:L1
        counts(kk, jj) = sum(l1((1:n(kk))+nn) == jj);
    end
    prop(kk, :) = counts(kk, :)/n(kk);
    nn = nn + n(kk);
end

base_cmap = lines(L1);  % Original bold colors
alpha = 0.5;            % Soften level: 0 = white, 1 = original color
cmap_user = alpha * base_cmap + (1 - alpha) * ones(size(base_cmap));  % blend with white
%cmap_user = summer(L1);
% Plot stacked bar chart
figure;

subplot(121)
hb = bar(counts, 'stacked');

for k = 1:L1
    hb(k).FaceColor = cmap_user(k, :);
end

xlabel('Servers')
ylabel('Number of users')
title('Community Distribution by Servers')
legend(arrayfun(@(x) sprintf('Class %d', x), 1:L1, 'UniformOutput', false))

% Plot stacked bar chart
subplot(122)
hb = bar(prop, 'stacked');

for k = 1:L1
    hb(k).FaceColor = cmap_user(k, :);
end

xlabel('Servers')
ylabel('Proportion of users')
title('Community Distribution by Servers')
legend(arrayfun(@(x) sprintf('User Group %d', x), 1:L1, 'UniformOutput', false))

%% Upvotes/Number of Movies
Meta_all = vertcat(Meta{:});

figure;
boxplot( log(Meta_all(:,1)./Meta_all(:,2)), l1)
title('log(Upvotes/Movies)')
xlabel('Group')
set(gca, 'XTickLabel', user_groups);

%% Movie clustering and relabelling
%Movie figures
Psi = result.Psi; 
L2 = 6;
l2 = kmeans(Psi(:,1:L2), L2, 'MaxIter', 100, 'Replicates', 50);

%Movie grouping results
summary(categorical(l2))

%Connectivity matrix
tmpl = categorical(l2);
cats = categories(tmpl);

T = zeros(L1, length(cats));
for i = 1:L1
    for j = 1:length(cats)
        T(i,j) = sum(sum(A_all(l1 == i, tmpl == cats{j})))/sum(l1 == i)/sum(tmpl == cats{j});
    end
end

%Generate the legend according to movie groups
relabel_map = zeros(1, 6);
counts = histcounts(l2, 0.5:1:(L2+0.5));  % size of each group
[~, main_group] = max(counts); 
relabel_map(main_group) = 1;

remaining_groups_old = setdiff(1:L2, main_group);
remaining_groups_new = setdiff(1:L1, 1);
while ~isempty(remaining_groups_old)
    idx = remaining_groups_old(1);
    scores = T(remaining_groups_new, idx);
    [~, new_idx] = max(scores);
    relabel_map(idx) = remaining_groups_new(new_idx);
    remaining_groups_old(1) = [];
    remaining_groups_new(new_idx) = [];
end
relabel_map(relabel_map == 2) = 0;
relabel_map(relabel_map == 3) = 2;
relabel_map(relabel_map == 0) = 3;

l2 = arrayfun(@(x) relabel_map(x), l2);

%Movie grouping results
summary(categorical(l2))
%% Movie Clustering analysis
%Degree
movie_groups = cell(1, L2);  % initialize legend text
counts = histcounts(l2, 0.5:1:(L2+0.5));  % size of each group
for k = 1:L2
    idx = (l2 == k);
    movie_groups{k} = sprintf('Movie Comm. %d', k);
end


movie_deg = sum(A_all, 1);
figure;
subplot(121)
histogram(movie_deg)
subplot(122)
boxplot(movie_deg, l2)
set(gca, 'XTickLabel', movie_groups);
title('Movie Degrees by Groups')


%% Connectivity matrix
tmpl = categorical(l2);
cats = categories(tmpl);

T = zeros(L1, length(cats));

for i = 1:L1
    for j = 1:length(cats)
        T(i,j) = sum(sum(A_all(l1 == i, tmpl == cats{j})))/sum(l1 == i)/sum(tmpl == cats{j});
    end
end

figure;
set(gca, 'ColorOrder', cmap_user, 'NextPlot', 'replacechildren');
plot(T', 'linewidth', 2)
xlabel("Movie Communities")
ylabel("Connectivity with Movies")
xticks(1:L1);
set(gca, 'XTickLabel', movie_groups);
legend(user_groups, 'Location', 'best');
title('Connectivity with Movie Communities');



%% Movie clusters interpretation: Type

T_type = groupsummary(table(l2, movie_info_all.inferred_type), {'l2', 'Var2'});

% Extract variables
g = T_type.l2;
y = T_type.Var2;
nsize = T_type.GroupCount;

prop = 0*nsize; prop2 = 0*nsize;
for ii = 1:size(T_type, 1)
    prop(ii) = nsize(ii)/sum( l2 == T_type.l2(ii));
    prop2(ii) = prop(ii)/sum(strcmp(movie_info_all.inferred_type, T_type(ii,:).Var2{1}));
end

% Convert categorical genre to numeric y-axis positions
[regCats, ~, regIdx] = unique(y);

% Assign a color to each genre
numRegion = numel(regCats);
cmap = lines(numRegion);  % Or use parula, jet, etc.
colors = cmap(regIdx, :);  % One color per point

% Plot
figure;
scatter(g, regIdx, 250000*prop2, colors, 'filled')
yticks(1:numel(regCats))
yticklabels(regCats)

% Set xticks based on groups
unique_groups = unique(g);
xticks(unique_groups)
xticklabels(movie_groups(unique_groups))
xlim([min(unique_groups)-0.5, max(unique_groups)+0.5])
ylim([-0.5, numel(regCats)+0.5])

xlabel('Group')
ylabel('Type')
title('Types by Movie Groups')
grid on


%% Clustering Interpretation: Genre
T_genre = groupsummary(table(l2, movie_info_all.main_genre), {'l2', 'Var2'});

% Extract variables
g = T_genre.l2;
y = T_genre.Var2;
nsize = T_genre.GroupCount;

prop = 0*nsize; prop2 = 0*nsize;
for ii = 1:size(T_genre, 1)
    prop(ii) = nsize(ii)/sum( l2 == T_genre.l2(ii));
    prop2(ii) = prop(ii)/sum(strcmp(movie_info_all.main_genre, T_genre(ii,:).Var2{1}));
end

% Convert categorical genre to numeric y-axis positions
[genreCats, ~, genreIdx] = unique(y);

% Assign a color to each genre
numGenres = numel(genreCats);
cmap = lines(numGenres);  % Or use parula, jet, etc.
colors = cmap(genreIdx, :);  % One color per point

% Plot
figure;
scatter(g, genreIdx, 250000*prop2, colors, 'filled')
yticks(1:numel(genreCats))
yticklabels(genreCats)

% Set xticks based on groups
unique_groups = unique(g);
xticks(unique_groups)
xticklabels(movie_groups(unique_groups))
xlim([min(unique_groups)-0.5, max(unique_groups)+0.5])

xlabel('Group')
ylabel('Genre')
title('Genres by Movie Groups')
grid on

%% Movie clusters: Language

T_lang = groupsummary(table(l2, movie_info_all.language), {'l2', 'Var2'});

% Extract variables
g = T_lang.l2;
y = T_lang.Var2;
nsize = T_lang.GroupCount;

prop = 0*nsize; prop2 = 0*nsize;
for ii = 1:size(T_lang, 1)
    prop(ii) = nsize(ii)/sum( l2 == T_lang.l2(ii));
    prop2(ii) = prop(ii)/sum(movie_info_all.language == T_lang(ii,:).Var2);
end

% Convert categorical genre to numeric y-axis positions
[regCats, ~, regIdx] = unique(y);

% Assign a color to each genre
numRegion = numel(regCats);
cmap = lines(numRegion);  % Or use parula, jet, etc.
colors = cmap(regIdx, :);  % One color per point

% Plot
figure;
scatter(g, regIdx, 250000*prop2, colors, 'filled')
yticks(1:numel(regCats))
yticklabels(regCats)

% Set xticks based on groups
unique_groups = unique(g);
xticks(unique_groups)
xticklabels(movie_groups(unique_groups))
xlim([min(unique_groups)-0.5, max(unique_groups)+0.5])
ylim([-0.5, numel(regCats)+0.5])

xlabel('Group')
ylabel('Language')
title('Languages by Movie Groups')
grid on


%% Movie clusters: Region

T_region = groupsummary(table(l2, movie_info_all.region), {'l2', 'Var2'});

% Extract variables
g = T_region.l2;
y = T_region.Var2;
nsize = T_region.GroupCount;

prop = 0*nsize; prop2 = 0*nsize;
for ii = 1:size(T_region, 1)
    prop(ii) = nsize(ii)/sum( l2 == T_region.l2(ii));
    prop2(ii) = prop(ii)/sum(strcmp(movie_info_all.region, T_region(ii,:).Var2{1}));
end

% Convert categorical genre to numeric y-axis positions
[regCats, ~, regIdx] = unique(y);

% Assign a color to each genre
numRegion = numel(regCats);
cmap = lines(numRegion);  % Or use parula, jet, etc.
colors = cmap(regIdx, :);  % One color per point

% Plot
figure;
scatter(g, regIdx, 1000000*prop2, colors, 'filled')
yticks(1:numel(regCats))
yticklabels(regCats)

% Set xticks based on groups
unique_groups = unique(g);
xticks(unique_groups)
xticklabels(movie_groups(unique_groups))
xlim([min(unique_groups)-0.5, max(unique_groups)+0.5])
ylim([-0.5, numel(regCats)+0.5])

ylabel('Region')
title('Regions by Movie Groups')
grid on




%% Movie clusters: Movie Festivals

T_festival = groupsummary(table(l2, movie_info_all.festival_class), {'l2', 'Var2'});

% Extract variables
g = T_festival.l2;
y = T_festival.Var2;
nsize = T_festival.GroupCount;

prop = 0*nsize; prop2 = 0*nsize;
for ii = 1:size(T_festival, 1)
    prop(ii) = nsize(ii)/sum( l2 == T_festival.l2(ii));
    prop2(ii) = prop(ii)/sum(strcmp(movie_info_all.festival_class, T_festival(ii,:).Var2{1}));
end

% Convert categorical genre to numeric y-axis positions
[regCats, ~, regIdx] = unique(y);

% Assign a color to each genre
numRegion = numel(regCats);
cmap = lines(numRegion);  
colors = cmap(regIdx, :); 

% Plot
figure;
scatter(g, regIdx, 250000*prop2, colors, 'filled')
yticks(1:numel(regCats))
yticklabels(regCats)

% Set xticks based on groups
unique_groups = unique(g);
xticks(unique_groups)
xticklabels(movie_groups(unique_groups))
xlim([min(unique_groups)-0.5, max(unique_groups)+0.5])
ylim([-0.5, numel(regCats)+0.5])

xlabel('Group')
ylabel('Movie Festivals')
title('Movie Festivals by Movie Groups')
grid on


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Paper Figure Section %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Figure 1. Basic User Analysis

figure;
%User degree by communities
tmpl = categorical(l2);
cats = categories(tmpl);

T = zeros(L1, length(cats));

for i = 1:L1
    for j = 1:length(cats)
        T(i,j) = sum(sum(A_all(l1 == i, tmpl == cats{j})))/sum(l1 == i)/sum(tmpl == cats{j});
    end
end

figure;
set(gca, 'ColorOrder', cmap_user, 'NextPlot', 'replacechildren');
plot(T', 'linewidth', 2)
ylabel("Connectivity with Movies")
xticks(1:L2);
xlim([0.8, L2 + 0.2])
set(gca, 'XTickLabel', movie_groups);
xtickangle(0)
legend(user_groups, 'Location', 'best');
title('User-Movie Connectivity');

exportgraphics(gcf, 'connectivity.pdf', 'ContentType', 'vector');

%% Figure 2. Movie community by servers
figure;

% Movie years
T_decade = groupsummary(table(l2, decade), {'l2', 'decade'});

% Extract variables
g = T_decade.l2;
y = T_decade.decade;
nsize = T_decade.GroupCount;

prop = 0*nsize; prop2 = 0*nsize;
for ii = 1:size(T_decade, 1)
    prop(ii) = nsize(ii)/sum( l2 == T_decade.l2(ii));
    prop2(ii) = prop(ii)/sum(strcmp(decade, T_decade(ii,:).decade{1}));
end

% Convert categorical genre to numeric y-axis positions
[regCats, ~, regIdx_orig] = unique(y, 'stable');
Decade_map = zeros(numel(regCats), 1);
Decade_map(strcmp(regCats, 'Before 1990')) = 1;
Decade_map(strcmp(regCats, '1990-2000')) = 2;
Decade_map(strcmp(regCats, '2000-2010')) = 3;
Decade_map(strcmp(regCats, '2010-2020')) = 4;
Decade_map(strcmp(regCats, 'After 2020')) = 5;
Decade_map(strcmp(regCats, 'Current Year')) = 6;
Decade_map(strcmp(regCats, 'Undefined')) = 7;

regIdx = Decade_map(regIdx_orig);

% Assign a color to each genre
numRegion = numel(regCats);
cmap = lines(numRegion);  
colors = cmap(regIdx, :); 

% Plot
subplot(221)
scatter(g, regIdx, 450000*prop2, colors, 'filled')
yticks(1:numel(regCats))
yticklabels({'Before 1990', '1990-2000', '2000-2010', '2010-2020', 'After 2020', 'Current Year', 'Undefined'})

% Set xticks based on groups
unique_groups = unique(g);
xticks(unique_groups)
xticklabels(movie_groups(unique_groups))
xlim([min(unique_groups)-0.5, max(unique_groups)+0.5])
ylim([0.5, numel(regCats)+0.5])

title('Movie Production Dates by Communities')
grid on



% Movie regions
subplot(222)
T_region = groupsummary(table(l2, movie_info_all.region), {'l2', 'Var2'});

% Extract variables
g = T_region.l2;
y = T_region.Var2;
nsize = T_region.GroupCount;

prop = 0*nsize; prop2 = 0*nsize;
for ii = 1:size(T_region, 1)
    prop(ii) = nsize(ii)/sum( l2 == T_region.l2(ii));
    prop2(ii) = prop(ii)/sum(strcmp(movie_info_all.region, T_region(ii,:).Var2{1}));
end

% Convert categorical genre to numeric y-axis positions
[regCats, ~, regIdx_orig] = unique(y);
Region_map = zeros(numel(regCats), 1);
Region_map(strcmp(regCats, '')) = 1;
Region_map(strcmp(regCats, 'Germany')) = 2;
Region_map(strcmp(regCats, 'France')) = 3;
Region_map(strcmp(regCats, 'UK')) = 4;
Region_map(strcmp(regCats, 'USA')) = 5;
Region_map(strcmp(regCats, 'Mainland China')) = 6;
Region_map(strcmp(regCats, 'Hong Kong')) = 7;
Region_map(strcmp(regCats, 'Taiwan')) = 8;
Region_map(strcmp(regCats, 'South Korea')) = 9;
Region_map(strcmp(regCats, 'Japan')) = 10;

regIdx = Region_map(regIdx_orig);

% Assign a color to each genre
numRegion = numel(regCats);
cmap = lines(numRegion);  % Or use parula, jet, etc.
colors = cmap(regIdx, :);  % One color per point

% Plot
scatter(g, regIdx, 400000*prop2, colors, 'filled')
yticks(1:numel(regCats))
yticklabels({'Others', 'Germany', 'France', 'UK', 'USA', 'Mainland China', ...
    'Hong Kong', 'Taiwan', 'South Korea', 'Japan'})

% Set xticks based on groups
unique_groups = unique(g);
xticks(unique_groups)
xticklabels(movie_groups(unique_groups))
xlim([min(unique_groups)-0.5, max(unique_groups)+0.5])
ylim([0.5, numel(regCats)+0.5])

title('Movie Regions by Communities')
grid on




% Movie type
T_type = groupsummary(table(l2, movie_info_all.inferred_type), {'l2', 'Var2'});

% Extract variables
g = T_type.l2;
y = T_type.Var2;
nsize = T_type.GroupCount;

prop = 0*nsize; prop2 = 0*nsize;
for ii = 1:size(T_type, 1)
    prop(ii) = nsize(ii)/sum( l2 == T_type.l2(ii));
    prop2(ii) = prop(ii)/sum(strcmp(movie_info_all.inferred_type, T_type(ii,:).Var2{1}));
end

% Convert categorical genre to numeric y-axis positions
[regCats, ~, regIdx] = unique(y);

% Assign a color to each type
numRegion = numel(regCats);
cmap = lines(numRegion);  % Or use parula, jet, etc.
colors = cmap(regIdx, :);  % One color per point

% Plot
subplot(223)
scatter(g, regIdx, 350000*prop2, colors, 'filled')
yticks(1:numel(regCats))
yticklabels(regCats)

% Set xticks based on groups
unique_groups = unique(g);
xticks(unique_groups)
xticklabels(movie_groups(unique_groups))
xlim([min(unique_groups)-0.5, max(unique_groups)+0.5])
ylim([0.5, numel(regCats)+0.5])

title('Movie Types by Communities')
grid on

% Movie Genres
[genre_vec, genreCats]= mapGenresToVec(all_genres);

prop = zeros(L2, numel(genreCats));
prop2 = prop;
for ii = 1:L2
    for jj = 1:numel(genreCats)
        prop(ii, jj) = sum(genre_vec(l2 == ii, jj))/sum( l2 == ii);
        prop2(ii, jj) = prop(ii, jj)/sum(genre_vec(:, jj));
    end
end

selected = [2, 3, 4, 6, 8, 9, 10, 11, 13, 14, 17, 19, 21, 22, 23];
genre_sel = genreCats(selected);

% Assign a color to each genre
numGenres = numel(genre_sel);
genreIdx = 1:numGenres;
cmap = lines(numGenres);  % Or use parula, jet, etc.
colors = cmap(genreIdx, :);  % One color per point

% Plot
subplot(224)
hold on;
prop2(prop2 == 0) = NaN;
for ii = 1:L2
    scatter(ones(1,numGenres)*ii, genreIdx, prop2(ii,selected)*150000, colors, 'filled');
end
yticks(1:numGenres)
yticklabels(genre_sel)
ylim([0.5, numGenres+0.5])

xticks(1:L2)
xticklabels(movie_groups)
xlim([0.5, L2+0.5])

title('Genres by Communities')
grid on



exportgraphics(gcf, 'movie.pdf', 'ContentType', 'vector');


%% Figure 3. User community by servers

% Count class occurrences at each location
n = cellfun(@(x) size(x,1), A); 
counts = zeros(length(n), L1);
prop = zeros(length(n), L1);

nn = 0;
for kk = 1:length(n)
    for jj = 1:L1
        counts(kk, jj) = sum(l1((1:n(kk))+nn) == jj);
    end
    prop(kk, :) = counts(kk, :)/n(kk);
    nn = nn + n(kk);
end

base_cmap = lines(L1);  % Original bold colors
alpha = 0.5;            % Soften level: 0 = white, 1 = original color
cmap_user = alpha * base_cmap + (1 - alpha) * ones(size(base_cmap));  % blend with white

figure;
subplot(121)
hb = bar(counts, 'stacked');

for k = 1:L1
    hb(k).FaceColor = cmap_user(k, :);
end

xlabel('Servers')
ylabel('Number of users')
title('Community Distribution by Servers: Counts')
legend(arrayfun(@(x) sprintf('User Comm. %d', x), 1:L1, 'UniformOutput', false))

% Plot stacked bar chart
subplot(122)
hb = bar(prop, 'stacked');

for k = 1:L1
    hb(k).FaceColor = cmap_user(k, :);
end

xlabel('Servers')
ylabel('Proportion of users')
title('Community Distribution by Servers: Proportion')
ylim([0, 1.1])

exportgraphics(gcf, 'servers.pdf', 'ContentType', 'vector');
