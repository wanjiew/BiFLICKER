% List of dates matching your CSV filenames
% dates = { ...
%     '04-14', '04-28', '05-12', '05-26', ...
%     '06-09', '06-23', '07-07', '07-21', ...
%     '08-04', '08-18', '09-01', '09-15' };

dates = { ...
    '04-14_04-28_05-12', ...
    '05-26_06-09_06-23', ...
    '07-07_07-21_08-04', ...
    '08-18_09-01_09-15' };

% Initialize cell arrays
A_all = cell(1, numel(dates));
movie_info_all = cell(1, numel(dates));


%Read adjacency matrices and remove the users who haven't watched any
%movies
usernull = [];
for i = 1:numel(dates)
    date_str = dates{i};

    % Read adjacency matrix
    filename = ['adj_matrix_' date_str '.csv'];
    opts = detectImportOptions(filename, 'VariableNamingRule', 'preserve');
    opts.VariableNamesLine = 1;
    opts = setvartype(opts, 1, 'string');
    T = readtable(filename, opts);
    A = T{2:end, 2:end};
    A = A';
    A = 1*(A > 0);

    usernull = union(usernull, find(sum(A, 1) == 0));
end

%Now create the adjacency matrices and the movie information matrix
for i = 1:numel(dates)
    date_str = dates{i};
    % Read adjacency matrix
    filename = ['adj_matrix_' date_str '.csv'];
    opts = detectImportOptions(filename, 'VariableNamingRule', 'preserve');
    opts.VariableNamesLine = 1;
    opts = setvartype(opts, 1, 'string');
    T = readtable(filename, opts);
    A = T{2:end, 2:end};
    A = A';
    A(:, usernull) = [];
    A = 1*(A > 0);
    movienull = find(sum(A,2) >= size(A, 2)*0.15);
    A(movienull, :) = [];
    A_all{i} = A;
    
    fprintf('density: %.3f \n', sum(sum(A))/size(A,1)/size(A,2))

    % Read movie info
    movie_info = readtable(['movie_info_' date_str '.csv']);
    movie_info(movienull, :) = [];
    movie_info_all{i} = movie_info;
end

%% all movie information
%language
movie_info = vertcat(movie_info_all{:});

lang = categorical(movie_info.language);
langCounts = countcats(lang);
langCategories = categories(lang);

threshold = 15;
keepIdx = find(langCounts >= threshold);  % indices to keep
keepCats = langCategories(keepIdx);

% Set others
langGrouped = mergecats(lang, setdiff(langCategories, keepCats), 'Other');

% update back to movie_info
movie_info.language = langGrouped;
movie_info.language(isundefined(movie_info.language)) = 'NA';
% Genres
uniqueStrings = unique(movie_info.main_genre);
counts = cellfun(@(x) sum(strcmp(movie_info.main_genre, x)), uniqueStrings);
for i = 1:length(uniqueStrings)
    fprintf('String: %s, Count: %d\n', uniqueStrings{i}, counts(i));
end

% Regions
reg = categorical(movie_info.region);
regCounts = countcats(reg);
regCategories = categories(reg);

% Add the type information and the movie festival information
writetable(movie_info, 'movie_info_for_type_inference.csv');
movie_info_all = readtable('movie_info_full.csv');

%% Apply BiFLICKER to find eigenvectors%%
T = 3000;
inflation = 30;
result = BiFLICKER(A_all, 10, 10, T, [], inflation);

z1 = cellfun(@(x) x*result.Psi, A_all, 'UniformOutput', false);
stackz1 = vertcat(z1{:});
values = result.Xi' * stackz1
eigval = diag(values);


%% Movie clustering analysis
% First level 
Xi = result.Xi;

L1 = 5;

l1 = kmeans(Xi(:,1:5), L1, 'MaxIter', 200, 'Replicates', 50);
summary(categorical(l1))

%Movie grouping results
tmpn = 1;
for idx = 1:length(dates)
    nn = size(A_all{idx}, 1) - 1 + tmpn - 1;
    sprintf(['Movie Clustering Results: timestamp:' date_str])
    tmpl = categorical(l1(tmpn:nn));
    cats = categories(tmpl);
    counts = countcats(tmpl);
    for j = 1:numel(cats)
        fprintf('%s: %d\n', cats{j}, counts(j));
    end
    tmpn = tmpn + size(A_all{idx}, 1) - 1;
end

%Degree
A = vertcat(A_all{:});
movie_deg = sum(A,2);

figure
subplot(131)
hist(movie_deg)
subplot(132)
boxplot(movie_deg, l1)
subplot(133)
boxplot(movie_deg, l1norm)

%Interpret the results
%Movie clusters: Genre

T_year = groupsummary(table(l1, movie_info.main_genre), {'l1', 'Var2'});

% Extract variables
g = T_year.l1;
y = T_year.Var2;
nsize = T_year.GroupCount;

prop = 0*nsize; prop2 = 0*nsize;
for ii = 1:size(T_year, 1)
    prop(ii) = nsize(ii)/sum( l1 == T_year.l1(ii));
    prop2(ii) = prop(ii)/sum(strcmp(movie_info.main_genre, T_year(ii,:).Var2{1}));
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
xlim([min(unique_groups)-0.5, max(unique_groups)+0.5])

xlabel('Group')
ylabel('Genre')
title('Genres by Group and Year')
grid on


%Movie clusters: Region

T_region = groupsummary(table(l1, movie_info.region), {'l1', 'Var2'});

% Extract variables
g = T_region.l1;
y = T_region.Var2;
nsize = T_region.GroupCount;

prop = 0*nsize; prop2 = 0*nsize;
for ii = 1:size(T_region, 1)
    prop(ii) = nsize(ii)/sum( l1 == T_region.l1(ii));
    prop2(ii) = prop(ii)/sum(strcmp(movie_info.region, T_region(ii,:).Var2{1}));
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
xlim([min(unique_groups)-0.5, max(unique_groups)+0.5])
ylim([-0.5, numel(regCats)+0.5])

xlabel('Group')
ylabel('Region')
title('Regions by Group')
grid on

%Movie clusters: Language

T_lang = groupsummary(table(l1, movie_info.language), {'l1', 'Var2'});

% Extract variables
g = T_lang.l1;
y = T_lang.Var2;
nsize = T_lang.GroupCount;

prop = 0*nsize; prop2 = 0*nsize;
for ii = 1:size(T_lang, 1)
    prop(ii) = nsize(ii)/sum( l1 == T_lang.l1(ii));
    prop2(ii) = prop(ii)/sum(movie_info.language == T_lang(ii,:).Var2);
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
xlim([min(unique_groups)-0.5, max(unique_groups)+0.5])
ylim([-0.5, numel(regCats)+0.5])

xlabel('Group')
ylabel('Language')
title('Languages by Group')
grid on

%Movie clusters: Type

T_type = groupsummary(table(l1, movie_info_all.inferred_type), {'l1', 'Var2'});

% Extract variables
g = T_type.l1;
y = T_type.Var2;
nsize = T_type.GroupCount;

prop = 0*nsize; prop2 = 0*nsize;
for ii = 1:size(T_type, 1)
    prop(ii) = nsize(ii)/sum( l1 == T_type.l1(ii));
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
xlim([min(unique_groups)-0.5, max(unique_groups)+0.5])
ylim([-0.5, numel(regCats)+0.5])

xlabel('Group')
ylabel('Type')
title('Types by Group')
grid on



%Movie clusters: Movie Festivals

T_festival = groupsummary(table(l1, movie_info_all.is_festival), {'l1', 'Var2'});

% Extract variables
g = T_festival.l1;
y = T_festival.Var2;
nsize = T_festival.GroupCount;

prop = 0*nsize; prop2 = 0*nsize;
for ii = 1:size(T_festival, 1)
    prop(ii) = nsize(ii)/sum( l1 == T_festival.l1(ii));
    prop2(ii) = prop(ii)/sum(strcmp(movie_info_all.is_festival, T_festival(ii,:).Var2{1}));
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
xlim([min(unique_groups)-0.5, max(unique_groups)+0.5])
ylim([-0.5, numel(regCats)+0.5])

xlabel('Group')
ylabel('Movie Festivals')
title('Movie Festivals by Group')
grid on
%% Second level
counts = histcounts(l1, 1:L1+1);
[~, largest_cluster] = max(counts);
idx_largest = find(l1 == largest_cluster);

%Adjust A_all so that we only consider idx_largest 
A_large_comm = cell(1, length(A_all));
start = 0;
for i = 1:length(A_all)
    % Read adjacency matrix
    A = A_all{i};
    nk = size(A, 1); 

    keep = find(idx_largest > start & idx_largest <= start + nk);
    A = A(idx_largest(keep) - start, :);
    A_large_comm{i} = A;
    
    start = start + nk;
    fprintf('density: %.3f \n', sum(sum(A))/size(A,1)/size(A,2))
end

T = 3000;
inflation = 30;
result1 = BiFLICKER(A_large_comm, 10, 10, T, [], inflation);

z1 = cellfun(@(x) x*result1.Psi, A_large_comm, 'UniformOutput', false);
stackz1 = vertcat(z1{:});
values_largest = result1.Xi' * stackz1
eigval_largest = diag(values_largest);

Xi1 = result1.Xi;

L12 = 4;
l12 = kmeans(Xi1(:,1:4), L12, 'MaxIter', 200, 'Replicates', 50);
summary(categorical(l12))


%Degree
A = vertcat(A_all{:});
A_largest = A(idx_largest,:);
movie_deg1 = sum(A_largest,2);

figure
subplot(121)
hist(movie_deg1)
subplot(122)
boxplot(movie_deg1, l12)

%Interpret the results
movie_info1 = movie_info(idx_largest, :);
%Movie clusters: Genre
T_genre = groupsummary(table(l12, movie_info1.main_genre), {'l12', 'Var2'});

% Extract variables
g = T_genre.l12;
y = T_genre.Var2;
nsize = T_genre.GroupCount;

prop = 0*nsize; prop2 = 0*nsize;
for ii = 1:size(T_genre, 1)
    prop(ii) = nsize(ii)/sum( l12 == T_genre.l12(ii));
    prop2(ii) = prop(ii)/sum(strcmp(movie_info1.main_genre, T_genre(ii,:).Var2{1}));
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
xlim([min(unique_groups)-0.5, max(unique_groups)+0.5])

xlabel('Group')
ylabel('Genre')
title('Genres by Group and Year')
grid on


%Movie clusters: Region

T_region = groupsummary(table(l12, movie_info1.region), {'l12', 'Var2'});

% Extract variables
g = T_region.l12;
y = T_region.Var2;
nsize = T_region.GroupCount;

prop = 0*nsize; prop2 = 0*nsize;
for ii = 1:size(T_region, 1)
    prop(ii) = nsize(ii)/sum( l12 == T_region.l12(ii));
    prop2(ii) = prop(ii)/sum(strcmp(movie_info1.region, T_region(ii,:).Var2{1}));
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
xlim([min(unique_groups)-0.5, max(unique_groups)+0.5])
ylim([-0.5, numel(regCats)+0.5])

xlabel('Group')
ylabel('Region')
title('Regions by Group')
grid on

%Movie clusters: Language

T_lang = groupsummary(table(l12, movie_info1.language), {'l12', 'Var2'});

% Extract variables
g = T_lang.l12;
y = T_lang.Var2;
nsize = T_lang.GroupCount;

prop = 0*nsize; prop2 = 0*nsize;
for ii = 1:size(T_lang, 1)
    prop(ii) = nsize(ii)/sum( l12 == T_lang.l12(ii));
    prop2(ii) = prop(ii)/sum(movie_info1.language == T_lang(ii,:).Var2);
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
xlim([min(unique_groups)-0.5, max(unique_groups)+0.5])
ylim([-0.5, numel(regCats)+0.5])

xlabel('Group')
ylabel('Language')
title('Languages by Group')
grid on

%Movie clusters: Type

T_type = groupsummary(table(l12, movie_info1.type), {'l12', 'Var2'});

% Extract variables
g = T_type.l12;
y = T_type.Var2;
nsize = T_type.GroupCount;

prop = 0*nsize; prop2 = 0*nsize;
for ii = 1:size(T_type, 1)
    prop(ii) = nsize(ii)/sum( l12 == T_type.l12(ii));
    prop2(ii) = prop(ii)/sum(strcmp(movie_info1.type, T_type(ii,:).Var2{1}));
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
xlim([min(unique_groups)-0.5, max(unique_groups)+0.5])
ylim([-0.5, numel(regCats)+0.5])

xlabel('Group')
ylabel('Type')
title('Languages by Group')
grid on

%% Combine two levels
l1orig = l1;
l1(idx_largest) = L1 + l12;
%% User clustering Analyasis %%
%Movie figures
Psi = result.Psi; 
L2 = 5;
l2 = kmeans(Psi(:,1:5), L2, 'MaxIter', 100, 'Replicates', 50);

%User grouping results
summary(categorical(l2))

tmpl = categorical(l1);
cats = categories(tmpl);

T = zeros(L2, length(cats));
A = vertcat(A_all{:});

for i = 1:L2
    for j = 1:length(cats)
        T(i,j) = sum(sum(A(tmpl == cats{j}, l2 == i)))/sum(l2 == i)/sum(tmpl == cats{j});
    end
end

figure;
plot(T')
xlabel("User Groups")
ylabel("Connectivity with Movies")

%Generate the legend according to groups
counts = histcounts(l2, 1:L2+1);  % size of each group
legend_entries = cell(1, L2);  % initialize legend text

for k = 1:L2
    idx = (l2 == k);
    legend_entries{k} = sprintf('Group %d (n = %d)', k, counts(k));
end

%Generate the legend according to groups
counts = histcounts(l1, 1:L1+1);  % size of each group
xtick_entries = cell(1, L1);  % initialize legend text
for k = 1:L1
    idx = (l1 == k);
    xtick_entries{k} = sprintf('Group %d (n = %d)', k, counts(k));
end
xticks(1:5);
set(gca, 'XTickLabel', xtick_entries);
legend(legend_entries, 'Location', 'best');
title('Connectivity with Movie Groups');


A = vertcat(A_all{:});
user_deg = sum(A, 1);

figure
subplot(121)
hist(user_deg)
subplot(122)
boxplot(user_deg, l2)
set(gca, 'XTickLabel', legend_entries);