function [genre_vec, vecname] = mapGenresToVec(all_genres)
% Map original English genre tags into a vector

    %--- helper: trim, lower‑case & strip numeric prefixes --------------%
    cleanTag = @(g) regexprep(strtrim(erase(g, {'[',']',''''})), '^[0-9]+\s*', '');

    % Define the corresponding matrix
    n = size(all_genres, 1);
    d = 23;
    vecname = {'Erotic', 'LGBTQ+', 'Horror', 'Thriller', 'Mystery', ...
        'Musical', 'Sci-Fi', 'Animation', 'Fantasy', 'Costume', 'Martial Arts', 'Adventure',...
        'Sports', 'Action', 'Crime', 'Comedy', 'Family', 'Children', 'Romance', 'War', ...
        'Biography', 'Drama', 'Other'};
    genre_vec = zeros(n, d);

    for i = 1:numel(all_genres)
        tags = all_genres{i};

        % split if stored as comma/semicolon separated string
        if ischar(tags) || isstring(tags)
            tags = strsplit(tags, {',',';'});
        end
        tags = cellfun(cleanTag, tags, 'uni', false);

        for t = tags
            genre_vec(i, strcmp(vecname, t)) = 1;
        end
        if sum(genre_vec(i,:)) == 0
            genre_vec(i, d) = 1;
        else
            genre_vec(i,:) = genre_vec(i,:)/sum(genre_vec(i,:));
        end
    end
end
