function mapped_genres = mapGenresToBig10(all_genres)
% Map original English genre tags (with optional numeric prefixes)
% into 10 short English classes defined above.

    %--- helper: trim, lower‑case & strip numeric prefixes --------------%
    cleanTag = @(g) regexprep(lower(strtrim(g)), '^[0-9]+\s*', '');

    %--- mapping table: lowercase keys ➜ short English class ------------%
    big10 = containers.Map( ...
        {'drama','romance','family', ...
         'comedy', ...
         'action','adventure','martial arts','sports', ...
         'crime','thriller','mystery','suspense', ...
         'sci-fi','scifi','sci fi','science fiction','fantasy', ...
         'horror', ...
         'animation','animated','children','kids', ...
         'history','war','biography','costume', ...
         'lgbtq+','lgbtq','gay','erotic','musical','music','sport'}, ...
        {'Drama', ...          % drama
         'RomFam','RomFam', ...% romance/family
         'Comedy', ...         % comedy
         'ActAdv','ActAdv','ActAdv','ActAdv', ...            % action group
         'Thriller','Thriller','Thriller','Thriller', ...    % thriller group
         'SciFiFan','SciFiFan','SciFiFan','SciFiFan','SciFiFan', ... % sci‑fi & fantasy
         'Horror', ...         % horror
         'Animation','Animation','Animation','Animation', ...% animation / kids
         'History','History','History','History', ...        % history group
         'Special','Special','Special','Special','Special','Special','Special'} );

    defaultClass = "Other";     % fallback for unknown tags
    mapped_genres = cell(size(all_genres));

    for i = 1:numel(all_genres)
        tags = all_genres{i};

        % split if stored as comma/semicolon separated string
        if ischar(tags) || isstring(tags)
            tags = strsplit(tags, {',',';'});
        end
        tags = cellfun(cleanTag, tags, 'uni', false);

        bigSet = strings(0);
        for t = tags
            key = char(t);
            if big10.isKey(key)
                bigSet(end+1) = big10(key); %#ok<AGROW>
            else
                bigSet(end+1) = defaultClass; %#ok<AGROW>
            end
        end
        bigSet = unique(bigSet);
        mapped_genres{i} = strjoin(bigSet, '/');  % e.g., 'Drama/Thriller'
    end
end
