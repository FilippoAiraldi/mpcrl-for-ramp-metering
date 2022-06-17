function m = ncell2mat(c, depth)
    % NCELL2MAT. Nested cell2mat. Converts the content of a cell array 
    % into a matrix, and, if the content is another array of cells, then 
    % the function is recursively applied to the nested cell array. 
    % An optional maximum recursive depth can be provided to stop the 
    % process early.
    %
    % The function works only if the content of the cell array are homogeneous, i.e., all cells or 
    % all matrices, otherwise the behavior is unknown.

    if nargin < 2
        depth = inf;
    end

    if isstruct(c)
        % if the argument is a structure, apply the function to each field
        m = struct;
        for n = fieldnames(c)'
            m.(n{1}) = util.ncell2mat(c.(n{1}), depth);
        end
    else
        % logics core
        if depth > 0 && iscell(c)
            m = util.ncell2mat([c{:}], depth - 1);
        else
            m = c;
        end
    end
end
