function m2d = reshape4d(m4d)
    % RESHAPE4D. Reshapes a 4D matrix of the form (ITER, EP, N, K) into a 
    % 2D matrix of the form (N, ITER*EP*K).
    
    if isstruct(m4d)
        % if the argument is a structure, apply the function to each field
        m2d = struct;
        for n = fieldnames(m4d)'
            m2d.(n{1}) = util.reshape4d(m4d.(n{1}));
        end
    else
        N = size(m4d, 3);
        m2d = reshape(permute(m4d, [3, 4, 2, 1]), N, []);
    end
end
