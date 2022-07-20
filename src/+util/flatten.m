function f = flatten(m)
    % FLATTEN. Flattens a multi-dimensional tensor into a 2D matrix. The 
    % inputs are processed according to
    %   * input [ITER, EP] -> output [1, ITER*EP]
    %   * input [ITER, EP, K] -> output [1, ITER*EP*K]
    %   * input [ITER, EP, N, K] -> output [N, ITER*EP*K]
    % If the input is a struct, then each field will be processed.
    
    if isstruct(m)
        % if the argument is a structure, apply the function to each field
        f = struct;
        for n = fieldnames(m)'
            f.(n{1}) = util.flatten(m.(n{1}));
        end
    else
        switch ndims(m)
            case 4
                N = size(m, 3);
                f = reshape(permute(m, [3, 4, 2, 1]), N, []);
            case 3
                f = reshape(permute(m, [3, 2, 1]), 1, []);
            case 2
                f = reshape(permute(m, [2, 1]), 1, []);
            otherwise
                error('invalid matrix dimension')
        end
    end
end
