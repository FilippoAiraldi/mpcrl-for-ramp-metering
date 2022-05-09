classdef ReplayMem < handle
    % REPLAYMEM Dataset for experience replay.
    
    
    properties (GetAccess = public, SetAccess = private)
        capacity (1, 1) double
        data (1, 1) struct
        length (1, 1) double
        headers (1, :) cell
    end

    properties (GetAccess = public, SetAccess = public)
        reduce (1, :) char {mustBeTextScalar} = 'none'
    end
    


    methods
        function obj = ReplayMem(capacity, reduce, varargin)
            % REPLAYMEM. Builds an instance handde to a buffer with the
            % given capacity, reduction method (applied after sampling),
            % and the given headers (in varargin)
            arguments
                capacity (1, 1) double {mustBePositive,mustBeInteger}
                reduce (1, :) char {mustBeTextScalar} = 'none'
            end
            arguments (Repeating)
                varargin (1, :) char {mustBeTextScalar}
            end
            
            % save capacity and reduce
            obj.capacity = capacity;
            if isempty(reduce)
                reduce = 'none';
            end
            obj.reduce = reduce;

            % initialize
            obj.clear(varargin{:});
        end

        function set.reduce(obj, val) 
            assert(ismember(val, {'sum', 'mean', 'none'}), ...
                'invalid reduce method');
            obj.reduce = val;
        end

        function clear(obj, varargin)
            % CLEAR. Clears the buffer memory completely. Reinitializes the
            % headers as well, if provided.
            arguments
                obj (1, 1) rlmpc.ReplayMem
            end
            arguments (Repeating)
                varargin (1, :) char {mustBeTextScalar}
            end

            % get headers
            if numel(varargin) > 0
                obj.headers = varargin;
            end

            % initialize data structure
            obj.data = struct;
            for h = obj.headers
                obj.data.(h{1}) = {};
            end
            obj.length = 0;
        end

        function add(obj, exp)
            % ADD. Adds the given experience item to the buffer.
            arguments
                obj (1, 1) rlmpc.ReplayMem
                exp (1, 1) struct
            end
            if obj.length < obj.capacity
                % append to last position
                obj.length = obj.length + 1;
                for h = obj.headers
                    obj.data.(h{1}){obj.length} = exp.(h{1});
                end
            else
                % shift each header by one position back and append item to
                % last position (now empty)
                for h = obj.headers
                    obj.data.(h{1}) = obj.data.(h{1})(2:end);
                    obj.data.(h{1}){end + 1} = exp.(h{1});
                end
            end
        end
        
        function [samples, idx_samples] = sample(obj, n, include_last_n)
            % SAMPLE. Extracts from the buffer a sample of given size at
            % random. If given, forcibly includes also the last N items.
            arguments
                obj (1, 1) rlmpc.ReplayMem
                n (1, 1) double {mustBeNonnegative,mustBeFinite}
                include_last_n ...
                    (1, 1) double {mustBeNonnegative,mustBeFinite} = 0
            end

            % check if percentage
            if floor(n) ~= n
                n = round(n * obj.capacity);
            end
            n = max(0, min(n, obj.length));
            if floor(include_last_n) ~= include_last_n
                include_last_n = round(n * include_last_n);
            end
            include_last_n = max(0, min(n, include_last_n));

            % get last n
            if include_last_n > 0
                last_n = obj.length-include_last_n+1:obj.length;
            else
                last_n = [];
            end

            % sample at random the remaining
            to_sample = n - include_last_n;
            if to_sample > 0
                rand_n = randperm(obj.length - include_last_n, to_sample);
            else
                rand_n = [];
            end
            
            % combine into output samples - reduce if necessary
            idx_samples = [last_n, rand_n];
            samples = struct;
            samples.n = numel(idx_samples);
            for h = obj.headers
                % concat in a single matrix
                datum = obj.data.(h{1})(idx_samples);
                datum = squeeze(...
                    cat(ndims(datum{1}) + 1, datum{:}));

                % perform reduce op
                switch obj.reduce
                    case 'sum'
                        datum = sum(datum, ndims(datum));
                    case 'mean'
                        datum = mean(datum, ndims(datum));
                end
                
                % assign final 
                samples.(h{1}) = datum;
            end
        end
    end
end
