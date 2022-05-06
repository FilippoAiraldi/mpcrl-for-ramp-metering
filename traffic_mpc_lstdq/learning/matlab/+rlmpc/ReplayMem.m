classdef ReplayMem < handle
    % REPLAYMEM Dataset for experience replay.
    
    
    properties (GetAccess = public, SetAccess = private)
        maxcapacity
        data
        length
        headers
    end
    

    methods
        function obj = ReplayMem(capacity, varargin)
            % buffer = ReplayMem(capacity, var1, var2, ...)
            obj.maxcapacity = capacity;
            obj.clear(varargin{:});
        end

        function clear(obj, varargin)
            % clear(capacity, var1, var2, ...)

            % get headers and sizes
            obj.headers = varargin;

            % initialize data structure
            obj.data = struct;
            for h = obj.headers
                obj.data.(h{1}) = {};
            end
            obj.length = 0;
        end

        function add(obj, exp)
            % assert(nargin == 2)
            if obj.length < obj.maxcapacity
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
            % check if percentage
            if floor(n) ~= n
                n = round(n * obj.maxcapacity);
            end
            n = max(0, min(n, obj.length));
            if nargin < 3
                include_last_n = 0;
            elseif floor(include_last_n) ~= include_last_n
                include_last_n = round(n * include_last_n);
                include_last_n = max(0, min(n, include_last_n));
            end

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
            
            % combine into output samples
            idx_samples = [last_n, rand_n];
            samples = struct;
            samples.n = numel(idx_samples);
            for h = obj.headers
                datum = obj.data.(h{1})(idx_samples);
                samples.(h{1}) = squeeze(...
                    cat(ndims(datum{1}) + 1, datum{:}));
            end
        end
    end
end
