classdef ReplayMem < handle
    % REPLAYMEM Dataset for experience replay.
    
    
    properties (GetAccess = public, SetAccess = private)
        maxcapacity
        data
        length
    end
    

    methods
        function obj = ReplayMem(capacity)
            obj.maxcapacity = capacity;
            obj.clear();
        end

        function clear(obj)
            obj.data = cell(0, 0);
            obj.length = 0;
        end

        function add(obj, experience)
            % assert(nargin == 2)
            if obj.length < obj.maxcapacity
                obj.length = obj.length + 1;
                obj.data{obj.length} = experience;
            else
                obj.data(1:end-1) = obj.data(2:end);
                obj.data{obj.maxcapacity} = experience;
            end
        end
        
        function [samples, idx_samples] = sample(obj, n, include_last_n)
            % check if percentage
            if floor(n) ~= n
                n = round(n * obj.maxcapacity);
            end
            if nargin < 3
                include_last_n = 0;
            elseif floor(include_last_n) ~= include_last_n
                    include_last_n = round(include_last_n * obj.maxcapacity);
            end
            n = min(n, obj.length);
            include_last_n = min(include_last_n, n);

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
            samples = obj.data(idx_samples);
        end
    end
end
