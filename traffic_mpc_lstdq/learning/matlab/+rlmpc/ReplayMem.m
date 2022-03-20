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
            assert(nargin == 2)
            if obj.length < obj.maxcapacity
                obj.length = obj.length + 1;
                obj.data{obj.length} = experience;
            else
                obj.data = obj.data(2:end);
                obj.data{obj.maxcapacity} = experience;
            end
        end
        
        function [samples, idx_samples] = sample(obj, n)
            % check if percentage
            if floor(n) ~= n
                n = round(n * obj.maxcapacity);
            end
            n = min(n, obj.length);

            % sample at random
            idx_samples = randperm(obj.length, n);
            samples = obj.data(idx_samples);
        end
    end
end
