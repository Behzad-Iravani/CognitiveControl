%'UTF-8'
classdef dipoles 

    properties
        mom
        time
    end
methods
    function obj = select(obj, latency)
                obj.mom = cellfun(@(x) x(:, obj.time>=latency(1) & obj.time<=latency(2)),...
                    obj.mom, 'UniformOutput', false);
    end
end

end 

