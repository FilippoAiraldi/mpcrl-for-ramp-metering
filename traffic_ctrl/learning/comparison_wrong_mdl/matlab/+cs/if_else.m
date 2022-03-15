function varargout = if_else(varargin)
    %IF_ELSE Branching on MX nodes Ternary operator, "cond ? if_true : if_false".
    %
    %  DM = IF_ELSE(DM cond, DM if_true, DM if_false, bool short_circuit)
    %  SX = IF_ELSE(SX cond, SX if_true, SX if_false, bool short_circuit)
    %  MX = IF_ELSE(MX cond, MX if_true, MX if_false, bool short_circuit)
    %
    %  Had to manually pick this function from the original CasADi Matlab 
    % installation folder since it is missing from its namespace.
    [varargout{1:nargout}] = casadiMEX(235, varargin{:});
end
