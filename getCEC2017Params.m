function [lb, ub, dim, fobj] = getCEC2017Params(function_name)
    % Define bounds and dimensions based on the function name
    switch function_name
        case 'F1'
            lb = -100; ub = 100; dim = 30; fobj = @F1;
        case 'F3'
            lb = -30; ub = 30; dim = 30; fobj = @F3;
        case 'F6'
            lb = -5.12; ub = 5.12; dim = 30; fobj = @F6;
        % Add more cases for F4 to F30 as needed
        otherwise
            error('Function not recognized');
    end
end
