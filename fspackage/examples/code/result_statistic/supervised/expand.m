function [x] = expand(m, factor, reduceEbars)
    x = zeros(1, factor * length(m));
    
    for i = 1:length(x)
        if (reduceEbars && mod(i/factor, 1) == 0) || ~reduceEbars
            x(i) = m(ceil(i/factor));
        else
            x(i) = 0;
        end
    end
end