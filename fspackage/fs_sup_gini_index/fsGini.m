function [out] = fsGini(X,Y)
    [~,n] = size(X);
    W = zeros(n,1);    
    
    for i=1:n
        values = unique(X(:,i));
        v = size(values,1);
        W(i) = 0.5;
        for j=1:v
            left_Y = Y(X(:,i) <= values(j));
            right_Y = Y(X(:,i) > values(j));
            
            gini_left = 0;
            gini_right = 0;
            
            for k=min(Y):max(Y)
               gini_left = gini_left + (size(left_Y(left_Y == k),1)/size(left_Y,1))^2;
               gini_right = gini_right + (size(right_Y(right_Y == k),1)/size(right_Y,1))^2;
            end
            gini_left = 1-gini_left;
            gini_right = 1-gini_right;
            
            current_gini = (size(left_Y,1)*gini_left + size(right_Y,1)*gini_right)/size(Y,1);
            if current_gini<W(i)
                W(i) = current_gini;
            end
        end
    end
    [out.W out.fList]= sort(W);
    out.prf = -1;
end