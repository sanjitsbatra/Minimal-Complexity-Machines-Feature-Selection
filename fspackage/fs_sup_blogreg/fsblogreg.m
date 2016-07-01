function [out] = fsblogreg( X,Y, param )
%function [ wFeat SF ] = fsblogreg( X,Y )
%   use sparse logistic regression to select features.
%   X - the data, each row is an instance
%   Y - the class label. in 1 2 3 4... format
%   the algorithm is able to handle multi-class problem using the One-vs-All Technique.

numL = size(X,1);
numF = size(X,2);
numC = length(unique(Y));

% center the data
X = X - ones(numL,1)*mean(X,1);

% normalize the features of the data
featNorm = sqrt(sum(X .* X,1));
X = X*diag(featNorm.^-1);

if numC == 2
    Y(Y==2) = -1;
    out.W = blogreg(X,Y,param.tol);
else
    weights = zeros(numF,numC);
    
    % evaluate features in one-vs-all strategy
    for i = 1:numC
        newY = Y;
        newY(Y~=i) = -1;
        newY(Y==i) = 1;
        weights(:,i) = blogreg(X,newY,param.tol);
    end
    
    out.W = sum(weights,2);
end

[out.W out.fList] = sort(abs(out.W), 'descend');

out.W = out.W(out.W > 0);
out.fList = out.fList(1:size(out.W));
out.prf = -1;
end
