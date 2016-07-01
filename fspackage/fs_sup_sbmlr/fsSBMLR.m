function [out] = fsSBMLR(X, Y)
%

t = SY2MY(Y);
t(t == -1) = 0;
   
if size(t,2) == 2 && size(unique(t),1)
    t = [t not(t)];
end

wFeat = sbmlr(X, t);

wFeat = wFeat.*wFeat;

wFeat = sum(wFeat,2);

[foo, featIDX] = sort(abs(wFeat),'descend');

featIDX(wFeat(featIDX)==0) = [];

foo(foo==0) = [];

out.W = wFeat;
out.fList = featIDX;
out.prf = -1;

end