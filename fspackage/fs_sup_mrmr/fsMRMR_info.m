function out = fsMRMR_info(X, Y, param)
% function out = fsMRMR(X, Y, param)
% 
% MIQ/MID scheme according to MRMR
% X - the data, each row is an instance  
% Y - the class label in  form of 1 2 3 4 5 6 7 ...
% param.k - number of selected features
% param.pool - the number features will be considered in the second iteration 
% param.type = 1: MIQ, -1:MID 

if nargin < 3
    param.k = 10; param.pool = 1000; param.type = 1;
end

if ~isfield(param, 'k')
    param.k = 10;
end

if ~isfield(param, 'pool')
    param.pool = 1000;
end

if ~isfield(param, 'type')
    param.type = 1;
end

nF = size(X,2); fea = zeros(param.k,1);

if ~isinteger(X)
    t = weka.filters.supervised.attribute.Discretize();
    t.setOptions(wekaArgumentString({'-R','first-last'}));
    B = SY2MY(Y);
    t.setInputFormat(wekaCategoricalData(X, B));
    dat = weka.filters.Filter.useFilter(wekaCategoricalData(X, B),t);
    for i = 0:nF-1
        X(:,i+1) = dat.attributeToDoubleArray(i);
    end
end

% information gain
t = zeros(nF,1);
for i=1:nF 
   t(i) = mutualinfo(X(:,i), Y);
end

% add features provide lowest redundancy
[orMI, idxs] = sort(-t); fea(1) = idxs(1); KMAX = min(param.pool,nF); 
idxleft = idxs(2:KMAX); mi_array = zeros(max(idxleft), param.k);
curlastfea = 1;

for ft = 2:param.k
   ncand = length(idxleft); c_mi = zeros(ncand,1);
   for i=1:ncand
      mi_array(idxleft(i),curlastfea) = mutualinfo(X(:,fea(curlastfea)), X(:,idxleft(i)));
      c_mi(i) = mean(mi_array(idxleft(i), :)); 
   end
   
   switch (param.type)
       case 1
           [tmp, tmpidx] = max(t(idxleft) ./ (c_mi + 0.01));
       case -1
           [tmp, tmpidx] = max(t(idxleft) - c_mi);
   end
   
   fea(ft) = idxleft(tmpidx); idxleft(tmpidx) = []; curlastfea = curlastfea + 1;
end

out.fList = fea; out.W = zeros(nF,1); out.W(fea) = param.k:-1:1; out.prf = -1;