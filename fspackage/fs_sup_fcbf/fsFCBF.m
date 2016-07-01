function [out] =  fsFCBF(X,Y)
%Using Weka's feature selection algorithm
% X,  the features on current trunk, each colum is a feature vector on all
%     instances, and each row is a part of the instance
% Y,  the label of instances, in single column form: 1 2 3 4 5 ...

% a.E = 'weka.attributeSelection.SymmetricalUncertAttributeSetEval';
% a.S = 'weka.attributeSelection.FCBFSearch -D false -T -1.7976931348623157E308 -N -1';


tempY = SY2MY(Y);
t = weka.filters.supervised.attribute.AttributeSelection();
  
%% handle options
a.E = 'weka.attributeSelection.SymmetricalUncertAttributeSetEval';
a.S = 'weka.attributeSelection.FCBFSearch -D false -T -1.7976931348623157E308 -N -1';

tmp=wekaArgumentString({'-E',a.E});
tmp=wekaArgumentString({'-S',a.S},tmp);
t.setOptions(tmp);

%% train classifier
cat = wekaCategoricalData(X,tempY);
t.setInputFormat(cat);
newDat = weka.filters.Filter.useFilter(cat,t);
clear cat;

out.fList = [];

numF = newDat.numAttributes()-2;
for i = 0:numF
    str = newDat.attribute(i).name;
    str = str.toCharArray()';
    strArry = strread(str,'%s');
    out.fList(i+1) = str2num(strArry{2});
end