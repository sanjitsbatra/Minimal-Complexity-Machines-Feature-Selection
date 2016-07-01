function [eva] = expFun_eva_sup(filePath, filename)
% function [acc] = expFun(filePath, filename)
%   filePath - the path of the data
%   filename - the name of the file

numF = 5:5:200;

disp('%%%%%%%%%%%%%%%%%%%%%%%%');
disp(filename);
disp('%%%%%%%%%%%%%%%%%%%%%%%%');

RES = []; X = []; Partition = [];
load(filePath{1}{1}, 'RES'); 
load(filePath{2}{1}, 'X', 'Y'); 
load(filePath{3}{1}, 'Partition');

%If it's feature-oriented
if(RES{1}.fImp)
    eva = expFun_all_feats(X, Y, Partition, RES);
    return;
end

iter = length(RES);

OrgX = X;

svmRes = cell(iter,1);
bayesRes = cell(iter,1);
j48Res = cell(iter,1);
redRes = cell(iter,1);

for it = 1:iter
    disp('------------------------------------');
    fprintf('              %i iteration\n',it);          
    disp('------------------------------------');
    % sample the data
    trainIDX = Partition(:,it) == 1;
    testIDX = Partition(:,it) == -1;
    
    % normalize feature to have zero mean and unit norm according to
    % training data
    trainX = OrgX(trainIDX, :);
    [~, meanX, nor] = normData(trainX, 1, 1);
    X = OrgX - ones(size(OrgX,1),1)*meanX;
    X = X.*repmat(nor,size(OrgX,1),1);
    clear trainX;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   Test Methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %----------------------------
    % Feature Selection
    %----------------------------
    fprintf('--------------------\n');
    fprintf('Iteration, %3i\n',it);
    featIDX = RES{it}.fList;

    svmRes{it} = evalFSClasSVMCV( X, Y, trainIDX, testIDX, featIDX, numF);
    bayesRes{it} = evalFSClasBayes(X, Y, trainIDX, testIDX, featIDX, numF);
    j48Res{it} = evalFSClasJ48(X, Y, trainIDX, testIDX, featIDX, numF);
    redRes{it} = evalFSRedncy( X, featIDX, numF);
    
    if length(numF) == 20
        svmRes{it} = mean(svmRes{it});
        bayesRes{it} = mean(bayesRes{it});
        j48Res{it} = mean(j48Res{it});
        redRes{it} = mean(redRes{it});
    end
end

eva.svm = svmRes;
eva.bayes = bayesRes;
eva.j48 = j48Res;
eva.red = redRes;