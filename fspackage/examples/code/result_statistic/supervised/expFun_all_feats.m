function [eva] = expFun_all_feats(X, Y, Partition, RES)
% function [acc] = expFun(filePath, filename)
%   filePath - the path of the data
%   filename - the name of the file

    iter = length(RES);

    OrgX = X;
    
    svmRes = cell(iter, 1);
    bayesRes = cell(iter, 1);
    j48Res = cell(iter, 1);
    redRes = cell(iter, 1);

    for it = 1:iter
        disp('------------------------------------');
        disp('No iterations, testing all features at once');
        disp('------------------------------------');

        

        trainIDX = Partition(:,it) == 1;
        testIDX = Partition(:,it) == -1;

        % normalize feature to have zero mean and unit norm according to
        % training data
        trainX = OrgX(trainIDX, :);
        [foo, meanX, nor] = normData(trainX, 1, 1);
        X = OrgX - ones(size(OrgX,1),1)*meanX;
        X = X.*repmat(nor,size(OrgX,1),1);
        clear trainX;

        %----------------------------
        % Feature Selection
        %----------------------------
        fprintf('---------\n');
        fprintf('Evaluation Step: %3i\n', it);
        
        featIDX = RES{it}.fList;
        
        if ~isempty(featIDX)
            svmRes{it} = evalFSClasSVMCV(X, Y, trainIDX, ...
                testIDX, featIDX, length(featIDX));
            bayesRes{it} = evalFSClasBayes(X, Y, trainIDX, ...
                testIDX, featIDX, length(featIDX));
            j48Res{it} = evalFSClasJ48(X, Y, trainIDX, ...
                testIDX, featIDX, length(featIDX));
            redRes{it} = evalFSRedncy(X, featIDX, length(featIDX));
        end
    end
    
    %Average the returned values.
    eva.res_svm = mean(cell2mat(svmRes));
    eva.res_svm_std = std(cell2mat(svmRes));
    
    eva.res_bayes = mean(cell2mat(bayesRes));
    eva.res_bayes_std = std(cell2mat(bayesRes));
    
    eva.res_j48 = mean(cell2mat(j48Res));
    eva.res_j48_std = std(cell2mat(j48Res));
    
    eva.res_red = mean(abs(cell2mat(redRes)));
    eva.res_red_std = std(abs(cell2mat(redRes)));
    
end