function RES = expFun_wi_sam_feat(filePath, filename, algorithmCode)
% function [RES] = expFun(filePath, filename)
%   filePath - the path of the data
%   filename - the name of the file
%   algorithmCode: the code for the algorithm you wish to have evaluated
%       Codes:
%           blogreg
%           relieff


disp('%%%%%%%%%%%%%%%%%%%%%%%%');
disp(filename);
disp('%%%%%%%%%%%%%%%%%%%%%%%%');

X = []; Y = []; Partition = [];

load(filePath{1}, 'X', 'Y');
load(filePath{2}, 'Partition');
    
iter = size(Partition,2);
RES = cell(iter,1);

for it = 1:iter
    start = clock();
    % sample the data
    curX = X(Partition(:,it)==1,:);
    curY = Y(Partition(:,it)==1,:);

    % zero mean, unit norm
    curX = normData(curX, 1, 1);

    numF = 200;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   Test Methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %---------------------------
    % REL_cv
    %---------------------------
    fprintf('REL_cv\n');

    switch algorithmCode
        case 'blogreg'
            param.tol = 1;
            out = fsblogreg( curX, curY, param);
            out.fImp = true;
        case 'cfs'
            out = fsCFS( curX, curY);
            out.fImp = true;
        case 'chi2'
            out = fsChiSquare( curX, curY );
        case 'fcbf'
            out = fsFCBF( curX, curY );
            out.fImp = true;
        case 'fisher'
            out = fsFisher( curX, curY );
        case 'gini'
            out = fsGini( curX, curY );
        case 'infogain'
            out = fsInfoGain( curX, curY );
        case 'kruskalwallis'
            out = fsKruskalWallis( curX, curY );
        case 'mrmr'
            out = fsMRMR( curX, curY );
        case 'relieff'
            out = fsReliefF_cv_sv( curX, curY, numF );
        case 'sbmlr'
            out = fsSBMLR( curX, curY );
            out.fImp = true;
        case 'spectrum'
            W = constructRBF(curX);
            [wFeat, ~] = fsSpectrum(W, curX, 0);
            out.prf = 1;
            out.W = sort(wFeat*out.prf);
        case 'ttest'
            out = fsTtest( curX, curY ); 
    end

    
    if(isfield(out, 'fImp'))
        RES{it}.fImp = true;
        RES{it}.fList = out.fList;
    else
        RES{it}.W = out.W;
        RES{it}.fImp = false;
        [~, RES{it}.fList ] = sort(out.W, 'descend');
    end
    finalTime = clock() - start;
    RES{it}.runtime = clock2secs(finalTime);
end

end