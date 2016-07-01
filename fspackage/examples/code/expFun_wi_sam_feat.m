function [RES, multiTest] = expFun_wi_sam_feat(filePath, filename, algorithmCode)
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

found = true;

switch algorithmCode
    case 'blogreg'
        normData(X,1,1);
        param.tol = 1;
        out = fsblogreg( X, Y, param);
    case 'fcbf'
        out = fsFCBF( X, Y );
    case 'sbmlr'
        out = fsSBMLR( X, Y );
    otherwise
        found = false;
end

multiTest = ~found;

if found
    RES.W = out.W;
    RES.fList = out.fList;
end

if ~found
    
    iter = size(Partition,2);
    RES = cell(iter,1);
    
    for it = 1:iter
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
            case 'cfs'
                out = fsCFS( curX, curY);
            case 'chi2'
                out = fsChiSquare( curX, curY );
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
            case 'ttest'
                out = fsTtest( curX, curY );
            case 'spectrum'
                W = constructRBF(curX);
                [wFeat, ~] = fsSpectrum(W, curX, 0);
                out.prf = 1;
                out.W = sort(wFeat*out.prf); 
        end

        RES{it}.W = out.W;
        [~, RES{it}.fList ] = sort(out.W, 'descend');
    end
end

end