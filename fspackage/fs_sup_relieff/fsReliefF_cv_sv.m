function out = fsReliefF_cv_sv( X, Y, k, grid )

if nargin <= 3
    grid = 1:10;
end

maxACC = 0;
for i=1:length(grid)
    out = fsReliefF(X, Y, grid(i), -1);
    wFeat = out.W;
    [foo oridx] = sort(wFeat,'descend');
    curX = X(:,oridx(1:k));
    [AC cvACC] = SVM_CV_estimate(curX,Y,curX,Y);
    cvACC = max(max(cvACC));
    if maxACC <= cvACC
        maxACC = cvACC; maxW = wFeat; maxK = grid(i);
    end
    fprintf('%i features, k: %i, Acc: %f\n', k, grid(i), cvACC);
end
out.W = maxW; out.k = maxK;