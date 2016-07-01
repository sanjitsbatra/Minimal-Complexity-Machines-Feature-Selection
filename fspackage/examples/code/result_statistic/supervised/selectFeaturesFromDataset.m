function [fileName] = selectFeaturesFromDataset(datasetName, algorithmCode, filePath)

%% Set some initial values we wish to use before running a test.
    fileName = {datasetName};
    if nargin < 3
        filePath = ['..' filesep '..' filesep '..' filesep 'data' filesep];
    end
    fileDir = cell(2,1);

    %run for each file in the dataset
    for i = 1: length(fileName)
%% This are the two files for the dataset we wish to evaluate
        fileDir{1} = strcat(filePath,fileName{i},'.mat');
        fileDir{2} = strcat(filePath,fileName{i},'_part.mat');
%         try
%% Calculate the feature selection on the dataset
            RES = expFun_wi_sam_feat(fileDir, fileName{i}, algorithmCode);
%% Save the results to a file (make sure it is algorithmCode-specific)            
            file = strcat(filePath, fileName{i},'_',algorithmCode,'_result', '.mat');
            save(file, 'RES');
%         catch ME
%             fprintf('An error was caught for %s data\n',fileName{i});
%             disp(ME.message);
%             ME.stack(1)
%         end
    end

end