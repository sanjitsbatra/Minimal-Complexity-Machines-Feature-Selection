function plotDataByAlgorithm(algorithm)

    %initialize an excel instance
    exl = actxserver('excel.application');
    exlWkbk = exl.Workbooks;
    wb = invoke(exlWkbk,'Add');

    files = what([algorithm '_results']);
    files = files.mat;
    
    for i = 1:length(files)
        result = load([algorithm '_results' filesep files{i}]);
        datasetEnd = findstr(files{i},'_') - 1;
        datasetName = substr(files{i},0,datasetEnd);
        writeXLS(result.res, datasetName,...
            algorithm...
            , wb);
    end
    
    %show the excel doc, kill the instance
    exl.visible = 1;
    clear exl;
end