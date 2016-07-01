function fn = writeXLS(result, datasetName, ~, wb)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   result:
%       The result object given from the function scriptStat_sup.
%
%   algorithmName:
%       Scalar character array
    
%% Get the field names and width of the dataset from the result   
    fn = fieldnames(result);
    width = size(result.(fn{1}), 2);
    
%% Transpose the fields into a matrix
    accum = zeros(length(fn), width);
    for n = 1:length(fn)
        a = result.(fn{n});
        accum(n,:) = a;
    end
    
%% Open the excel server
    if nargin < 4
        exl = actxserver('excel.application');
        exlWkbk = exl.Workbooks;
        wb = invoke(exlWkbk,'Add');
    end
    
%% Create a new sheet, name it
    curSheet = invoke(wb.Sheets,'Add');
    curSheet.Name = datasetName;
    
%% Set the labels
    range = calcrange('A1',1,length(fn));
    values = get(curSheet, 'Range', range);
    values.Value = fn';
    
%% Write the raw data (output from stat)
    accum = accum';
    range = calcrange('A2',size(accum,1), size(accum,2));
    values = get(curSheet, 'Range', range);
    values.Value = accum;
    
%% Make graph of the three classifiers (+1 for redundancy is 4) and their
%  accuracies. The classifiers will all be on the same graph.
    accum = accum'; %straighten out accumulated data
    
%% The plot
    cf= figure('Visible','off','PaperPositionMode','auto');

    %Plot the data
    hold on;
    
    titleStr = strcat('Accuracies of various classifiers on the dataset: "', datasetName, '"');
    
    if size(accum,2) ~= 1
        errorbar(expand(accum(1,:), 5, false), expand(accum(2,:), 5, true), 'b');
        errorbar(expand(accum(3,:), 5, false), expand(accum(4,:), 5, true), 'r');
        errorbar(expand(accum(5,:), 5, false), expand(accum(6,:), 5, true), 'g');
        plot(expand(accum(1,:), 5, false), 'LineWidth',2, 'Color', 'blue');
        plot(expand(accum(3,:), 5, false), 'LineWidth',2, 'Color', 'red');
        plot(expand(accum(5,:), 5, false), 'LineWidth',2,'Color', 'green');
        
        %Set up some cosmetic stuff
        xlabel('Feature number');
        ylabel('Accuracy');
        title(titleStr);
        axis('tight');
    else
        barweb([result.res_svm result.res_bayes result.res_j48], ...
            [result.res_svm_std result.res_bayes_std result.res_j48_std], ...
            [],[],...
            titleStr,...
            'Evaluator',...
            'Accuracy' ...
            );
    end

    legend('SVM','Bayes', 'J48', 'Location', 'Best');
    
    
    hold off;

%% Put the plot in the Excel diagram.
    pngLoc = [pwd filesep 'temp_pngs' filesep 'output.png'];
    print(cf,'-dpng',pngLoc);
    shapes = curSheet.Shapes;
    shapes.AddPicture(pngLoc,0,1,500,18,510,330);
    close(cf);
    
%% Make the redundancy plot
    cf= figure('Visible','off','PaperPositionMode','auto');

    %Plot the data
    hold on;
    
    titleStr = ['Redundancy of the dataset: "' datasetName '"'];
    
    insertRedPlot = true;
    
    if size(accum,2) ~= 1
        errorbar(expand(accum(7,:), 5, false), expand(accum(8,:), 5 ,true), 'b');
        plot(expand(accum(7,:), 5, false), 'Color', 'blue', 'LineWidth', 2);

        %Set up some cosmetic stuff
        xlabel('Feature number');
        ylabel(strcat('Redundancy'));
        axis('tight');
        title(titleStr);
    else
        if(result.res_red ~= 0)
            barweb(result.res_red, result.res_red_std,.25,[],titleStr,[],...
                'Redundancy');
        else
            insertRedPlot = false;
        end
    end
    
    hold off;
    
%% Put the plot in the Excel diagram.
    if insertRedPlot
        
        legend('Redundancy', 'Location', 'Best');
        
        pngLoc = [pwd filesep 'temp_pngs' filesep 'red.png'];
        print(cf,'-dpng',pngLoc);
        shapes = curSheet.Shapes;
        shapes.AddPicture(pngLoc,0,1,500,400,510,330);
    end
    
    close(cf);
    
    if nargin < 4
    %% Show the Excel file the the user
        exl.visible = 1;

    %% Kill the exl object
        clear exl;
    end
end