clear;clc;

javaaddpath('/Users/Sanjit/Desktop/Play/ML_JD/MCM/Feature_Selection/Feature_Selection_Experiments/fspackage/lib/weka/weka.jar');

datasetname = 'GLI-85';

% Get data
data = load(strcat('~/Desktop/Play/ML_JD/MCM/Feature_Selection/Feature_Selection_Experiments/Data/',datasetname,'.csv'));



for rand_seed = 2:5 % random seed number (we proceed from 1-(7)19, if the need arises) 

    rand_seed


% Initialize Variables
size(data);
x=data(:,1:end-1);
y=data(:,end);

N = size(x,1);
D = size(x,2);
l1 = length(find(y==1));
Class_Ratio = max(l1,(N-l1))/min(l1,(N-l1));

nfolds = 5; % Number of Folds used in Cross Validation

% Change path
cd ~/Desktop/Play/ML_JD/MCM/Feature_Selection/Feature_Selection_Experiments/Results/

% Open file handles for storing results

f_lmcm = fopen(strcat(datasetname,'_LMCM.stats'),'w');
f_jmi = fopen(strcat(datasetname,'_JMI.stats'),'w');
f_cmim = fopen(strcat(datasetname,'_CMIM.stats'),'w');
f_mrmr = fopen(strcat(datasetname,'_MRMR.stats'),'w');
f_relieff = fopen(strcat(datasetname,'_RELIEFF.stats'),'w');

f_summary = fopen(strcat('SUMMARY_',datasetname,'.stats'),'a');

fprintf(f_lmcm,'Date is %s\nDataset name is %s\nNumber of Folds is %d\nMethod name is LMCM\nFormat is: \nAccuracy, MCC, AUC(ROC), BCR, Average Precision\n',date,datasetname,nfolds);
fprintf(f_jmi,'Date is %s\nDataset name is %s\nNumber of Folds is %d\nMethod name is JMI\nFormat is: \nAccuracy, MCC, AUC(ROC), BCR, Average Precision\n',date,datasetname,nfolds);
fprintf(f_cmim,'Date is %s\nDataset name is %s\nNumber of Folds is %d\nMethod name is CMIM\nFormat is: \nAccuracy, MCC, AUC(ROC), BCR, Average Precision\n',date,datasetname,nfolds);
fprintf(f_mrmr,'Date is %s\nDataset name is %s\nNumber of Folds is %d\nMethod name is MRMR\nFormat is: \nAccuracy, MCC, AUC(ROC), BCR, Average Precision\n',date,datasetname,nfolds);
fprintf(f_relieff,'Date is %s\nDataset name is %s\nNumber of Folds is %d\nMethod name is ReliefF\nFormat is: \nAccuracy, MCC, AUC(ROC), BCR, Average Precision\n',date,datasetname,nfolds);

fprintf(f_summary,'\n\nDate is %s\nDataset name is %s\nN = %d \t D = %d \t Class Ratio = %f \nRand Seed is %d\n\nNumber of Folds is %d\nFormat is: \nAccuracy, MCC, AUC(ROC), BCR, Average Precision\n',date,datasetname,N, D, Class_Ratio,rand_seed,nfolds);


cd ../

% Generate folds and Evaluate the performances of different FS methods


rand('state',rand_seed);
r=randperm(size(y,1));
y=y(r,:);
x=x(r,:);
nsize = size(x, 1);
foldstart = 1 + round(nsize * (0:nfolds)/nfolds);


for fold = 1:nfolds
    s = foldstart(fold);
    e = foldstart(fold + 1) - 1;
    xTest = x((s:e), :);
    yTest = y((s:e), :);
    e1 = foldstart(fold) - 1;
    s2 = foldstart(fold + 1);
    e2 = nsize;
    xTrain = [x(1:e1, :); x(s2:e2, :)];
    yTrain = [y(1:e1, :); y(s2:e2, :)];
    
    % Normalize Data for this fold (Mean = 0, Stdev = 1, for every feature)
    for i = 1:D
        m(i) = mean(xTrain(:,i));
        stdev(i) = std(xTrain(:,i));
    end
    
    for i = 1:D
        xTrain(:,i) = xTrain(:,i) - m(i);
        xTest(:,i) =  xTest(:,i)  - m(i);
        if(stdev(i) ~= 0 )
            xTrain(:,i) = xTrain(:,i)/stdev(i);
            xTest(:,i)  =  xTest(:,i)/stdev(i);
        end
    end
    
    
    
    %--------------------------------------------------------------------------
    
    % Save Folds as mat files
    
    % LMCM
    [LMCM_Metrics{fold}, NF] = Run_LMCM(xTrain,yTrain,xTest,yTest);

    
    % JMI
    JMI_Metrics{fold} = Run_JMI(xTrain,yTrain,xTest,yTest,NF);
    
    
    % CMIM
    CMIM_Metrics{fold} = Run_CMIM(xTrain,yTrain,xTest,yTest,NF);
    
           
    % MRMR
    MRMR_Metrics{fold} = Run_MRMR(xTrain,yTrain,xTest,yTest,NF);
    
    
    % Relief
    ReliefF_Metrics{fold} = Run_ReliefF(xTrain,yTrain,xTest,yTest,NF);

    % Write the metrics to the stats files
    Write_Metrics(LMCM_Metrics{fold},f_lmcm,fold);
    Write_Metrics(JMI_Metrics{fold},f_jmi,fold);
    Write_Metrics(CMIM_Metrics{fold},f_cmim,fold);
    Write_Metrics(MRMR_Metrics{fold},f_mrmr,fold);
    Write_Metrics(ReliefF_Metrics{fold},f_relieff,fold);
    
    
    % Summary file: Write results of each method at the end of all folds
    if(fold == nfolds)
        fprintf(f_summary,'\nOrder is:\nLMCM\nJMI\nCMIM\nMRMR\nReliefF\n\n\n');
        Write_Summary_Metrics(LMCM_Metrics, f_summary, nfolds);
        Write_Summary_Metrics(JMI_Metrics, f_summary, nfolds);
        Write_Summary_Metrics(CMIM_Metrics, f_summary, nfolds);
        Write_Summary_Metrics(MRMR_Metrics, f_summary, nfolds);
        Write_Summary_Metrics(ReliefF_Metrics, f_summary, nfolds);
    end

    
    fprintf(2,'Dataset: %s\tFold %d evaluation complete.\n',datasetname, fold);

end

fclose(f_lmcm);
fclose(f_jmi);
fclose(f_cmim);
fclose(f_mrmr);
fclose(f_relieff);

end


