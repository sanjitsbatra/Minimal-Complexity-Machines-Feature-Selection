function [ Metrics ] = Run_JMI( xTrain,yTrain,xTest,yTest, NF )
% This function runs JMI and produces the output metrics

    % Obtain features for JMI
    % Perform training and prediction using the selected features, using
    % the linear SVM (Based on the empirical observation that changing the 
    % kernel doesn't have much effect)

    jmi_indices = feast('jmi',NF, xTrain, yTrain); %% selecting the top features(as many as LMCM, using the jmi algorithm)
    
    model = svmtrain2(yTrain,xTrain(:,jmi_indices),'-s 0 -t 0 -q');
    [pred_labels,acc,decision_values] = svmpredict(yTest,xTest(:,jmi_indices),model,'-q');


    % Accuracy
    Accuracy = acc(1);


    % MCC(Matthew's Correlation Coefficient)
    MCC = mcc(0.5*(yTest+1), 0.5*(pred_labels+1));

    % AUC-ROC
    prob = sigmf(decision_values,[1,1]);
    AUC = auc(0.5*(yTest+1),prob);


    % Balanced Error Rate(BER)
    BER = balancedErrorRate(yTest, pred_labels);


    % F-score
    stats = F_Stats(yTest,pred_labels);
    Fs = stats.Fscore;


    % Average Precision

    % Rank predicted labels. Farther from 0, more confident the label, so
    % we use the abs(decision_values) to rank
    %     Ranked_Labels = [1:size(yTest,1)',abs(decision_values)];
    %     Ranked_Labels = -sortrows(-Ranked_Labels,2);
    %     Ranking = Ranked_Labels(:,1);

    AP = averagePrecision(yTest, pred_labels, abs(decision_values));


    Metrics = [Accuracy,MCC,AUC,BER,AP];

end




    