function Write_Summary_Metrics(Collated_Metrics,f_handle, nfolds)
% This function writes the Summary Metrics to the file handle

% Format is: Accuracy, MCC, AUC(ROC), BER, Average Precision


for fold = 1:nfolds

    Metrics = Collated_Metrics{fold};
    
    Acc(fold) = Metrics(1);
    MCC(fold) = Metrics(2);
    AUC(fold) = Metrics(3);
    BER(fold) = Metrics(4);
    AP(fold) = Metrics(5);
end

fprintf(f_handle,'\n%f+-%f, %f+-%f, %f+-%f, %f+-%f, %f+-%f\n',mean(Acc),std(Acc),mean(MCC),std(MCC),mean(AUC),std(AUC),mean(BER),std(BER),mean(AP),std(AP));


end

