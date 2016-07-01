%Nguyen X. Vinh, Jeffrey Chan, Simone Romano and James Bailey, "Effective Global Approaches for Mutual Information based Feature Selection". 
%To appear in Proceeedings of the 20th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD'14), August 24-27, New York City, 2014. 
% (C) 2014 Nguyen Xuan Vinh   
% Email: vinh.nguyen@unimelb.edu.au, vinh.nguyenx@gmail.com
% Demo script
% Recompile the mex files in the Mex directory for your system if necessary

clear;clc;

load demo_data;

data=normalize_max(data); %normalize the data to the [-1,1] range
a=myQuantileDiscretize(data,5); %discretize the data to 5 equal-frequency bins

[QPFSFS,maxRelFS,MRMRFS,QPFS_MImat,minRedFS,MIQFS]=myQPFS(a,C);
[QPJMIFS,MRMTRFS,QPFS_JMImat,weights,CMIMFS,CIFEFS]=mySPEC_CMI(a,C);