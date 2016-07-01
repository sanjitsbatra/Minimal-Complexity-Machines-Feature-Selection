% Vinh Nguyen, Jeffrey Chan and James Bailey, "Reconsidering Mutual Information Based Feature Selection: A Statistical Significance View",
% Proceedings of the Twenty-Eighth AAAI Conference on Artificial Intelligence (AAAI-14), Quebec City, Canada, July 27-31 2014.
% (C) 2014 Nguyen Xuan Vinh   
% Email: vinh.nguyen@unimelb.edu.au, vinh.nguyenx@gmail.com
% Demo script

clear;clc;

load demo_data;

data=normalize_max(data); %normalize the data to the [-1,1] range
a=myQuantileDiscretize(data,5); %discretize the data to 5 equal-frequency bins

nThreads=1; %number of parallel threads


alpha=0.99; %statistical test at 1% significance level
tic;GlobalFS_single =globalFS_single_wrapper(a,C,alpha);toc

alpha=0.99; %statistical test at 1% significance level
tic;GlobalFS =globalFS_multiThread_wrapper(a,C,alpha,nThreads);toc

alpha=0.95; %statistical test at 5% significance level
tic;GlobalFS95 =globalFS_multiThread_wrapper(a,C,alpha,nThreads);toc

%Incremental feature selection
[iSelect_fs,time]=iSelect(a,C,alpha)

%Matlab version, single threaded
%Warning: much slower!!!
alpha=0.99; %statistical test at 5% significance level
[GlobalFS95_M,best_score,time]=globalFS(a,C,alpha);

