function [outputFile] = selectFeatures(dataset, algorithm)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Input
%      algorithm:
%       The code for the algorithm you wish to run
%       Possible Codes:
%			blogreg
%			cfs
%			chi2
%			fcbf
%			fisher
%			gini
%			infogain
%			kruskalwallis
%			mrmr
%			relieff
%			sbmlr
%			ttest
%			

%
%       dataset:
%        The code for the dataset you wish to run.
%        This will be the name of the dataset without
%        its extension.

%% Select the features based on the data set
    [outputFile] = selectFeaturesFromDataset(dataset, algorithm);