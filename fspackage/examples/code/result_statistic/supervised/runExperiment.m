function result = runExperiment(algorithm, dataset, plot)
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
%			spectrum
%
%       dataset:
%        The code for the dataset you wish to run
%        Possible Codes:
%           orl
%           rel
%
%       plot:
%        boolean value denoting whether you wish to plot the ensuing
%        results

%% Get the name of the algo/dataset from the command line if none was
%% passed in
if nargin == 0
    dataset = getenv('fsDataset');
    algorithm = getenv('fsAlgorithm');
    
    %% Make sure load_fspackage has been run
    myDir = pwd;
    cd( [pwd filesep '..' filesep '..' filesep '..' filesep '..' filesep ]);
    load_fspackage();
    cd(myDir);
end

if nargin < 3
    plot = false;
end

%% Select the features based on the dataset and algorithm
outputFile = selectFeatures(dataset, algorithm);

%% Calculate evaluations and the statistics
result = runEvaluations(outputFile, algorithm);

%% Call the plot of the data    
    if(plot)
        writeXLS(result, algorithm, outputFile);
    end
end