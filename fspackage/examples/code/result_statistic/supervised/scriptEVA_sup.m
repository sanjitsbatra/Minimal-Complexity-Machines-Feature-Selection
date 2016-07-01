function scriptEVA_sup(fileName, algorithmName, varargin)

fileDir = cell(3,1);

if length(varargin) < 1
    dataPath = ['..' filesep '..' filesep '..'  filesep 'data' filesep];
end

for i = 1: length(fileName)
   fileDir{1} = strcat(dataPath,fileName{i},'_',algorithmName,'_result.mat');
   fileDir{2} = strcat(dataPath,fileName{i},'.mat');
   fileDir{3} = strcat(dataPath,fileName{i},'_part.mat');
%    try
        eva = expFun_eva_sup(fileDir, fileName{i});
        save(strcat(dataPath, fileName{i}{1},'_acc_result', '.mat'),'eva');
%    catch ME
%        fprintf('there is an error caught for %s data\n',fileName{i});
%        disp(ME.message);
%        ME.stack(1)
%    end
end

end