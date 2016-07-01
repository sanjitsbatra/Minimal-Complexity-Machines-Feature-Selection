function res = scriptStat_sup(fileName, algorithm, filePath)

fileDir = cell(1,1);

for i = 1: length(fileName)
   fileDir{1} = strcat(filePath, fileName{i},'_acc_result.mat');

   try
        res = stat_supRes(fileDir);
        
        save(strcat(fileName{i},'_', algorithm, '_stat_result', '.mat'),'res');
   catch ME
       fprintf('An error was caught for %s data\n',fileName{i});
       disp(ME.message);
       ME.stack(1)
   end
end

end