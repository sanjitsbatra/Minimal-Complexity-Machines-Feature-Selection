function Write_Metrics(Metrics,f_handle,fold)
% This function writes the Metrics to the file handle

fprintf(f_handle,'\n%f, %f, %f, %f, %f\n',Metrics(1),Metrics(2),Metrics(3),Metrics(4),Metrics(5));

end

