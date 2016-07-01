%Vinh Nguyen, Jeffrey Chan and James Bailey, "Reconsidering Mutual Information Based Feature Selection: A Statistical Significance View",
%Proceedings of the Twenty-Eighth AAAI Conference on Artificial Intelligence (AAAI-14), Quebec City, Canada, July 27-31 2014.
%(C) 2014 Nguyen Xuan Vinh   
%Email: vinh.nguyen@unimelb.edu.au, vinh.nguyenx@gmail.com
%Matlab wrapper for the Matlab mex parallel GlobalFS
%Input:
%   -data: matrix of samples x features
%   -C   : class labels of samples
%   -alpha: significance threshold
%   -nthreads: number of parallel threads

function best_FS =globalFS_multiThread_wrapper(data,C,alpha,nthreads)

n_state=max(data);
n_state=sort(n_state);
n_stateC=max(C);

[n dim]=size(data);
MAX_FEATURE=min(dim, 20);

penalty=zeros(1,MAX_FEATURE+1);
for i=1:MAX_FEATURE
  penalty(i+1)= chi2inv(alpha,(prod(n_state(1:i))-1)*(n_stateC-1));
end

filename=['./iSelectChiValue' num2str(alpha) '.bin'];
fid=fopen(filename, 'r');
MAX_DEG=500000;

if fid<0  %only create new file if not exist
    fprintf('Creating Chi lookup-table for the first time, may take time...\n');
    n_state=max(data);    
    max_degree=min(MAX_DEG,prod(n_state));
    chi=zeros(max_degree+1,1);
    chi(1)=MAX_DEG;
    for i=1:max_degree
        chi(i+1)= chi2inv(alpha,i);
    end
    
    fid = fopen(filename, 'w');
    fwrite(fid, chi, 'double');
    fclose(fid);
    fprintf('Chi file writing done...\n');
end


fid=fopen(filename);
chi=fread(fid,MAX_DEG,'double');
chi(1)=0;

best_FS =globalFS_multiThread([data,C],penalty,chi,nthreads,alpha);