%Vinh Nguyen, Jeffrey Chan and James Bailey, "Reconsidering Mutual Information Based Feature Selection: A Statistical Significance View",
%Proceedings of the Twenty-Eighth AAAI Conference on Artificial Intelligence (AAAI-14), Quebec City, Canada, July 27-31 2014.
%(C) 2014 Nguyen Xuan Vinh   
%Email: vinh.nguyen@unimelb.edu.au, vinh.nguyenx@gmail.com
%Incremental feature selection
%Usage:[best_fs,score,time]=iSelect(a,C,alpha) 
% Input:
%       a: a data matrix, rows are samples, columns are variables
%          values of a must be discrete, taking interger values >=1;
%          all variables have the same number of categories
%       C: class variable
%       alpha: significance level for the mutual information test for
%       startSet: seed
% Output:
%       best_fs: best feature set
%       time: execution time

function [best_fs,time]=iSelect(a,C,alpha,startSet)
tic;
[N dim]=size(a);
if nargin<3 alpha=0.95;end;  %default 5% significance 
if nargin<4 startSet=[];end;

r=max(a);
rC=max(C);
selected=zeros(1,dim);

fprintf('iSelect started...\n');

if isempty(startSet)   
    %select first feature as the one with max MI with C
    max_MI=0;firstFeature=1;
    for i=1:dim
        CMI=conditional_MI(a,i, C, 0);
        if CMI>max_MI
            max_MI=CMI;
            firstFeature=i;
        end
    end
    best_fs=firstFeature;
    fprintf('Adding first feature %d\n',firstFeature);
else
    best_fs=startSet;
    fprintf('Starting from seed set: ');fprintf(' %d',best_fs);
end

stop=0;
selected(best_fs)=1;

%forward selection
fprintf('Forward selection...\n');
while ~stop
    stop=1;max_inc=0;bestFeature=0;
    for i=1:dim
       if selected(i) continue;end;
       inc=2*N*conditional_MI(a,i, C,best_fs)-chi2inv(alpha,(rC-1)*(r(i)-1)*prod(r(best_fs)));            
        
       if inc>max_inc
           max_inc=inc;
           bestFeature=i;
           stop=0;
       end
    end
    if ~stop
        best_fs=[best_fs bestFeature];
        selected(bestFeature)=1;
        fprintf('Adding %d\n',bestFeature);
    end
    
end
fprintf('Best features: ');fprintf('%d ',best_fs);

time=toc;
end %main function
