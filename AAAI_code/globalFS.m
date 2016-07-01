%Vinh Nguyen, Jeffrey Chan and James Bailey, "Reconsidering Mutual Information Based Feature Selection: A Statistical Significance View",
%Proceedings of the Twenty-Eighth AAAI Conference on Artificial Intelligence (AAAI-14), Quebec City, Canada, July 27-31 2014.
%(C) 2014 Nguyen Xuan Vinh   
%Email: vinh.nguyen@unimelb.edu.au, vinh.nguyenx@gmail.com
%GlobalFS: globally optimal mutual information based feature selection
%Usage:[best_fs,score,time]=globalFS(a,C,alpha) 
% Input:
%       a: a data matrix, rows are samples, columns are variables
%          values of a must be discrete, taking interger values >=1;
%          all variables have the same number of categories
%       C: class variable
%       alpha: significance level for the mutual information test for
% Output:
%       best_fs: best feature set
%       best_score_ori: the optimal adjusted dependancy score
%       time: execution time

function [best_fs,best_score_ori,time]=globalFS(a,C,alpha)
tic;
[n dim]=size(a);
if nargin<3 alpha=0.95;end;  %default 5% significance 

n_state=max(a);
n_stateC=max(C);

MAX_FEATURE=min(dim,20);    %can be set to dim in the general case
chi=inf*ones(1,MAX_FEATURE); %precalculate chi values
n_state_sorted=sort(n_state); %sorting in increasing number of states
for i=1:MAX_FEATURE
  penalty(i)= chi2inv(alpha,(prod(n_state(1:i))-1)*(n_stateC-1));
end
g_score=penalty;  %the precompute penalty score
g_MIT=g_score;

Ne=n;
HC=myEntropy_ab(C) 
Pstar=sum(g_score<2*Ne*HC)+1;
fprintf('\nGlobalFS started, alpha=%f, m*=%d...\n',alpha,Pstar);       

best_s_MIT=2*Ne*HC; %best adjusted independancy score
score=zeros(1,dim); %initial 1-d array to store the scores of all single features
best_Pa=[];

for p=1:min(Pstar-1,dim) %investigate all set from 1->P*-1 elements
    if g_MIT(p)>=best_s_MIT 
        fprintf('Stopped at |Sm|=%d. ',p-1);break;
    else
        fprintf('m=%d. Allocating %dx8 bytes MI cache...\n',p,nchoosek(dim,p));
    end;
    %otherwise, find the best feature set of this cardinality value
    
    all_Pa = nchoosek([1:dim],p);   %warning: enumerate all feature sets, this is only practical for small dim <15
    %evaluate at level p and fill-in the score matrix
    [nn ll]=size(all_Pa);
    new_score_arr=zeros(1,nchoosek(dim,p));  %huge array to cache scores
    
    for j=1:nn   %loop through all potential parent set
        Pa=all_Pa(j,:);
        nPa=length(Pa);
        if nPa==1  %only canculate the score for the 1st level
            CMI=conditional_MI(a,Pa(1), C, 0);            
            d_MIT=2*Ne*(HC-CMI);
            thisPenalty= chi2inv(alpha,(prod(n_state(Pa))-1)*(n_stateC-1));
            s_MIT=thisPenalty+d_MIT;
            if  best_s_MIT-s_MIT>10^-12  %found a better score
                best_s_MIT=s_MIT;
                best_Pa=Pa;
            end
            position=findLexicalIndex(dim,Pa(1));  %position of this score in the score caching array
            score(position)=CMI;
            
        else  %get score from the cache,  and calculate score only for the last added variable
            position=findLexicalIndex(dim, Pa(1:end-1) ); %position from the cache
            score_i=score(position);
            
            %calculate the last score and store
            CMI=conditional_MI(a,Pa(end), C,Pa(1:end-1));             
            score_i=score_i+CMI;
            
            position= findLexicalIndex(dim, Pa ); %position from the cache
            new_score_arr(position)=score_i; %store
            
            d_MIT=2*Ne*(HC-score_i);
            thisPenalty= chi2inv(alpha,(prod(n_state(Pa))-1)*(n_stateC-1));
            s_MIT=thisPenalty+d_MIT;
            if best_s_MIT-s_MIT>10^-12  %found a better score
                best_s_MIT=s_MIT;
                best_Pa=Pa;
                fprintf('New best feature set={ ');for tt=1:length(best_Pa)-1 fprintf('%d ; ',best_Pa(tt));end;fprintf('%d }\n ',best_Pa(end));
            end
        end
    end %of loop through parent sets
    if p>1 score=new_score_arr;end; %update the MI cache
end % of p loop
best_fs=best_Pa;
best_score=best_s_MIT;%best adjusted independancy score
best_score_ori= (2*Ne*HC-best_s_MIT)/(2*Ne);%best adjusted dependancy score


fprintf('Best adjusted dependancy score:\t %f \n', best_score_ori);
if ~isempty(best_Pa) 
    fprintf('Best feature set={ ');for tt=1:length(best_Pa)-1 fprintf('%d ; ',best_Pa(tt));end;fprintf('%d }\n ',best_Pa(end));
else
    fprintf('There is no relevant feature! Consider reducing alpha.\n');
end;
time=toc;
end %of main function

function e=myEntropy_ab(x)
n_state=max(x);
n=length(x);
H=zeros(1,n_state);
for i=1:n H(x(i))=H(x(i))+1;end;

H=H/n;
H(find(H==0))=1;
e=-sum(H.*log(H));

end

