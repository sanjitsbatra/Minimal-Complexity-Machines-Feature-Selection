%GlobalFS: globally optimal mutual information based feature selection
%(C) 2010-2011 Nguyen Xuan Vinh   
%Email: n.x.vinh@unsw.edu.au, vinh.nguyenx@gmail.com
%
%Usage: CMI=conditional_MI(a,Xi, C, Pa,n_state)
%Calculating the conditional mutual information I(Xi;C|Pa)
% Input:
%       a: a data matrix, rows are samples, columns are features
%          values of a must be discrete, taking interger values >=1
%       Xi: feature index
%       C: class vector
%       Pa: other feature, to condition on. Set Pa=0 for unconditioned MI
% Output:
%       CMI: the conditional mutual information I(Xi;C|Pa)

function CMI=conditional_MI(a,Xi, C, Pa)
[n dim]=size(a);


scanned=zeros(1,n);  %mark the scanned row of a
prob_Pa=[];   %probability of each Pa configuration encountered
MI_arr=[];    %the corresponding MI value

Ne=n; 
nPa=length(Pa);

if Pa==0  %calculate the unconditional MI
   Cont=Contingency(a(:,Xi),C);
   CMI=Mutu_info(Cont);
   return;
end

for i=1:n  %scan all rows of a, find a new combination of Pa, and construct the corresponding contingency table
    if scanned(i)==0 %found a new combination of Pa       
        count=1;  %the number of times this combination appears in a
        scanned(i)=1;
        T=zeros(max(a(:,Xi)),max(C));
        T(a(i,Xi),C(i))=T(a(i,Xi),C(i))+1;
        for j=i+1:n
            if (scanned(j)==0 && sum(a(i,Pa)==a(j,Pa))==nPa)
                scanned(j)=1;
                T(a(j,Xi),C(j))=T(a(j,Xi),C(j))+1;
                count=count+1;
            end
        end
        MI_arr=[MI_arr Mutu_info(T)];
        prob_Pa=[prob_Pa count];
    end
end

prob_Pa=prob_Pa/sum(prob_Pa);  %normalize the probability 
CMI=sum(prob_Pa.*MI_arr);


function MI=Mutu_info(T)
%calculate the MI 
[r c]=size(T);
a=sum(T');
b=sum(T);
N=sum(a);

MI=0;
for i=1:r
    for j=1:c
        if T(i,j)>0 MI=MI+T(i,j)*log(T(i,j)*N/(a(i)*b(j)));end;
    end
end
MI=MI/N;

function Cont=Contingency(Mem1,Mem2)
Cont=zeros(max(Mem1),max(Mem2));
for i = 1:length(Mem1);
   Cont(Mem1(i),Mem2(i))=Cont(Mem1(i),Mem2(i))+1;
end