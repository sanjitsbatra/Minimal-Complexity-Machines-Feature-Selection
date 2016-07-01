function [xTrain2, xTest2,w, nW] = Linear_MCM_features(xTrain,yTrain,xTest,yTest,C)

%Clean up
% clc;
% clear;

%Get data

%Initialize
N=size(xTrain,1);
D=size(xTrain,2);           

%Average xTrain,xTest
for i = 1:D
    m(i) = mean(xTrain(:,i));
    s(i) = std(xTrain(:,i));
    xTrain(:,i) = (xTrain(:,i)-m(i));
    xTest(:,i) = xTest(:,i)-m(i);
end

%Solve linear program for MCM
X = [randn(D,1);randn(1,1);randn(N,1);randn(1,1)];       %[w,b,q,H]
f = [zeros(D,1);zeros(1,1);C*ones(N,1);1];
Y = yTrain*ones(1,D);

A = [ Y.*xTrain   ,   yTrain   ,   eye(N,N)   ,   -ones(N,1) ; 
     -Y.*xTrain   ,  -yTrain   ,  -eye(N)       ,   zeros(N,1) ;];
b = [zeros(N,1);-1*ones(N,1)];

lb = [-inf*ones(D,1);-inf*ones(1,1);zeros(N,1);0];
ub = [ inf*ones(D,1); inf*ones(1,1);inf*ones(N,1);inf];
options=optimset('display','off', 'Largescale', 'off', 'Simplex', 'on'); 
X = linprog(f,A,b,[],[],lb,ub,X,options);
w = X(1:D,:);
b = X(D+1,:);
q = X(D+1+1:D+1+N,:);
H = X(D+1+N+1,:);        

nW = 0;
xTrain2 = [];
xTest2 = [];
for i=1:D
    if(abs(w(i))~=0)
        xTrain2 = [xTrain2,xTrain(:,i)];
        xTest2 = [xTest2,xTest(:,i)];
        nW = nW+1;
    end
end

% fprintf(2,'D was: %f and nW is: %f \n',D,nW);



