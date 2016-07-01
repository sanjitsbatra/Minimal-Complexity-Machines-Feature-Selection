//Vinh Nguyen, Jeffrey Chan and James Bailey, "Reconsidering Mutual Information Based Feature Selection: A Statistical Significance View",
//Proceedings of the Twenty-Eighth AAAI Conference on Artificial Intelligence (AAAI-14), Quebec City, Canada, July 27-31 2014.
//(C) 2014 Nguyen Xuan Vinh   
//Email: vinh.nguyen@unimelb.edu.au, vinh.nguyenx@gmail.com
//Matlab Mex code for GlobalFS
//This parallel version is for used with Windows, compiled with Visual Studio
        
#include "mex.h" /* Always include this */
#include <windows.h>
#include <process.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <time.h>
#include <limits>
#include <vector>
#include <string>
#include "combination.h"
#include <cstdlib>
#include <time.h>
#include <string.h>
#include <windows.h>
#include <dos.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <conio.h>
#include <process.h>


//#define RELEASE
//#define DEBUG

using namespace std;
using namespace stdcomb;
template<class BidIt>
        void display(BidIt begin,BidIt end)
{
    for (BidIt it=begin;it!=end;++it)
        cout<<*it<<" ";
    cout<<endl;
}

struct workerParam{
    int threadID;
    int *initPos;
    int *endPos;
    int *bestPa;
    double bestScore; //best score 
    double bestMI; //best MI value (of the best scored set)
    int p;
};

typedef unsigned long long int UL;

//double log2(double x){return log(x)/log(double(2));}
double log2(double x){return log(x);}

int compare_feature_config(double *data,int nPa,int* Pa,int posi, int posj);
double conditional_MI(int **T,int *scanned, double *data,int a, int b,int nPa, int* Pa, int* n_state, int n_stateC);
void  Contingency(int**T, int Mem1,int Mem2,int n_state,int n_stateC);
double Mutu_Info(int **T, int n_state,int n_stateC);
void ClearT(int**T, int n_state,int n_stateC);			//clear the share contingency table
double compteHiDimMI(double* data,int n_state,int n_stateC);

double getij(double* A, int nrows,int i,int j){return A[i + nrows*j];}
void setij(double* A, int nrows,int i,int j,double val){A[i + nrows*j]=val;}

double myEntropy(int x,int n_state);			//calculate entropy of a vector

int findPstar(double *g_score,double g);
void updateBestPa(int * &best_Pa,int * Pa,int p);

void worker(workerParam* wParam); //worker thread, for multithreading
//helper functions
int getPosition(int* powo,int p, int * Pa);
int findLexicalIndex(int n, int p, int * Pa); //Find lexical index of a feature set, variable IDs are from 0->dim-1
double optimized_double_C(UL n,UL k);//nchoosek function double version
unsigned long long nchoosek(int n,int k);//nchoosek function
unsigned long long Factorial(int num);
void copyVector(int* src,int* &tar, int p){
    if(tar!=NULL) delete[] tar; tar=new int[p];
    for(int i=0;i<p;i++) tar[i]=src[i]; }

void printSet(int nPa,int* Pa){cout<<"[";for(int i=0;i<nPa;i++) cout<<Pa[i]<<",";cout<<"]\n";}

int findPstar(double *g_score,double g); //search for the max feature set size m*
double getPenalty(int nPa,int* Pa,int* nState, double* chi,int nStateC);

time_t time_=time(NULL);
void tic(){time_=time(NULL);}
double toc(){return difftime(time(NULL),time_);}
void cleanUp();


//Data & Data structure
int **T;
int            N = 0;				//number of samples
int			   Ne= 0;				//number of effective samples
int            dim=0;				//number of variables
double*        data=NULL;			//data matrix, pre-discretized
int*		   C=NULL;				//class variable
int*     	   n_state=0;		    //number of states, contiguously mapped within [1,n_state], i.e Matlab style
int			   max_state=0;			//max number of states of the features
int			   n_stateC=0;			//number of classes
double*		   chi=NULL;		    //chi values
double*		   g_score=NULL;		//the g-score
int            MaxDegree=0;         //max degree of the chi array

int			   nThreads=1;			//number of threads
int*		   threadStatus=NULL;	//status of each thread

double		   best_s_MIT=0;		//best adjusted independancy score
double		   best_MI=0;		    //best MI so far
int*		   best_Pa=NULL;		//best feature set overall
int**		   best_Pa_arr=NULL;	//best feature set for each thread
int			   best_nPa=0;			//cardinality of best feature set
double		   HC=0;				//entropy of the class variable

double*	       score=NULL;			//MI cache
double*		   new_score=NULL;
workerParam*   wParam;				//parameter for the worker

// I/O & timing
time_t		   Start_t=time(NULL);
time_t         End_t  =time(NULL);
double         setupTime=0;

//Parameters
double         alpha=0.95;          //significance level for MI test, in the current version, chi values are pre-supplied
int			   maxFeatures=20;		//maxFeatures preset to 20, can be set to dim


//call as: [FS]=globalFS(data,g_score,chi_arr,nthreads,alpha)
//g_score: penalty array: g_score[0]->0 feature, g_score[1]->1 feature
//chi_arr: precalculated chi values

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    n_state=0;
    n_stateC=0;
    N=0;
    dim=0;
    data=NULL;
    max_state=0;
    
    best_MI=0;
    best_s_MIT=0;
    
   
    N = mxGetM(prhs[0]); /* Get the dimensions of data */
    Ne=N;
    dim = mxGetN(prhs[0])-1;  //last col is class
    data = mxGetPr(prhs[0]);
    cout<<"N="<<N<<" Dim=" << dim<<endl;
    
    g_score= mxGetPr(prhs[1]);
    chi= mxGetPr(prhs[2]);
    MaxDegree=mxGetM(prhs[2]);
    cout<<"Max chi degree="<<MaxDegree<<endl;
    
    nThreads=(int)mxGetScalar(prhs[3]);
    cout<<"Number of threads="<<nThreads<<endl;
    alpha=mxGetScalar(prhs[4]);
    cout<<"Alpha="<<alpha<<endl;
    
//get number of state of the data
    n_state=new int[dim];
    for (int j=0;j<dim;j++) n_state[j]=0;
    for(int i=0;i<N;i++){
        for (int j=0;j<dim;j++){
            if (n_state[j]<data[i+N*j]) n_state[j]=data[i+N*j];
        }
    }
    
    max_state=1;
    for (int j=0;j<dim;j++){
        n_state[j]++;
        cout<<n_state[j]<<" ";
        if (max_state<n_state[j]) max_state=n_state[j];}
    
// printf("initial n_stateC= %d\n",n_stateC);
    for(int i=0;i<N;i++){
        if (n_stateC<data[i+(dim)*N]) {n_stateC=data[i+(dim)*N];cout<<"n_stateC= "<<(n_stateC)<<endl;}
    }
    
    n_stateC++;
    printf("Increased n_stateC= %d maxState=%d\n",(n_stateC),max_state);
    
//initializing thread data
    threadStatus=new int[nThreads];
    wParam= new workerParam[nThreads];
    for(int i=0;i<nThreads;i++){
        wParam[i].bestPa=NULL;
        wParam[i].endPos=NULL;
        wParam[i].initPos=NULL;
    }
    
    printf("Number of threads = %d ",nThreads);
    //HC=compteHiDimMI(data,n_state,n_stateC);	//MI(X;C): tighter bound ?
    HC=myEntropy(dim,n_stateC);
    printf("I(X;C)= %f ",HC);
    
    
//investigate all set from 1->P*-1 elements
    int Pstar=findPstar(g_score,2*Ne*HC);
    printf("m*= %d \n",Pstar);
    best_s_MIT=2*Ne*HC; //score of the empty set
    printf("Empty set score= %f ",best_s_MIT);
    
//co-ordinate coding: [Pai] -> p*-1 elements
    score=new double[dim]; //1-d array to store the scores of all feature combination
    new_score=NULL; //1-d array to store the scores of all feature combination
    int* ca=NULL;
    int* cb=NULL;
    
    for(int p=1;p<=min(Pstar-1,dim);p++) { //loop through feature sets of increasing size
        if(g_score[p]>=best_s_MIT){    //g_score[p]>=g_score[p-1] + 2N(HC-max(I(S_{m-1};C)))
            printf("Stopped at |Sm|= %d  g_score[p]= %f  bestSmit=%f \n",p-1,g_score[p-1],best_s_MIT);
            break;
        }
        
        //printf("m= %d, Allocating %f MB of cache\n ",p,double(nchoosek(dim,p))*sizeof(double)/(1024*1024));
        cout<<" m="<<p<<" Allocating "<< nchoosek(dim,p)<<"x"<<sizeof(double) <<" bytes = " << double(nchoosek(dim,p))*sizeof(double)/(1024*1024)<<" Mb of MI Cache."<<endl;
        
        if(p>1) {new_score=new double[nchoosek(dim,p)];}
        for(int i=0;i<nThreads;i++) {threadStatus[i]=0;}
        
        ca=new int[dim];for(int i=0;i<dim;i++) ca[i]=i;
        cb=new int[p];for(int i=0;i<p;i++) cb[i]=i;
        
        int totalSubset=nchoosek(dim,p);
        int subsetPerWorker=ceil(double(totalSubset)/nThreads);
        cout<<"Subsets per worker: "<<subsetPerWorker;
        int count=0;
        int totalCount=0;
        int isReset=0;
        
        
        int* initPos=new int[p];
        int* endPos=new int[p];
        for(int j=0;j<p;j++) {initPos[j]=cb[j];}
        int currentThread=0;
        
        
        do{  //generate all feature combination and score
            if(isReset){ for(char j=0;j<p;j++) {initPos[j]=cb[j];};isReset=0;}
            count++;
            totalCount++;
            
            if( count==subsetPerWorker || totalCount==totalSubset){ //call new worker thread
                for(int j=0;j<p;j++) {endPos[j]=cb[j];}
                //call worker here
                wParam[currentThread].threadID=currentThread;
                wParam[currentThread].p=p;
                if(wParam[currentThread].initPos!=NULL) delete[] wParam[currentThread].initPos;
                wParam[currentThread].initPos=new int[p];
                if(wParam[currentThread].endPos!=NULL) delete[] wParam[currentThread].endPos;
                wParam[currentThread].endPos=new int[p];
                if(wParam[currentThread].bestPa!=NULL) delete[] wParam[currentThread].bestPa;
                wParam[currentThread].bestPa=new int[p];
                wParam[currentThread].bestScore=best_s_MIT;
                wParam[currentThread].bestMI=best_MI;
                
                for(int i=0;i<p;i++){
                    wParam[currentThread].initPos[i]=initPos[i];
                    wParam[currentThread].endPos[i]=endPos[i];
                }
                
                _beginthread((void(*)(void*))worker, 0,&wParam[currentThread]);
                
                isReset=1;
                currentThread++;
                count=0;
            } //if
        }while(next_combination(ca,ca+dim,cb,cb+p));
        
        int N_usedThreads=currentThread--;
        cout<<"Number of threads used: "<<N_usedThreads;
        //check for all thread to stop
        bool allThreadFinished=false;
        while(!allThreadFinished){
            allThreadFinished=true;
            Sleep(1000);
            for(int i=0;i<N_usedThreads;i++){
                if(threadStatus[i]==0) allThreadFinished=false;
            }
        }
        
        //update best score
        for(int i=0;i<N_usedThreads;i++){
            if(best_s_MIT-wParam[i].bestScore>1e-12){
                best_nPa=p;
                best_s_MIT=wParam[i].bestScore;
                copyVector(wParam[i].bestPa,best_Pa,p);
                cout<<"Master Thread --- Best score " <<best_s_MIT<<" Best set: ";
                printSet(p,best_Pa);
                best_MI=wParam[i].bestMI;
                cout<<"Master Thread --- BestMI" <<best_MI<<endl;
            }
        }
        
        delete[] ca;
        ca=NULL;
        delete[] cb;
        cb=NULL;
        if(p>1) {
            delete[] score;
            score=new_score;
        }
    }// of p loop
    
//returning result
    plhs[0] = mxCreateDoubleMatrix(1, best_nPa, mxREAL); /* Create the output MI */
    double* B = mxGetPr(plhs[0] ); /* Get the pointer to the data of B */
    for(int i=0;i<best_nPa;i++) B[i]=best_Pa[i]+1; //convert back to Matlab feature index
    
    return;
}//----------------------------end of main------------------------------


void worker(workerParam* wParam){
    int p=wParam->p;
    int* ca=new int[dim];for(int i=0;i<dim;i++) ca[i]=i;
    int* endPos=new int[p];for(int i=0;i<p;i++) endPos[i]=wParam->endPos[i];
    int* cb =new int[p];for(int i=0;i<p;i++) cb[i] =wParam->initPos[i];
    int threadID=wParam->threadID;
    double bestScore=wParam->bestScore;
    
    //printf("Thread %d\n",threadID);
    cout<<"Thread "<<threadID<<endl;
    
    int **T; //local
    T=new int* [max_state];
    for (int i=0;i<max_state;i++){
        T[i]=new int[n_stateC];
    }
    int  * scanned=new int[N];
    
    ///actual work
    do  //generate all feature combination and score
    {
        bool stop=true;
        int pos=findLexicalIndex(dim,p,cb);
        //score this set of features
        if (p==1){ //only canculate the score for the 1st level
            //conditional_MI(int **T,int *scanned, int **data,int a, int b,int nPa, int* Pa, int n_state, int n_stateC);
            double CMI=conditional_MI(T,scanned,data,cb[0], dim, 0,cb,n_state,n_stateC);
            //double CMI=0;
            double d_MIT=2*Ne*(HC-CMI);
            //printf("dmit=%f\n",d_MIT);
            //cout<<"Hello"<<endl;
            
            //mexEvalString("drawnow()");
            //mexEvalString("");
            
            double penalty=getPenalty(1,cb,n_state,chi,n_stateC);
            double s_MIT=penalty+d_MIT;
            
            if(bestScore-s_MIT>1e-12){
                wParam->bestScore=s_MIT;
                wParam->bestMI=CMI;
                copyVector(cb,wParam->bestPa,p);
            }
            int pos=cb[0];
            score[pos]=CMI; //store the score
            //for(int i=0;i<p;i++) {outFile<<cb[i]<<" ";}outFile<< "score = "<< s_MIT<<endl;
        }else{
            double score_i=0;
            double penalty=getPenalty(p,cb,n_state,chi,n_stateC);
            double maxBonus=2*Ne*(HC-wParam->bestMI); //maximum bonus
            
            if (maxBonus>penalty){
                int pos=findLexicalIndex(dim,p-1,cb);
                score_i=score[pos];
                
                //calculate the last score and store
                double CMI=conditional_MI(T,scanned,data,cb[p-1],dim, p-1 ,cb,n_state,n_stateC);
                //double CMI=0;
                score_i+=CMI;
                
                //pos=getPosition(powo,p,cb);
                pos=findLexicalIndex(dim,p,cb);
                new_score[pos]=score_i; //store the last calculated score
                
                double d_MIT=2*Ne*(HC-score_i); //actual bonus
                double s_MIT=penalty+d_MIT;
                
#ifdef DEBUG
                cout<<"updating score"<<endl;
#endif
                if(bestScore-s_MIT>1e-12){
                    cout<<"Thread "<<threadID<<" bestSMIT "<<s_MIT<<" Penalty "<<getPenalty(1,cb,n_state,chi,n_stateC)<< " Best set: ";
                    printSet(p,cb);
                    bestScore=s_MIT;
                    wParam->bestScore=s_MIT;
                    wParam->bestMI=score_i;
                    copyVector(cb,wParam->bestPa,p);
                }
#ifdef DEBUG
                cout<<"end updating score"<<endl;
#endif
                //for(int i=0;i<p;i++) {outFile<<cb[i]<<" ";}outFile<< "score = "<< s_MIT<<endl;
            }//end of if (d_MIT>penalty), i.e. max bonus> current penalty
            else{
                //cout<<"Max bonus < current penalty"<<endl;
            }
        }
        
        for(int j=0;j<p;j++) if(endPos[j]!=cb[j]){stop=false;break;}
        if(stop) break; //exit when reach the last combination for this thread
        
    } while(next_combination(ca,ca+dim,cb,cb+p));
    
#ifdef DEBUG
    cout<<"clearing thread"<<endl;
#endif
    threadStatus[threadID]=1; //inform master thread that this has finished
    delete[] ca; delete[] cb; delete[] endPos;
    if (T!=NULL){
        for (int i=0;i<max_state;i++){delete T[i];}
        delete[] T;
    }
    delete[] scanned;
    
#ifdef DEBUG
    cout<<"end clearing thread"<<endl;
#endif
    cout<<"Thread "<<threadID<<" terminating..."<<endl;
    _endthread();
}///--------------------------------end of worker-----------------------


//conditional MI between node a-> node b given other feature Pa
double conditional_MI(int **T,int *scanned, double *data,int a, int b,int nPa, int* Pa, int* n_state, int n_stateC){
    double MI=0;
    //the shared contingency table
    
    
    ClearT(T,n_state[a],n_stateC);
    if (nPa==0){ //no feature
#ifdef DEBUG
        printf("Computing single MI\n");
#endif
        Contingency(T,a,b,n_state[a], n_stateC);
        return Mutu_Info(T, n_state[a], n_stateC);
    }
    else {	//with some features?
#ifdef DEBUG
        printf("Computing high dim CMI\n");
#endif
        for(int i=0;i<N;i++){scanned[i]=0;}
        for(int i=0;i<N;i++){ //scan all rows of data
            if(scanned[i]==0){  //a new  combination of Pa found
                scanned[i]=1;
                double count=1;
                ClearT(T,n_state[a], n_stateC);
                T[(int)data[i+N*a]][(int)data[i+N*b]]++;
                
                for(int j=i+1;j<N;j++){
                    if(scanned[j]==0 && compare_feature_config(data,nPa,Pa,i,j)){
                        scanned[j]=1;
                        T[(int)data[j+N*a]][(int)data[j+N*b]]++;
                        count++;
                    }
                }
                MI+=(count/N)*Mutu_Info(T,n_state[a], n_stateC);
            }
        }
#ifdef DEBUG
        cout<<"end Computing high dim CMI\n";
#endif
    }
    
    return MI;
}//------------------------end of computeCMI fast version---------------


double  compteHiDimMI(double* data,int n_state,int n_stateC){
//compute MI 	.
    int *scanned=new int[N];
    int nPa=dim-1;
    int *Pa=new int[nPa];
    for(int i=0;i<nPa;i++){ Pa[i]=i;}
    
//count number of configs
    for(int i=0;i<N;i++){scanned[i]=0;}
    int count=0;
    for(int i=0;i<N;i++){ //scan all rows of data
        if(scanned[i]==0){  //a new  combination of Pa found
            count++;
            scanned[i]=1;
            for(int j=i+1;j<N;j++){
                if(scanned[j]==0 && compare_feature_config(data,nPa,Pa,i,j)){
                    //printf("Count= %d\n",(count));
                    scanned[j]=1;
                }
            }
        }
    }
    
    printf("Count= %d\n",(count));
//build contingency table
    n_state=count;
    T=new int*[n_state];
    for(int i=0;i<n_state;i++){
        T[i]=new int[n_stateC];
        for(int j=0;j<n_stateC;j++){
            T[i][j]=0;
        }
    }
    
    for(int i=0;i<N;i++){scanned[i]=0;}
    count=-1;
    for(int i=0;i<N;i++){ //scan all rows of data
        if(scanned[i]==0){  //a new  combination of Pa found
            count++;
            scanned[i]=1;
            T[count][(int)data[i+N*(dim)]]++;
            for(int j=i+1;j<N;j++){
                if(scanned[j]==0 && compare_feature_config(data,nPa,Pa,i,j)){
                    T[count][(int)data[j+N*(dim)]]++;
                    //printf("T[%d,%d]= %d\n",count,(int)data[j+N*(dim)]-1, (T[count][(int)data[j+N*(dim)]-1]));
                    scanned[j]=1;
                }
            }
        }
    }
    
    
    count++; //as count start from -1
    return  Mutu_Info(T, count,n_stateC);
    
    delete[] Pa;
    delete[] scanned;
    for(int i=0;i<n_state;i++) delete[] T[i];
    delete T;
}


//compare a feature set configuration of node a at two position in the data: char type
int compare_feature_config(double *data,int nPa,int* Pa,int posi, int posj){
    int	isSame=1;
    for (int i=0;i<nPa;i++){ //scan through the list of features
        if(data[posi+N*Pa[i]]!=data[posj+N*Pa[i]]){//check this feature value at posi & posj
            return 0;
        }
    }
    return isSame;
}

// data[i][j] => data[i + nrows*j];
void Contingency(int** T,int a,int b,int n_state,int n_stateC){
    //printf("Computing contingency table\n");
    //cout<<"a "<<a <<" b "<<b <<" nstate "<< n_state <<" n_stateC" <<n_stateC<<endl;
    for(int i=0;i<n_state;i++)
        for(int j=0;j<n_stateC;j++)
            T[i][j]=0;
    //build table
    //printf("count\n");
    for(int i =0;i<N;i++){
        //T[data[i][a]][data[i][b]]++;
        T[(int)data[i+N*a]][(int)data[i+N*b]]++;
    }
    //printf("end\n");
}

void ClearT(int **T,int n_state,int n_stateC){
    for(int i=0;i<n_state;i++){
        for(int j=0;j<n_stateC;j++){
            T[i][j]=0;
        }
    }
}

double Mutu_Info(int **T, int n_state,int n_stateC){  //get the mutual information from a contingency table
    double MI=0;
    int *a = new int[n_state];
    int *b = new int[n_stateC];
    int N=0;
    
    for(int i=0;i<n_state;i++){ //row sum
        a[i]=0;
        for(int j=0;j<n_stateC;j++)
        {a[i]+=T[i][j];}
    }
    
    for(int i=0;i<n_stateC;i++){ //col sum
        b[i]=0;
        for(int j=0;j<n_state;j++)
        {b[i]+=T[j][i];}
    }
    
    for(int i=0;i<n_state;i++) {N+=a[i];}
    
    for(int i=0;i<n_state;i++){
        for(int j=0;j<n_stateC;j++){
            if(T[i][j]>0){
                MI+= T[i][j]*log2(double(T[i][j])*N/a[i]/b[j]);
            }
        }
    }
    delete []a;
    delete []b;
    
    return MI/N;
}


double myEntropy(int x,int n_state){
    
    double *H =new double[n_state];
    for(int i=0;i<n_state;i++) {H[i]=0;}
    
    for(int i=0;i<N;i++){
        H[(int)data[i+N*x]]++;
    }
    double e=0;
    for(int i=0;i<n_state;i++) {
        H[i]/=(N);
        if (H[i]!=0) {e-=H[i]*log2(H[i]);}
    }
    return e;
    delete []H;
}


int findLexicalIndex(int n, int p, int * Pa){
    if(p==0) return -1;
    if(p==1) return Pa[0];
    
    int pos=0;
    int last_pos=0;
    
    for(int i=0;i<p-1;i++){
        if(i==0) {last_pos=0;}
        else{
            last_pos=Pa[i-1]+1;
        }
        for(int j=last_pos;j<Pa[i];j++){
            pos=pos+nchoosek(n-(j+1),p-(i+1));
        }
        
    }//for i
    
    pos=pos+Pa[p-1]-Pa[p-2]-1;
    return pos;
    
}


unsigned long long nchoosek(int n,int k){
    unsigned long long i,temp = 1;
    if(k > (n/2))
        k = n-k;
    for(i = n; i >= (n-k+1); i--){
        temp = temp * i;
    }
    return (temp/Factorial(k));
}

double optimized_double_C(UL n,UL k){
    double answer = 1.0;
    UL i;
    if(k > (n/2))
        k = n-k;
    for(i = 0; i < k; i++){
        answer = answer * ((double)(n-i)/(double)(i+1));
    }
    return answer;
}

unsigned long long Factorial(int num) {
    unsigned long long res = 1;
    while(num > 0){
        res = res * num;
        num = num - 1;
    }
    return res;
}

int findPstar(double *g_score,double g){ //search for the max feature set size m*
    int p=0;
    while (g_score[p]<g && p<=dim){
        p++;
    }
    return p;
}

double getPenalty(int nPa,int* Pa,int* nState, double* chi,int nStateC){
#ifdef DEBUG
    cout<<"getting penalty"<<endl;
#endif
    double chival=0;
    double degree=1;
    for(int i=0;i<nPa;i++){
        degree*=nState[Pa[i]]-1;//adjustment for difference between C++ and Matlab index
    }
    degree=(degree-1)*(nStateC-1-1);//adjustment for difference between C++ and Matlab index
    if(degree>= MaxDegree){
        //fake
        degree=MaxDegree-1;
        chival= (chi[(int)degree]);
        
        //extrapolation
        double chi2=chi[MaxDegree-2],x2=MaxDegree-2;
        double chi1=chi[MaxDegree-1],x1=MaxDegree-1;
        
        chival=(chi2-chi1)/(x2-x1) * (degree-x1) + chi1;
       
        
        //cout<<"Calling chi2inv: Degree="<<degree<< " Alpha=" <<alpha<<" Chi=" << degree<<endl;
        /*mxArray* rhs[2];
         * mxArray* lhs[1];
         *
         * rhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
         * mxGetPr(rhs[0])[0]=alpha;
         *
         * rhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
         * mxGetPr(rhs[1])[0]= degree;
         *
         * lhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
         * mexCallMATLAB(1, lhs, 2, rhs, "chi2inv");
         * cout<<"Calling chi2inv: Degree="<<degree<< " Alpha=" <<alpha<<" Chi=" << mxGetPr(lhs[0])[0]<<endl;
         *
         * chival=mxGetPr(lhs[0])[0];
         *
         * mxDestroyArray(rhs[0]);
         * mxDestroyArray(rhs[1]);
         * mxDestroyArray(lhs[0]);
         */
        
    }
    else{
        chival= (chi[(int)degree]);
    }
    
#ifdef DEBUG
    cout<<"end getting penalty nPa="<<nPa<< "nState=[";
    for(int i=0;i<nPa;i++) {cout<<" {Pa="<<Pa[i]<< " s="<< nState[Pa[i]]<<"},";}
    cout<<"] degree= "<<degree<<endl;
#endif
    return(chival);
    
}