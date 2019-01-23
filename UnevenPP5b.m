%Uneven projection pursuit on uneven simulated data
%The number in each class is used to obtain a ratio between the classes
%If the ratio isn't 1:1 then, by resampling, a cluster is inflated
%Ordinary PP is run on the new data matrix, and then the cycle repeats

%Setting a seed to be able to replicate results
randn('seed',10);

%Setting the initial parameters - easier to change things later this way
nchan=20;            %number of variables
na=100;              %number of samples in classes a and b
nb=200;
sep=6;               %separation b/w the two classes
sw=1;                %variance within the classes
sprd=10;             %spread between the other variables
snois=0.1;           %percentage of noise added to the data

%generate data - make variable of separation smaller than other variables
X0=zeros(na+nb,nchan);
x0=[randn(na,sw)-(sep*sw/2); randn(nb,sw)+(sep*sw/2)];
X0(:,1)=x0;
X0(:,2:end)=sprd*sw*randn(na+nb,nchan-1);
X1=X0+randn(size(X0))*sw*snois;

%Get t-test statistics of original data
Spi=sqrt(((na-1)*sw^2+(nb-1)*sw^2)/(na+nb-2));
Tti=sep/(Spi*sqrt((1/na)+(1/nb)));

%Plot the original data before the rotation matrix
figure(1)
clf
class=[ones(na,1);2*ones(nb,1)];
clrstr='rbkgc';
shpstr='<>^oo';
for ii=1:2
    ind=find(class==ii);
    plot(X1(ind),ind,[clrstr(ii) shpstr(ii)])
    hold on
end

%generate and apply a rotation matrix
[U,S,V]=svd(randn(nchan,nchan),'econ');
X=X1*V;

%get size of X and mean-center X
[r,c]=size(X);
Morig=ones(size(X,1),1)*mean(X);
X=X-Morig;
Xor=X;
[~,~,Vorig]=svd(X,'econ');   %Use Vorig to project into the original space

%Apply PCA and plot the separation of PCA
%plot PCA as a comparison
figure(2)
clf
[Ui,Si,Vx]=svd(X,'econ');
Tt=Ui*Si;
for ii=1:2
    ind=find(class==ii);
    plot(Tt(ind),ind,[clrstr(ii) shpstr(ii)])
    hold on
end

%Find the t-test statistics of PCA to compare to original and kPPA
var1PCA=std(Tt(1:na));
var2PCA=std(Tt(na+1:end));
x1PCA=mean(Tt(1:na));
x2PCA=mean(Tt(na+1:end));

SpPCA=sqrt(((na-1)*var1PCA^2+(nb-1)*var2PCA^2)/(na+nb-2));
TPCA=(x1PCA-x2PCA)/(SpPCA*sqrt((1/na)+(1/nb)));


%Initialize variables used in the loops
guess=500;              %number of intial guesses
p=1;                    %number of dimensions we're looking for separation in
R=nb/na;                %ratio of elements in each cluster, assuming nb>na
kurt=zeros(guess,p);    %save the final kurtosis value
maxcount=500;           %Number of times the iteration will run before it diverges
convFlag=cell(guess,p); %cell array that stores whether the code converged or diverged
oldv1=zeros(c,1);       %stores old projection vectors to check against current for convergence
oldv2=zeros(c,1);

%for loop to go through the dimensions we want to separate the data into
for numdim=1:p
    cc=c;
    vall=zeros(cc,guess);   %Place to store all of my projection vectors for sime initial guess
    convlimit=0.0075; %(0.0001)*cc;       %Convergence limit to check if the current projection converges
    [Utem,Stem,Vj]=svd(X,'econ');
    Xi=X;                   %Tried it without changing data to the scores - same as Xor now
    %Xi=X*Vj;               %Take the data and change it to the scores of the data
    Vm=zeros(cc*cc,r);      %create an empty matrix to add X'*X into - reshape later
    results=cell(100,4);    %Intialize cell array for results
    seedlist=1:500;         %seedlist to be able to replicate each initial guess
    for iguess=1:length(seedlist)
        randn('seed',seedlist(iguess));
        v=randn(cc,1);      %Initial guess of the projection vector
        v=v/norm(v);
        convkurt=kurtosis((Xi*v),1);
        convkurtbest=convkurt;
        count=0;           %intial count so it can used inside the loop to check for divergence
        conv=0;            %Setting the convergence/divergence break point
        oldv2=oldv1;
        oldv1=v;
        
        while conv==0  %only breaks under convergence or maximum iterations
            
            count=count+1;    %Updates the count so can check if max count is reached
                       
            if count<0
                t=Xi*v;
                class=[ones(na,1);2*ones(nb,1)];
                clrstr='rb';
                shpstr='><';
                figure(3)
                clf
                for ii=1:2
                    ind=find(class==ii);
                    plot(t(ind),ind,[clrstr(ii) shpstr(ii)])
                    hold on
                end
                plot([0 0], ylim,'--k')
                pause;
            end
            
            [X,sampclass]=resamp(Xi,v,na,nb,1); %function to resmaple my data - type 1 so on same side
            [r,c]=size(X);
            
            tnew=X*v;                %project data matrix onto new projection vector
            if count<0
                class1=[ones(na,1);2*ones(nb,1);sampclass];
                figure(4)
                clrstr='rb';
                shpstr='><';
                clf
                for ii=1:2
                    ind=find(class1==ii);
                    plot(tnew(ind),ind,[clrstr(ii) shpstr(ii)])
                    hold on
                end
                plot([0 0], ylim,'--k')
                pause;
            end
            
            Mat2=diag(X'*X);      %use diag because of svd to get X'*X
            Vm=zeros(cc*cc,r);
             
            for i=1:r
                tem=X(i,:)'*X(i,:);   %Gets xi'*xi and then reshapes VM
                Vm(:,i)=reshape(tem,cc*cc,1);
            end
            
            %search for a minimum
            Mat1=sum(Vm*(tnew.*tnew),2);
            Mat1=reshape(Mat1,cc,cc);
            v=inv(Mat1)*(Mat2.*v);         %Get a new projection vector
            
            %test convergence
            v=v/norm(v);                   %Scale it by it's norm and check it for convergence
            %disp(count)
            convT=Xi*v;                    %Get the scores and apply kurtosis to check for the convergence
            convkurt=kurtosis(convT,1);    %Get current kurtosis
            if count<0
                figure(5)
                plot(count,convkurt,'*k')
                hold on
            end
           if (abs(convkurt-convkurtbest))<convlimit
                convFlag(iguess,numdim)={'Converged'};
                conv=1;
%                 disp("Converged")
%                 disp(iguess)
            elseif count>maxcount
                convFlag(iguess,numdim)={'Not converged'};
                conv=1;
%                 disp("Not converged")
%                 disp(iguess)
            else                           %Check to see if the shifted algorithm should be applied - when standard isn't stable
                L1=(v'*oldv1)^2;
                L2=(v'*oldv2)^2;
                if L2>L1
                    v=v+0.5*oldv1;
                    v=v/norm(v);
                end
                oldv2=oldv1;
                oldv1=v;
                convT=Xi*v;                    %Get the scores and apply kurtosis to check for the convergence
                convkurt=kurtosis(convT,1);    %Get current kurtosis
                %disp("current kurtosis")
                %disp(convkurt)
                %disp("best kurtosis")
                %disp(convkurtbest)
                if convkurt<convkurtbest       %If the current kurtosis is lower than the previous one update it as the best kurtosis
                    convkurtbest=convkurt;
                end
            end
        end
%         figure(5)
%         pause;
%         clf
        
        %save projections of initial guesses
        vall(:,iguess)=v;
       
        %get t-test of final projection for the ith initial guess
        t=X*v;
        var1=std(t(1:na));
        var2=std(t(na+1:end));
        mnx1=mean(t(1:na));
        mnx2=mean(t(na+1:end));
        Spf=sqrt(((na-1)*var1^2+(nb-1)*var2^2)/(na+nb-2));
        Ttf=(mnx1-mnx2)/(Spi*sqrt((1/na)+(1/nb)));
        
        Xnew = resamp(Xor,v,na,nb,0);
        tfinal = Xnew*v;
        
        %Store results in an array to save and compare later            
        results{iguess,1}=kurtosis(tfinal,1);
        results{iguess,2}=tfinal;
        results{iguess,3}=[Tti TPCA Ttf];
        results{iguess,4}=t;
    end
    
    for fill=1:iguess
        kurt(fill,1)=results{fill,1};        %fill kurt variable will each value of the kurtosis from the results
    end
    [~,indx]=min(kurt(:,numdim));           %find minimum kurtosis and the projection vector that correponds to it
    Vi=vall(:,indx);
    
    %make projections orthogonal
    %     T=zeros(r,p);
    %     V=zeros(c,p);
    T(:,numdim)=Xor*Vi;
    V(:,numdim)=Vj*Vi;
    X=Xor-Xor*V*V';
    
end

%transform back into original space (orthogonal projections)
V=Vorig*V;
W=[];
P=[];
T=T+Morig*V;

%Plot the scores in original space to see the separation
class=[ones(na,1);2*ones(nb,1)];
clrstr='rb';
shpstr='<>';
figure(3)
clf
for ii=1:2
    ind=find(class==ii);
    plot(T(ind),ind,[clrstr(ii) shpstr(ii)])
    hold on
end
plot([0 0], ylim, '--k')

%Plot histogram of kurtosis values to look at statistics
figure(4)
clf
histogram(kurt,20)