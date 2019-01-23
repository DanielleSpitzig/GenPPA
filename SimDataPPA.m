%A script to run to generate data for the purposed of my simulations

%Define all variables that would need to be changed
nchan=3;            %number of variables
na=10;              %number of samples in classes a and b
nb=20;
sep=4;               %separation b/w the two classes
sw=1;                %variance within the classes
sprd=10;             %spread between the other variables
snois=0.1;           %percentage of noise added to the data

%
X0=zeros(na+nb,nchan);
x0=[randn(na,sw)-(sep*sw/2); randn(nb,sw)+(sep*sw/2)];
X0(:,1)=x0;
X0(:,2:end)=sprd*sw*randn(na+nb,nchan-1);
X1=X0+randn(size(X0))*sw*snois;

%Apply rotation to the data
[U,S,V]=svd(randn(nchan,nchan),'econ');
X=X1*V;

%get size of X and mean-center X
[r,c]=size(X);
Morig=ones(size(X,1),1)*mean(X);
X=X-Morig;
Xor=X;

%define a class vector to use while plotting
class=[ones(na,1); 2*ones(nb,1)];