function [Xnew,sampclass] = resamp(X,v,na,nb,type)
%function to resample my data matrix
%returns the new data matrix with the resampled points added and their coresponding classes
%input the data matrix to resample, the projection vector, the class sizes, and the type of resampling
%type=0 means take the smaller side to resmaple
%type=1 means take the same side regardless of size

R=nb/na;                             %like in PPUneven it's assumed that nb>=na
t=X*v;                               %project the data onto v and check position

indx1=find(t<0);                     %if negative it's A, if positive or 0 it's B
indx2=find(t>=0);
A=X(indx1,:);                        %gets the corresponding values in X
B=X(indx2,:);

Na=size(A,1);                        %Find the sizes of each set where X is positive or negative
Nb=size(B,1);

if type==0                           %if type is inputted as 0 then the smallest size is resampled
    if Na<=Nb                        %checks which set is smaller to resample, if same size resample A
        Nc=(R-1)*Na;
        Nc=ceil(Nc);
        [Ap,idx]=datasample(A,Nc,1); %selecting Nc random values from A
        id=indx1(idx);               %id is the index of where the values of A came from in the data matrix
    else
        Nc=(R-1)*Nb;
        Nc=ceil(Nc);
        [Ap,idx]=datasample(B,Nc,1); %selecting Nc random values from B
        id=indx1(idx);               %id is the index of where the values of B came from in the data matrix
    end
elseif type==1                       %if type is inputted as 1 then the same side gets resmapled regradless of size
    Nc=(R-1)*Na;
    Nc=ceil(Nc);
    [Ap,idx]=datasample(A,Nc,1);     %selecting Nc random values from A
    id=indx1(idx);                   %id is the index of where the values of A came from in the data matrix
else
    disp("Please input a 0 or 1 for the type")
end

sampclass=zeros(length(Ap(:,1)),1);   %sampclass is a vector that contains the class of the resampled data
count1=find(id<=na);                  %finds which class the resampled data belongs to
count2=find(id>na);
sampclass(count1,1)=1;                %uses the index in find to create the filled sampclass
sampclass(count2,1)=2;

if isempty(Ap)==0                     %If the ratio isn't 1:1 then add the new samples into X
    Xnew=[X; Ap];
else                                  %If the ratio is 1:1 just take it as the previous X with no new samples - regular kPPA
    Xnew=X;
end
end