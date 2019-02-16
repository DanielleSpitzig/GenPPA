%Use wood data as an indicator that it works on experimental data
load wood                                %trying the wood data with replicates first to see how that works with rkPPA

[index1,a1]=find(class_rep==1);          %isolate the first two classes of the experimental data
[index2,a2]=find(class_rep==2);

if length(a1)<=length(a2)                %Find the class that smaller - if the same size take the first class
    int=length(a2)/2;                    %Find half the size of the other class
    [newa1,idx]=datasample(a1,int,1);    
    id1=index1(idx);
    id2=index2;
    na=length(newa1);
    nb=length(a2);
    X=[X(id1,:); X(id2,:)];
else
    int=length(a1)/2;
    [newa2,idx]=datasample(a2,int,1);
    id1=index1;
    id2=index2(idx);
    na=length(newa2);
    nb=length(a1);
    X=[X(id2,:); X(id1,:)];
end

class=[ones(na,1);2*ones(nb,1)];

[r,c]=size(X);

if ceil(r/c)<10
    cv=ceil(r/10);
    [U,S,~]=svd(X,'econ');
    X=U*S;
    X=X(:,1:cv);
end

%get size of X and mean-center X
[r,c]=size(X);
Morig=ones(size(X,1),1)*mean(X);
X=X-Morig;
Xor=X;
[~,~,Vorig]=svd(X,'econ');   %Use Vorig to project into the original space

%Plot the original data 
figure(1)
clf
clrstr='rb';
shpstr='^o';
for ii=1:2
    ind=find(class==ii);
    plot(X(ind),ind,[clrstr(ii) shpstr(ii)])
    hold on
end
plot([0 0], ylim,'--k')
title('Original Data')
ylabel('Sample Number')
xlabel('')


%Apply PCA and plot the separation of PCA
%plot PCA as a comparison
figure(2)
clf
[Ui,Si,Vx]=svd(X,'econ');
Tt=Ui*Si;
% rnd=rand(length(class));
for ii=1:2
    ind=find(class==ii);
    plot(Tt(ind),ind,[clrstr(ii) shpstr(ii)])
    hold on
end
plot([0 0], ylim,'--k')
title('PCA Separation')
ylabel('Sample Number')
xlabel('Score 1')

%run Siyuan's projpursuit algorithm to check separation against mine
T=projpursuit(X,1);
figure(3)
clf
for ii=1:2
ind=find(class==ii);
plot(T(ind),ind,[clrstr(ii) shpstr(ii)])
hold on
end
plot([0 0], ylim,'--k')
title('Ordinary kPPA After Rotation')
ylabel('Sample Number')
xlabel('Score 1')