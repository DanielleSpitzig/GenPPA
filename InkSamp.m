%%Just a quick check to see if it works on experimental data

load AllInkUsed
class=Inx;
X=All;
[r,c]=size(X);

if ceil(r/c)<10
    cv=ceil(r/10);
    [U,S,~]=svd(X,'econ');
    X=U*S;
    X=X(:,1:cv);
end

ntotal=length(class);
n1=find(class==1);
na=length(n1);
nb=ntotal-na;

%Plot the original data 
figure(1)
clf
clrstr='rbkgcrbkgc';
shpstr='<>^<>ooooo';
for ii=1:10
    ind=find(class==ii);
    plot(X(ind),ind,[clrstr(ii) shpstr(ii)])
    hold on
end
plot([0 0], ylim,'--k')
title('Original Data')
ylabel('Sample Number')
xlabel('')

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
% rnd=rand(length(class));
for ii=1:10
    ind=find(class==ii);
    plot(Tt(ind),ind,[clrstr(ii) shpstr(ii)])
    hold on
end
plot([0 0], ylim,'--k')
title('PCA After Rotation')
ylabel('Sample Number')
xlabel('Score 1')

%run Siyuan's projpursuit algorithm to check separation against mine
T=projpursuit(X,1);
figure(3)
clf
for ii=1:7
ind=find(class==ii);
plot(T(ind),ind,[clrstr(ii) shpstr(ii)])
hold on
end
plot([0 0], ylim,'--k')
title('Ordinary kPPA After Rotation')
ylabel('Sample Number')
xlabel('Score 1')
