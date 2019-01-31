%Function that uses a genetic algorithm to carry out rebalanced projection pursuit
%Driscoll and Spitzig 2019

function [T,gen,pop]=OptGenPPA(X,sub,dim,rt)
%function outputs
%T - final scores outputted
%gen - the generation number of the final output
%pop - final population

%function outputs
%X - pre-processed data matrix
%sub - the number of samples in the smallest class
%dim - dimensions of separation
%rt - max count to run the algorithm

%Intialize the variables
m=size(X,1);
R=(m-sub)/sub;          %ratio of the wanted class to the rest of the data
T1=[];                  %Scores 1
T2=[];                  %Scores 2
T3=[];                  %Scores 3

Tval=zeros(sub/5,10);
results=cell(10*sub/5,2);

for pind=1:ceil(sub/5) %checking population sizes of 5,10,15,...,50 for optimization
    for cind=1:10 %Checking mutation rates of 1-10 for optimization
        popsize=pind*5;
        initpopsize=100; % Burn in population size
        popi=zeros(popsize,sub); % Preallocate initial population
        kurts=zeros(1,popsize); % Preallocate kurtosis values
             
        %convergence criteria
        maxcount = rt;
        convcount=0;
        currkurt=0;
        bestkurt=0;
        count=0;
        convcrit=0.001;       % the allowed difference between two kurtosis values to say they're the same - optimize this later
        
        for i=1:initpopsize % Generate initial population at random
            popi(i,:) = randperm(m,sub);
        end
        gen=1; % Counter
        while convcount<50 && count<maxcount
            if gen==1 % Burn-in initial population
                pop=popi; % Initial population
                parfor i=1:initpopsize % Calculate fitness of all individuals in initial population
                    X_i=X(pop(i,:),:); % Sub sample of data
                    a=0*std(X_i); % Standard deviation of data we will use to inflate (0)
                    b=1.1*mean(X_i); % Add to the center of sub sample (10% larger than true seems to solve local optima problem)
                    XClone=[X;a.*randn((R-1)*sub,1)+b]; % Generate inflated data
                    [~,~,pp]=projpursuit(XClone,1); % PP of inflated data
                    kurts(i)=pp.K; % Store kurtosis as fitness
                end
                [kurts, idx]=sort(kurts,'Ascend'); % Sort the kurtosis values and grab index vector
                pop=pop(idx(1:popsize),:); % Sorted initial population by kurtosis out to running population size
                kurts=kurts(1:popsize); % Sorted initial population by kurtosis out to running population size
                samp_elite=pop(1:2,:); % Pick out elites (picking 2 in this case... can change).
                samp_parents=pop(1:end,:); % Every individual can be a parent
            else % if gen~=1
                pop=newpop; % The running population, this is updated in each generation
                parfor i=1:popsize % Calculate fitness of all individuals in running population
                    X_i=X(pop(i,:),:); % Sub sample of data
                    a=0*std(X_i); % Standard deviation of data we will use to inflate (0)
                    b=1.1*mean(X_i); % Add to the center of sub sample (10% larger than true seems to solve local optima problem)
                    XClone=[X;a.*randn((R-1)*sub,1)+b]; % Generate inflated data
                    [~,~,pp]=projpursuit(XClone,1); % PP of inflated data
                    kurts(i)=pp.K; % Store kurtosis as fitness
                end
                [kurts, idx]=sort(kurts,'Ascend'); % Sort the kurtosis values and grab index vector
                samp_elite=pop(idx(1:2),:); % Pick out elites (picking 2 in this case... can change).
                samp_parents=pop(idx(1:end),:); % Every individual can be a parent
            end
            fitot=sum(kurts); % Calculate the total fitness of the current population
            probm=kurts./fitot; % Probability of individual to be picked as a parent
            
            % This section should be cleaned up to make sure parents arent mating with themselves...
            for i=1:size(samp_parents,1) % We now roulette-wheel select parents based on probabilty (lower kurtosis => higher probability to be a parent)
                a=min(probm); % min bound on random num
                b=max(probm); % max bound on random num
                t=(b-a).*rand + a; % Our critical value for current parent
                %%
                while true % Fill parents until break
                    for j=1:length(kurts)
                        if probm(j) < t % Less fit solutions are less likely to be picked
                            samp_parents(j,:)=pop(j,:); % Pick parent
                            break
                        end
                    end
                    break % Break the while
                end
            end
            k=1; % Counter for first parent
            i=1; % Counter for children
            while k<size(samp_parents,1) % Mate parents to create children
                p1=samp_parents(k,:); % First parent
                p2=samp_parents(k+1,:); % Second parent
                cross=randi([1 sub]); % Random point to crossover genes
                child(i,:)=p1; % Grab initial state of parent 1
                child2(i,:)=p2; % Grab initial state of parent 2
                % This section should be checked to make sure sample indices are unique in resulting children...
                for j=1:cross % Swap elemnts up to corssover point
                    if isempty(find(child(i,j)==p2(1,j),1))==1 && isempty(find(p2(1,j)==child(i,j),1))==1
                        child(i,j)=p2(1,j);
                        child2(i,j)=p1(1,j);
                    end
                end
                k=k+2;
                i=i+1;
            end
            children=[child; child2]; % Children
            for i=1:size(children,1) % Mutation step
                for j=1:size(children,2)
                    c=randi([1 100]);  % Critical value
                    if c<cind % uses the above for ind to try to optimize
                        re=randperm(m,1); % Value of mutant
                        if isempty(find(children(i,:)==re,1))==1
                            children(i,j)=re;
                        end
                    end
                end
            end
            newpop=[samp_elite; children;]; % New population for next iteration
            fitness=kurts; % Fitness of final current population
            fitness_t(gen)=median(fitness);
            fitness_tt(gen)=min(fitness);
            kst(gen)=median(fitness_t(1:gen)); % Store median for plot diagonostic
            kst_m(gen)=min(fitness_tt(1:gen)); % Store min for plot diagonostic
            bestkurt=currkurt;
            currkurt=kst_m(gen);
            figure(10)
            plot(1:gen,kst(1:gen),'r','LineWidth',2.0)
            hold on
            plot(1:gen,kst_m(1:gen),'b','LineWidth',2.0)
            xlabel('Generation number')
            ylabel('Fitness')
            drawnow
                    
            if abs(currkurt-bestkurt)<convcrit %tests if the current fitness is within a certain value from the last best fitness
                convcount=convcount+1;         %If they are count them as the same, as soon as they're not set back to 0 and start again
            else
                convcount=0;
            end
            count=count+1;                    %max count is what the person sets it, but might change to be a set value later
            gen=gen+1;
        end
        
        %after it runs through max time or reaches convergence gives an answer
        pop=samp_elite(1,:);
        X_i=X(pop,:);
        a=0*std(X_i);
        b=1.1*mean(X_i);
        XClone=[X;a.*randn((R-1)*sub,1)+b];
        [Tr,~,~]=projpursuit(XClone,1);
        if dim==1
            T1=Tr;
            T1(m+1:end)=[];
            T1=(T1-((min(T1)+max(T1))/2))/std(T1);
            % Need to make it work for multiple dimensions...
        elseif dim==2
            [Tr,Vr,pp]=projpursuit(XClone,2);
            T1=Tr(:,1);
            T2=Tr(:,2);
            T1(m+1:end)=[];
            T2(m+1:end)=[];
            T1=(T1-((min(T1)+max(T1))/2))/std(T1);
            T2=(T2-((min(T2)+max(T2))/2))/std(T2);
        end
        if dim==1 % Clean T and send back T, gen, and pop
            T(:,1)=T1;
        elseif dim==2
            T(:,1)=T1;
            T(:,2)=T2;
        end
             
        var1=std(T(1:sub));
        var2=std(T(sub+1:end));
        mnx1=mean(T(1:sub));
        mnx2=mean(T(sub+1:end));
        Spf=sqrt(((sub-1)*var1^2+((m-sub)-1)*var2^2)/(m-2));
        Ttf=abs((mnx1-mnx2)/(Spf*sqrt((1/(m-sub)+(1/sub)))));
        
        Tval(pind,cind)=Ttf;
        Cplot(cind,:)=cind;
        results{cind+10*(pind-1),1}=pop;
        results{cind+10*(pind-1),2}=currkurt;
        results{cind+10*(pind-1),3}=T;
        
    end
    Pplot(pind,:)=popsize;
end
figure(5)
clf
mesh(Pplot,Cplot,Tval)
end