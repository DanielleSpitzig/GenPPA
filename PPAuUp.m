% Function to carry out rebalanced projection pursuit (Not sure what to call it)
% Steve Driscoll 2018

function [T,gen,pop] = PPAuUp(X,sub,dim,rt)
% Init
m=size(X,1);
T1=[]; % Scores 1
T2=[]; % Scores 2
T3=[]; % Scores 3
popsize=10; 
initpopsize=100; % Burn in population size
popi=zeros(popsize,sub); % Preallocate initial population
kurts=zeros(1,popsize); % Preallocate kurtosis values
for i=1:initpopsize % Generate initial population at random
    popi(i,:) = randperm(m,sub);
end
gen=1; % Counter
    while true % Wait on break
        if gen==1 % Burn-in initial population
            pop=popi; % Initial population
            parfor i=1:initpopsize % Calculate fitness of all individuals in initial population
                X_i=X(pop(i,:),:); % Sub sample of data
                a=0*std(X_i); % Standard deviation of data we will use to inflate (0)
                b=1.1*mean(X_i); % Add to the center of sub sample (10% larger than true seems to solve local optima problem)
                XClone=[X;a.*randn((m-sub)-sub,1)+b]; % Generate inflated data
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
                XClone=[X;a.*randn((m-sub)-sub,1)+b]; % Generate inflated data
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
            while true % Fill parents until break
                for j=1:length(kurts)
                    if probm(j) < t % Less fit solutions are less likely to be picked
                        samp_parents(j,:)=pop(j,:); % Pick parent
                        break
                    end
                end
                break % Break the while (might be redundant, not sure)
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
% This section should be checked to makre sure sample indices are unique in resulting children...
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
                if c<2 % 2% mutation (can change this... helps avoid local optima/keep diversity)
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
        plot(1:gen,kst(1:gen),'r','LineWidth',2.0)
        hold on
        plot(1:gen,kst_m(1:gen),'b','LineWidth',2.0)
        xlabel('Generation number')
        ylabel('Fitness')
        drawnow
% Need to add real convergence test... check if last x generations have same best population...
        if gen>rt % Convergence test
            pop=samp_elite(1,:);
            X_i=X(pop,:);
            a=0*std(X_i);
            b=1.1*mean(X_i);
            XClone=[X;a.*randn((m-sub)-sub,1)+b];
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
            break
        end
        gen=gen+1;
    end
if dim==1 % Clean T and send back T, gen, and pop
    T(:,1)=T1;
elseif dim==2
    T(:,1)=T1;
    T(:,2)=T2;
end
end
