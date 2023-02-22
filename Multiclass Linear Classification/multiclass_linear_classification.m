%% Multiclass Linear Classification, all-vs-all
clc;
clear all;
dim = 2;
%% Create distribution for data points
%Uniform
left_boundary = -1;right_boundary = 1;
pd = makedist('Uniform','lower',left_boundary,'upper',right_boundary);
%% Make training data
data_size = 100;
% points
Xdata = random(pd,[data_size dim]);
%classification conditions
cond1 = @(x,y) 0.5*x +y < 0.1;
cond2 = @(x,y) x +y < 0.5;
% 3 classes 1 2 3
K = 3;
cond = @(x,y) 1 + cond1(x,y) + cond2(x,y);
Ydata = cond(Xdata(:,1),Xdata(:,2));
%% Learning
% hyper parameters
lambda = 0.0;
%add x0 = 1 for w0
X = [ones(1,data_size) ; Xdata']';
%weights for all classificators
W  = zeros(K,K,dim+1);

for i = 1:K
    for j = 1:i-1
        Xij = X(Ydata == i | Ydata == j,:);
        Yij = Ydata(Ydata == i | Ydata == j);
        % {i,j} -> {-1, 1}
        Yij = 2/(j-i) * Yij - (i+j)/(j-i); 
        % unknown weights
        w = zeros(1,dim+1);
        options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,...
            'MaxIterations',1000, 'MaxFunctionEvaluations', 500);
        w = fminunc(@(w) L(w,lambda,Xij,Yij), w,options);
        W(i,j,:) = w;
    end
end

%% Ploting
hold on;
grid on;
colors = ['r','g','b'];
for i = 1:K
    plot(Xdata(Ydata == i,1),Xdata(Ydata == i,2),['*' colors(i)],'MarkerSize',10)
end

detalization = 200;
v = linspace(left_boundary,right_boundary,detalization);  % plotting range
[x, y] = meshgrid(v);  % get 2-D mesh for x and y
c = zeros(detalization,detalization); %predictions
for p = 1:detalization
    for q = 1:detalization
        %voting
        votes = [];       
        for i =1:K
            for j = 1:i-1
                if [W(i,j,1) W(i,j,2) W(i,j,3)]*[1 x(p,q) y(p,q)]' > 0
                    votes = [votes j];
                else
                    votes = [votes i];
                end
            end
        end
        %counting of votes
        c(p,q) = mode(votes);
    end
end

for i=1:K
    ci = c;    
    ci(ci ~= i) = NaN;
    surf(x,y,ci,'EdgeColor','none','FaceAlpha',0.1,'FaceColor',colors(i))
end

xlim([left_boundary right_boundary]);
ylim([left_boundary right_boundary]);
 t = title('Линейная классификация');
 xlabel('Признак 1');
 ylabel('Признак 2');
 legend('Объекты 1','Объекты 2','Объекты 3','Классифицированная область объекта 1', ...
     'Классифицированная область объекта 2','Классифицированная область объекта 3','Location','southeast');