%% Binary Linear Classification
clc;
clear all;
dim = 2;
%% Create distribution for data points
%Normal
mu = 0;sigma = 0.5;
pd = makedist('Normal','mu',mu,'sigma',sigma);
%Uniform
left_boundary = -1;right_boundary = 1;
%pd = makedist('Uniform','lower',left_boundary,'upper',right_boundary);

%% Make training data
data_size = 100;

% points
Xdata = random(pd,[data_size dim]);
%classification conditions
cond1 = @(x,y) x < y;
cond2 = @(x,y) 20*x < y;
% multiply to keep only the common points
cond = @(x,y) cond1(x,y) & cond2(x,y);
%classes 0 1
Ydata = cond(Xdata(:,1),Xdata(:,2));

%% Learning

%add x0 = 1 for w0
X = [ones(1,data_size) ; Xdata']';
% {0,1} -> {-1, 1}
Y = 2*Ydata - 1;

% hyper parameters
lambda = 0.01;

% unknown weights
w = zeros(1,dim+1);

%error function
L = @(w) lambda * norm(w(2:end))^2 + sum(max(0,1-Y.*(X*w')));
options = optimset('TolX',10^(-6),'TolFun',10^(-6),'MaxFunEvals',10000);
[w,error] = fminsearch(@(w) L(w), w, options)

%% Ploting
plot(Xdata(Ydata,1),Xdata(Ydata,2),'*r','MarkerSize',10)
hold on;
grid on;
plot(Xdata(~Ydata,1),Xdata(~Ydata,2),'*b','MarkerSize',10)
x = linspace(left_boundary,right_boundary);
plot(x,-(w(1) + x.*w(2))./w(3))
v = linspace(left_boundary,right_boundary,500);  % plotting range
[x, y] = meshgrid(v);  % get 2-D mesh for x and y
c1 = double(cond(x,y));
c1(c1 == 0) = NaN;
surf(x,y,c1,'EdgeColor','none','FaceAlpha',0.1,'FaceColor','r')
xlim([left_boundary right_boundary]);
ylim([left_boundary right_boundary]);
t = title('Линейная классификация');
t.FontSize = 12;
xlabel('Признак 1','FontSize',12);
ylabel('Признак 2','FontSize',12);
legend('Объекты 1','Объекты 2','Обученная модель', ...
    'Истинная область класса 1','Interpreter','latex','Location','southeast', ...
    'FontSize',12);



