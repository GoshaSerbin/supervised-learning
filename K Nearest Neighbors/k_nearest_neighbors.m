%% weighted k_nearest_neighbors
dim = 2;
%% Create distribution for data points
%Uniform
left_boundary = -1;right_boundary = 1;
pd = makedist('Uniform','lower',left_boundary,'upper',right_boundary);
%% Make training data
data_size = 60;
% points
Xdata = random(pd,[data_size dim]);
%classification conditions
cond1 = @(x,y) x + y < 0.3;
cond2 = @(x,y) x.^2 + y < 0.6;
K = 3;% 3 classes 1 2 3
cond = @(x,y) 1 + cond1(x,y) + cond2(x,y);
Ydata = cond(Xdata(:,1),Xdata(:,2));
%% Ploting
hold on;
grid on;
colors = ['r','g','b'];
for i = 1:K
    plot(Xdata(Ydata == i,1),Xdata(Ydata == i,2),['*' colors(i)],'MarkerSize',10)
end
%k nearest neighbors
k = 2;
detalization = 100;
v = linspace(left_boundary,right_boundary,detalization);  % plotting range
[x, y] = meshgrid(v);  % get 2-D mesh for x and y
c = zeros(detalization,detalization);
gauss_kernel = @(r) exp(-2*r^2)/sqrt(20*pi);
for p = 1:detalization
    for q = 1:detalization
        votes = zeros(1,K);
        pnt = [x(p,q) y(p,q)];
        a = zeros(data_size,1);
        for i = 1:data_size
            a(i) = norm(pnt-Xdata(i,:));
        end
        for i = 1:k            
            [m, ind] = min(a);
            a(ind) = 6;
            votes(Ydata(ind)) = votes(Ydata(ind)) + gauss_kernel(m);
        end
        [m, ind] = max(votes);
        c(p,q) = ind;
    end
end
for i=1:K
    ci = c;    
    ci(ci ~= i) = NaN;
    surf(x,y,ci,'EdgeColor','none','FaceAlpha',0.1,'FaceColor',colors(i))
end

xlim([left_boundary right_boundary]);
ylim([left_boundary right_boundary]);
 t = title('метод ближайших соседей');
 xlabel('Признак 1');
 ylabel('Признак 2');
 legend('Объекты 1','Объекты 2','Объекты 3','Классифицированная область объекта 1', ...
     'Классифицированная область объекта 2','Классифицированная область объекта 3','Location','southeast');