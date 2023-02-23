%% decision tree
%% read data
file = fopen('x_data.txt','r');
Xdata = fscanf(file,'%f', [2 Inf])';
file = fopen('y_data.txt','r');
Ydata = fscanf(file,'%d');
K = length(unique(Ydata));
file = fopen('predictions.txt','r');
data = fscanf(file,'%d');
left_boundary = data(1,1); right_boundary = data(2,1);
detalization = data(3,1);
predictions = reshape(data(4:end,1),[detalization detalization]);
fclose(file);
%% Ploting
hold on;
grid on;
colors = ['r','g','b'];
for i = 1:K
    plot(Xdata(Ydata == i,1),Xdata(Ydata == i,2),['*' colors(i)],'MarkerSize',10)
end
v = linspace(left_boundary,right_boundary,detalization);  % plotting range
[x, y] = meshgrid(v);  % get 2-D mesh for x and y
for i=1:K
    ci = predictions;    
    ci(ci ~= i) = NaN;
    surf(x,y,ci,'EdgeColor','none','FaceAlpha',0.1,'FaceColor',colors(i))
end
xlim([left_boundary right_boundary]);
ylim([left_boundary right_boundary]);
 t = title('решающее дерево');
 xlabel('Признак 1');
 ylabel('Признак 2');
 legend('Объекты 1','Объекты 2','Объекты 3','Классифицированная область объекта 1', ...
     'Классифицированная область объекта 2','Классифицированная область объекта 3','Location','southeast');