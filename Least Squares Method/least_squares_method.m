clc;

divorce=[3.6
4.5
5.0
4.3
4.1
3.8
4.4
4.5
3.5
4.2
4.1
4.0
4.1
1.5
4.8
4.6
4.0
0.8
4.9
2.8
5.1
3.5
4.6
5.7
3.0
4.4
4.4
4.1
4.3
4.6
4.8
3.6
4.8
4.4
4.2
4.3
6.0
3.6
3.3
3.6
4.7
5.3
3.1
4.2
4.6
4.7
4.4
4.3
4.4
3.9
4.2
4.8
4.4
4.2
5.0
1.7
4.3
4.5
4.3
4.7
4.1
5.6
4.6
4.6
2.6
4.5
3.7
4.0
3.5
4.4
4.2
4.2
4.7
3.5
3.9
5.2
4.7
5.5
4.6
0.8
3.1
5.6
4.1
5.4
4.1];

salary = 0.001*[23040.0
23595.0
39606.0
43089.0
27900.0
30956.0
30112.0
24919.0
32659.0
27746.0
27355.0
33647.0
28633.0
20509.0
35047.0
33370.0
23459.0
22829.0
39578.0
22368.0
33204.0
21682.0
34266.0
63677.0
21864.0
35986.0
33438.0
25974.0
26900.0
31278.0
42157.0
25358.0
25218.0
26962.0
41231.0
29355.0
69262.0
24154.0
24134.0
78111.0
47511.0
53551.0
68876.0
33133.0
29752.0
34442.0
28105.0
27777.0
23983.0
24994.0
32988.0
41180.0
23513.0
25056.0
45977.0
28928.0
29019.0
29322.0
31721.0
55938.0
25435.0
70086.0
36214.0
28886.0
23205.0
26206.0
25443.0
24363.0
32183.0
28078.0
36915.0
32574.0
42363.0
28184.0
26506.0
44182.0
33629.0
63475.0
32759.0
20525.0
24315.0
80796.0
57425.0
84214.0
30709.0];

data = [salary';divorce'];
plot(salary,divorce,'*');
hold on;

%аппроксимация линейной функцией
polycoeffs1 = polyfit(data(1,:),data(2,:),1);
error = sum((data(2,:) - polyval(polycoeffs1,data(1,:))).^2);
error = sqrt(error/length(data));
fprintf('y=%fx+%f, error = %f\n',polycoeffs1(1),polycoeffs1(2),error);

%аппроксимация полиномом третьей степени
polycoeffs3 = polyfit(data(1,:),data(2,:),3);
error = sum((data(2,:) - polyval(polycoeffs3,data(1,:))).^2);
error = sqrt(error/length(data));
fprintf('y=%fx^3+%fx^2+%fx+%f, error = %f\n',polycoeffs3(1), polycoeffs3(2), polycoeffs3(3),polycoeffs3(4),error);

%модель
model = @(params,x) params(1)- params(2)*1./(x+params(3));

%функционал ошибок
errorFunc = @(params,data) sum((data(2,:)-model(params,data(1,:))).^2);

options = optimset('TolX',10^(-6),'TolFun',10^(-6),'MaxFunEvals',10000);

[params,error] = fminsearch(@(p) errorFunc(p,data), [5 10 10], options);
fprintf('y =%f - %f/(x+%f) error = %f\n',params(1),params(2),params(3),sqrt(error/length(data)));

a = min(data(1,:));
b = max(data(1,:));
x=linspace(a,b,100);
plot(x,polyval(polycoeffs1,x),x,polyval(polycoeffs3,x),x,model(params,x))
t = title({'Зависимость количества разводов','от размера зарплаты'});
t.FontSize = 12;
xlabel({'Средняя зарплата', 'тыс. руб.'},'FontSize',12);
ylabel('Число разводов на 1000 человек','FontSize',12);
legend('Данные по субъектам РФ (2017)','Линейная аппроксимация','Полином 3-го порядка', ...
    '$p_1 - \frac{p_2}{x+p_3}$','Interpreter','latex','Location','southeast', ...
    'FontSize',12);
grid on;

