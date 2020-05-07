close all; clear all; clc

cl2 = [32 50 12 10 13 36 15 12 6 6 65 70 40 9 20 34 45 40 15 15 60 80 26 18 37 50 35 12 12 25];
sh2 = [20 20 52 83 64 68 83 12 36 150 110 60 7 10 70 100 92 70 10 11 137 137 18 22 52 83 18 10 9 65];
t2 = 1845:2:1903;

t = 1845:.01:1903;
cl = abs(spline(t2, cl2, t));
sh = abs(spline(t2, sh2, t));
pop = [sh; cl];

f = hist(pop, t);

figure(1)
subplot(2,1,1), plot(t2, cl2, 'b-', t2, sh2, 'r-', 'Linewidth', [2]),
xlabel('year'), ylabel('population (in thousands)'),
legend('Canada Lynx', 'Snowshoe Hare'),
title('historical Canadian lynx and snowshoe hare population');
subplot(2,1,2), plot(t, cl, 'b-', t, sh, 'r-', 'Linewidth', [2]),
xlabel('year'), ylabel('population (in thousands)'),
legend('Canada Lynx', 'Snowshoe Hare'),
title('Canadian lynx and snowshoe hare population with interpolation');

%%

% PCA decomposition.
[u,s,v] = svd(pop);

figure(2);
subplot(3,3,1)
plot(diag(s)/sum(diag(s)), 'ro', 'LineWidth', [2]), xlabel('modes'),
title('Energy of Each Mode');

subplot(3,1,2), plot(t,v(:,1)/max(v(:,1)), t,v(:,2)/max(v(:,2)),'LineWidth',[2]),
xlabel('year');
% We actually only need the first column.

subplot(3,1,3), plot(u(:,1:2));

%%

% Develop DMD model to forecast future population states.

% DMD J-Tu decomposition.
X1 = pop(:, 1:(end-1)); X2 = pop(:, 2:end);
r = 2;
[U2, S2, V2] = svd(X1); U = U2(:,1:r); S = S2(1:r,1:r); V = V2(:,1:r);

Atilde = U'*X2*V/S;
[W,D] = eig(Atilde);
Phi = X2*(V/S)*W; % DMD modes.

dt = t(2)-t(1);
mu = diag(D);
omega = log(mu)/dt;

figure(2)
subplot(3,3,2), plot(diag(S2)/sum(diag(S2)), 'ko', 'Linewidth', [2]);
subplot(3,1,3), hold on, plot(1:1:2,real(-Phi(:,1)), 'Linewidth', [2]);
subplot(3,1,3), plot(1:1:2,real(Phi(:,2)), 'Linewidth', [2]);

u0 = pop(:,1);
y0 = Phi\u0;
tt = linspace(1845, 1963, 5801); % going twice of the time into the future.
u_modes = zeros(r, length(tt));
for i = 1:length(tt)
    u_modes(:, i) = (y0.*exp(omega*tt(i)));
end

u_dmd = Phi*u_modes;
%%

g1 = hist(100*abs(real(u_dmd(1:2,1:length(t)))),t);

%%

figure(3)
plot(t, cl, 'b-', t, sh, 'r-', 'Linewidth', [2]), hold on,
plot(tt, 100*abs(real(u_dmd(1,:))),'r--', tt, 100*abs(real(u_dmd(2,:))), 'b--', 'Linewidth', [2]),
legend('Canada Lynx', 'Snowshoe Hare', 'Snowshoe Hare - DMD','Canada Lynx - DMD'),
xlabel('year'), ylabel('population (in thousands)'),
xlim([1845 1963]);

%%

x1 = pop(1,:); x2 = pop(2,:);

H1 = [x1(1:5700)
    x2(1:5700)
    x1(2:5701)
    x2(2:5701)
    x1(3:5702)
    x2(3:5702)
    x1(4:5703)
    x2(4:5703)
    x1(5:5704)
    x2(5:5704)
    x1(6:5705)
    x2(6:5705)];

H2 = [x1(1:5700)
    x2(1:5700)
    x1(2:5701)
    x2(2:5701)
    x1(3:5702)
    x2(3:5702)
    x1(4:5703)
    x2(4:5703)
    x1(5:5704)
    x2(5:5704)
    x1(6:5705)
    x2(6:5705)
    x1(7:5706)
    x2(7:5706)
    x1(8:5707)
    x2(8:5707)
    x1(9:5708)
    x2(9:5708)
    x1(10:5709)
    x2(10:5709)];

figure(4)
[u,s,v] = svd(H1,'econ');
subplot(2,1,1), plot(diag(s)/sum(diag(s)), 'ro', 'Linewidth', [2]);
figure(5)
subplot(2,1,1), plot(u(:,1:3), 'Linewidth',[2])
subplot(2,1,2), plot(v(:,1:3), 'Linewidth',[2])
figure(7), subplot(2,1,1), plot(v(:,1:2), 'Linewidth', [2]); hold on

figure(4)
[u,s,v]=svd(H2,'econ');
subplot(2,1,2), plot(diag(s)/sum(diag(s)), 'ro', 'Linewidth', [2]);
figure(6)
subplot(2,1,1), plot(u(:,1:3), 'Linewidth',[2])
subplot(2,1,2), plot(v(:,1:3), 'Linewidth',[2])
figure(7), subplot(2,1,2), plot(v(:,1:2)), legend('H2-1','H2-2');
% Time embedding does not change that much here. But it is non-linear.
    
%%

H3 = [];
for j = 1:1800
    H3 = [H3; pop(:,j:(4000+j))];
end

%%

figure(8)
[u,s,v] = svd(H3,'econ');
subplot(3,1,1), plot(diag(s)/sum(diag(s)), 'ro', 'Linewidth', [2]);
subplot(3,1,2), plot(u(:,1:3), 'Linewidth',[2])
subplot(3,1,3), plot(v(:,1:3), 'Linewidth',[2])

%%

H1 = [];
for j = 1:1800
    H1 = [H1; pop(:,j:(4000+j))];
end
H2 = [];
for j = 2:1801
    H2 = [H2; pop(:,j:(4000+j))];
end

%%

r = 10; [u,s,v]=svd(H1, 'econ');
U = u(:,1:r); S = s(1:r,1:r); V = v(:,1:r);

Atilde = U'*H2*V/S;
[W,D] = eig(Atilde);
Phi = H2*(V/S)*W; % DMD modes.

dt = t(2)-t(1);
mu = diag(D);
omega = log(mu)/dt;

u0 = H1(:,1);
y0 = Phi\u0;

tt = linspace(1845, 1963, 5801); % going twice of the time into the future.
u_modes = zeros(r, length(tt));
for i = 1:length(tt)
    u_modes(:, i) = (y0.*exp(omega*tt(i)));
end

u_dmd = Phi*u_modes;
g2 = hist(abs(real(u_dmd(1:2,1:length(t))))/10^12, t);

%%

figure(9)

plot(t, cl, 'b-', t, sh, 'r-', 'Linewidth', [2]), hold on,
plot(tt, abs(real(u_dmd(1,:)))/10^12,'r--', tt, abs(real(u_dmd(2,:)))/10^12, 'b--', 'Linewidth', [2]),
legend('Canada Lynx', 'Snowshoe Hare', 'Snowshoe Hare - DMD','Canada Lynx - DMD'),
xlabel('year'), ylabel('population (in thousands)'),
xlim([1845 1963]);


%%

n = length(t);
xdot = zeros(2, n-2);
for i = 1:2
    for j = 2:n-1
        xdot(i, j-1) = (pop(i,j+1)-pop(i, j-1))/(2*dt);
    end
end

xs = pop(1,2:n-1).';
ys = pop(2,2:n-1).';

A = [xs xs.^2 xs.^3 ys ys.^2 ys.^3 ys.*xs ys.*xs.^2 ys.^2.*xs];

xi = A\xdot(1,:).';
yi = A\xdot(2,:).';

figure(10)
subplot(2,1,1), bar(xi),
subplot(2,1,2), bar(yi);

%%

dx = @(xs, ys) .28*xs - .29*ys;
dy = @(xs, ys) .1*xs - .29*ys;

pop1 = zeros(2, length(tt));
pop1(1:2,1) = pop(1:2,1);

for j = 2:length(tt)
    pop1(1, j) = pop1(1, j-1) + dx(pop1(1, j-1), pop1(2, j-1));
    pop1(2, j) = pop1(2, j-1) + dy(pop1(1, j-1), pop1(2, j-1));
end

g3 = hist(real(pop1(:,1:length(t))), t);

figure(10)

plot(t, cl, 'b-', t, sh, 'r-', 'Linewidth', [2]), hold on,
plot(tt, abs(real(pop1(1,:))),'r--', tt, abs(real(pop1(2,:))), 'b--', 'Linewidth', [2]),
legend('Canada Lynx', 'Snowshoe Hare', 'Snowshoe Hare - DMD','Canada Lynx - DMD'),
xlabel('year'), ylabel('population (in thousands)'),
xlim([1845 1963]);


%%

xi = lasso(A, xdot(1,:).', 'Lambda', .1);
yi = lasso(A, xdot(2,:).', 'Lambda', .1);

figure(10)
subplot(2,1,1), bar(xi),
subplot(2,1,2), bar(yi);

%%

dx = @(xs, ys) .1521*xs - .4129*ys;
dy = @(xs, ys) .004*xs.*ys - .09078*ys;

pop1 = zeros(2, length(tt));
pop1(1:2,1) = pop(1:2,1);

for j = 2:length(tt)
    pop1(1, j) = pop1(1, j-1) + dx(pop1(1, j-1), pop1(2, j-1));
    pop1(2, j) = pop1(2, j-1) + dy(pop1(1, j-1), pop1(2, j-1));
end

g4 = hist(real(pop1(:,1:length(t))), t);

figure(10)

plot(t, cl, 'b-', t, sh, 'r-', 'Linewidth', [2]), hold on,
plot(tt, abs(real(pop1(1,:))),'r--', tt, abs(real(pop1(2,:))), 'b--', 'Linewidth', [2]),
legend('Canada Lynx', 'Snowshoe Hare', 'Snowshoe Hare - DMD','Canada Lynx - DMD'),
xlabel('year'), ylabel('population (in thousands)'),
xlim([1845 1963]);

%%

xi = lasso(A, xdot(1,:).', 'Lambda', .1, 'Alpha', .8);
yi = lasso(A, xdot(2,:).', 'Lambda', .1, 'Alpha', .8);

figure(10)
subplot(2,1,1), bar(xi),
subplot(2,1,2), bar(yi);

%%

dx = @(xs, ys) .1067*xs - .4634*ys;
dy = @(xs, ys) .01429*xs + .0002432*xs.^2 + .001657*xs.*ys;

pop1 = zeros(2, length(tt));
pop1(1:2,1) = pop(1:2,1);

for j = 2:length(tt)
    pop1(1, j) = pop1(1, j-1) + dx(pop1(1, j-1), pop1(2, j-1));
    pop1(2, j) = pop1(2, j-1) + dy(pop1(1, j-1), pop1(2, j-1));
end

g5 = hist(real(pop1(:,1:length(t))), t);

figure(10)

plot(t, cl, 'b-', t, sh, 'r-', 'Linewidth', [2]), hold on,
plot(tt, abs(real(pop1(1,:))),'r--', tt, abs(real(pop1(2,:))), 'b--', 'Linewidth', [2]),
legend('Canada Lynx', 'Snowshoe Hare', 'Snowshoe Hare - DMD','Canada Lynx - DMD'),
xlabel('year'), ylabel('population (in thousands)'),
xlim([1845 1963]);


%%

% KL-divergence.

% f = f/trapz(f);
% g1 = g1/trapz(g1); g2 = g2/trapz(g2); g3 = g3/trapz(g3); 
% g4 = g4/trapz(g4); g5 = g5/trapz(g5);
% 
% 
% %%
% plot(t,f,t,g1,t,g2,t,g3,t,g4,t,g5,'Linewidth',[2]);





%%

close all; clear all; clc

bz = load("/Users/yuhongliu/Downloads/BZ.mat");
BZ_tensor = bz.BZ_tensor;

%%

% Visualize the data and reshape each snapshot into a vector.
[m,n,k]=size(BZ_tensor); % x vs y vs time data.
BZ_matrix = zeros(m*n, k);

figure(1);
%pcolor(BZ_tensor(:,:,1)), shading interp;
for j=1:k
    A=BZ_tensor(:,:,j);
    %pcolor(A), shading interp, pause(0.01);
    BZ_matrix(:,j) = reshape(A, m*n, 1);
end

%%

% Develop DMD model to forecast future population states.

% PCA decomposition.
[u,s,v] = svd(BZ_matrix,'econ');

%%

figure(2);
subplot(2,1,1)
plot(diag(s)/sum(diag(s)), 'ro', 'LineWidth', [2]), xlim([0 30]), xlabel('modes');
subplot(2,1,2), plot(v(:,1)/max(v(:,1)), 'LineWidth', [2]), hold on,
subplot(2,1,2), plot(v(:,2)/max(v(:,2)), 'LineWidth', [2]), 
xlabel('t'), legend('1st mode', '2nd mode');
figure(3)
subplot(2,1,1), pcolor(reshape(u(:,1), m, n)), shading interp;
subplot(2,1,2), pcolor(reshape(u(:,2), m, n)), shading interp;

%%

% DMD J-Tu decomposition.
X1 = BZ_matrix(:, 1:(end-1)); X2 = BZ_matrix(:, 2:end);
r = 5; % Based on the first subplot we see.
[U2, S2, V2] = svd(X1, 'econ'); U = U2(:,1:r); S = S2(1:r,1:r); V = V2(:,1:r);

Atilde = U'*X2*V/S;
[W,D] = eig(Atilde);
Phi = X2*V/S*W;

mu = diag(D);
omega = log(mu);

u0 = BZ_matrix(:,1);
y0 = Phi\u0;
u_modes = zeros(r, k);
for i = 1:k
    u_modes(:, i) = (y0.*exp(omega*i));
end

u_dmd = Phi*u_modes;

%%

figure(4)
pcolor(reshape(real(u_dmd(:,1)), m, n)), shading interp

%%
for j=1:k
    pcolor(reshape(real(u_dmd(:,j)), m, n)), shading interp, pause(0.01);
end

%%

% Time-Embedding.

H1 = [];
for j = 1:4
    H1 = [H1; BZ_matrix(:,j:(1195+j))];
end
H2 = [];
for j = 2:5
    H2 = [H2; BZ_matrix(:,j:(1195+j))];
end

%%

[u,s,v]=svd(H1, 'econ');

%%
r = 10; 
U = u(:,1:r); S = s(1:r,1:r); V = v(:,1:r);

%%

Atilde = U'*H2*V/S;
[W,D] = eig(Atilde);
Phi = H2*(V/S)*W; % DMD modes.

dt = t(2)-t(1);
mu = diag(D);
omega = log(mu)/dt;

u0 = H1(:,1);
y0 = Phi\u0;

tt = linspace(1845, 1963, 5801); % going twice of the time into the future.
u_modes = zeros(r, length(tt));
for i = 1:length(tt)
    u_modes(:, i) = (y0.*exp(omega*tt(i)));
end

u_dmd = Phi*u_modes;
g2 = hist(abs(real(u_dmd(1:2,1:length(t))))/10^12, t);

%%

figure(9)

plot(t, cl, 'b-', t, sh, 'r-', 'Linewidth', [2]), hold on,
plot(tt, abs(real(u_dmd(1,:)))/10^12,'r--', tt, abs(real(u_dmd(2,:)))/10^12, 'b--', 'Linewidth', [2]),
legend('Canada Lynx', 'Snowshoe Hare', 'Snowshoe Hare - DMD','Canada Lynx - DMD'),
xlabel('year'), ylabel('population (in thousands)'),
xlim([1845 1963]);


