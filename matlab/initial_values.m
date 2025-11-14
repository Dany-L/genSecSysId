clear, close all

nx = 2; nd=1;ne=1;nw=4;nz=nw;


L = sdpvar(nz,nx);
P = sdpvar(nx,nx);
m = sdpvar(nz,1);
lambda = sdpvar(1,1);

A_tilde = sdpvar(nx,nx);
B = sdpvar(nx,nd);
B2_tilde = sdpvar(nx,nw);

C = sdpvar(ne,nx);
D = sdpvar(ne,nd);
D12 = sdpvar(ne,nw);

C2_tilde = sdpvar(nz,nx);
D21 = sdpvar(nz,nd);
D22 = zeros(nz,nw);
% S_hat = 1;

multiplier_constraint = [];
S_hat = sdpvar(1,1);
for i=1:nw
    multiplier_constraint=[multiplier_constraint;m(i,1)>=eps];
end
M = diag(m);

lmis = [];
alpha = 0.925;
% F = [-alpha*P zeros(nx,nd) P*C2d' + L' P*Ad';
%     zeros(nd,nx) -eye(nd) D21d' Bd';
%     C2d*P+L D21d -2*M M*B2d';
%     Ad*P Bd B2d*M -P];
F = [-alpha*P zeros(nx,nd) C2_tilde' + L' A_tilde';
    zeros(nd,nx) -eye(nd) D21' B';
    C2_tilde+L D21 -2*M B2_tilde';
    A_tilde B B2_tilde -P];
lmis = lmis + (F <= -eye(nx+nd+nw+nx)*eps);
for i = 1:nz
    li = L(i,:);
    lmis = lmis + ([S_hat, li;li', P] >= eps*(eye(nx+1)));
end
lmis = lmis + multiplier_constraint;
lmis = lmis + (P>=eps*eye(nx));

sol = optimize(lmis, [], sdpsettings('solver','mosek','verbose', 0))
% sol = optimize(lmis, S_hat, sdpsettings('solver','mosek','verbose', 0))

s = sqrt(1/double(S_hat))
% s = 1
disp('max real EV of F:'); max(real(eig(double(F))))

% verify solution
Pinv = inv(double(P));
X = Pinv;
Lambda = diag(1./diag(double(M)));
H = double(L) * Pinv;


A = double(A_tilde) * X;
B = double(B);
B2 = double(B2_tilde) * Lambda;

C = double(C);
D = double(D);
D12 = double(D12);

C2 = double(C2_tilde) * X;
D21 = double(D21);

X, H, Lambda
g = 1;
sat = @(x) max(min(x, g), -g);
dzn = @(x) x-sat(x);
dt = 0.1;
dsys_ = struct('Ad', A, 'Bd', B, 'B2d', B2, ...
    'Cd', C, 'Dd', D, 'D12d', D12, ...
    'C2d', C2, 'D21d', D21, 'D22d', D22, 'ne', ne, 'nl', dzn, 'dt', dt);


%% simulate system with initial feasible variables
N = 100;
min_ = -7; max_ = -min_;
x0 = -max_ + (max_ - min_) * rand(nx, 1);

% d = sqrt(s^2*(1-alpha))*sin(linspace(0,(N-1)*dt,N));
d = sqrt(s^2*(1-alpha))*2*(rand(nd,N) - 0.5);
% d = sqrt(s^2*(1-alpha))*ones(nd,N);

% Call the simulation function
[e, x, z, w] = simulate_system(dsys_, x0, d);
plot(x(1,:), x(2,:))

