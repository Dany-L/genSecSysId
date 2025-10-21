%% System
% lets use a simple example with two states, one input, one output and the
% same size of the nonlinear channel as the states
% [x^k+1, e, z] = [A, B, B2; C, D, D12; C2, D21, 0] [x^k, d, w], w =
% Delta(z)

A = [0 1; -0.5 -0.8]; % State matrix
% A = rand(2,2);
B = [0; 1];      % Input matrix
C = [1 0];      % Output matrix
D = 0;          % Feedthrough matrix
dt = 0.1;

nx=2;nz=2;nw=nz;nd=1;ne=1;

scale= 2;
% B2 = scale*ones(nx,nw);
B2 = scale*eye(nx);
% C2 = scale*ones(nz,nx);
C2 = scale*eye(nz);
D12 = scale*ones(nd,nw);
D21 = scale*ones(nz,nd);
D22 = zeros(nz,nw); % at this point must be zero

disp('eig A:'); eig(A)
sys = ss(A, [B, B2], [C;C2], [D, D12;D21, D22]);
dsys = c2d(sys, dt);
Ad = dsys.A; Bd =dsys.B(1:nx,1:nd); B2d = dsys.B(1:nx,nd+1:end);
Cd = C; Dd = D; D12d=D12;
C2d = C2; D21d = D21; D22d = D22;
disp('| eig Ad |'); abs(eig(Ad))


%% nonlinearity
% Define the deadzone nonlinear function with variable bound g
g = 1; % Set the bound for the deadzone
sat = @(x) max(min(x, g), -g);
dzn = @(x) x-sat(x);
% x = -5:0.1:5;
% figure(),plot(x, dzn(x), x, tanh(x), x, sat(x));
% legend('dzn', 'tanh', 'sat');
% xlabel('x');
% ylabel('Function values');
% title('Comparison of dzn, tanh, and sat functions');
% grid on;

dsys_ = struct('Ad', Ad, 'Bd', Bd, 'B2d', B2d, ...
    'Cd', Cd, 'Dd', Dd, 'D12d', D12d, ...
    'C2d', C2d, 'D21d', D21d, 'D22d', D22d, 'ne', ne, 'nl', dzn, 'dt', dt);
