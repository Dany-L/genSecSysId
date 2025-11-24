%% System
% lets use a simple example with two states, one input, one output and the
% same size of the nonlinear channel as the states
% [x^k+1, e, z] = [A, B, B2; C, D, D12; C2, D21, 0] [x^k, d, w], w =
% Delta(z)

A = [0 1; -0.5 -0.8]; % State matrix
B = [0; 1];      % Input matrix
C = [1 0];      % Output matrix
D = 0;          % Feedthrough matrix
dt = 0.1;
% A = -10; B=1; C=1;D = 0;

nx=2;nz=2;nw=nz;nd=1;ne=1;

scale= 1;
B2 = 4*ones(nx,nw); % state channel

D12 = ones(nd,nw); % output channel

% C2 = 0.1*eye(nz); % nonlinearity channel
C2 = 0* ones(nz,nx);
C2(1:nx,:) = 0.18*eye(nx);
D21 = ones(nz,nd); 
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
x = -5:0.1:5;
figure(),plot(x, dzn(x) , x, tanh(x), x, sat(x), 'LineWidth',2.0);
   
legend('dzn', 'tanh', 'sat', 'Interpreter', 'latex', 'Location', 'northeast', 'FontSize', 18);
xlabel('x');
ylabel('Function values');
title('Comparison of dzn, tanh, and sat functions');
grid on;
exportgraphics(gca, './plots/nonlinearities.png');
matlab2tikz('./plots/nonlinearities.tex')

dsys_ = struct('Ad', Ad, 'Bd', Bd, 'B2d', B2d, ...
    'Cd', Cd, 'Dd', Dd, 'D12d', D12d, ...
    'C2d', C2d, 'D21d', D21d, 'D22d', D22d, 'ne', ne, 'nl', dzn, 'dt', dt);
