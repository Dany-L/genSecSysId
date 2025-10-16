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

%% simulation
N = 1000;
d = zeros(nd,N); % no input

% Call the simulation function
[e, x] = simulate_system(dsys_, [0;0], d);

t = linspace(0,(N-1)*dt, N);
plot(t,e)

%% Analysis
% standard sector
eps = 1e-4;
disp('--- STANDARD SECTOR CONDITIONS ---')
a = 0; b = 1; %lower and upper bound of sector condition
P_r = [-eye(nw) b*eye(nw); eye(nw) -a*eye(nw)];

% make the multiplier variable
Pm =@(L) P_r' * [zeros(nw,nw), L'; L, zeros(nw,nw)] * P_r;

multiplier_constraint = [];
lambda = sdpvar(nz,1);
for i=1:nw
    multiplier_constraint=[multiplier_constraint;lambda(i,1)>=eps];
end
Lambda = diag(lambda);
X = sdpvar(nx,nx);

L1 = [eye(nx), zeros(nx,nw);
    Ad, B2d];
L3 = [zeros(nz,nx), eye(nw);
    C2d, D22d]; 
lmis = [];
% lmi = L1' * [-X, zeros(nx,nx); zeros(nx,nx), X] * L1 + ...
%     L3' * Pm(Lambda) * L3;
lmi = [-X C2d'*Lambda Ad'*X;
    Lambda*C2d -2*Lambda B2d'*X;
    X*Ad X*B2d -X];
lmis = lmis + (lmi <= -eye(3*nx)*eps);
lmis = lmis + multiplier_constraint;

% no objective, just looking for a feasible solution
sol = optimize(lmis, [], sdpsettings('solver','mosek','verbose', 0))
disp('max real EV of LMI:'); max(real(eig(double(lmi))))
disp('X:'); double(X)
disp('Lambda:'); double(Lambda)


% generalized sector
disp('--- GENERALIZED SECTOR CONDITIONS ---')

L = sdpvar(nz,nx);
P = sdpvar(nx,nx);
m = sdpvar(nz,1);
for i=1:nw
    multiplier_constraint=[multiplier_constraint;m(i,1)>=eps];
end
M = diag(m);

lmis = [];
F = [-P P*C2d' + L' P*Ad';
    C2d*P+L -2*M M*B2d';
    Ad*P B2d*M -P];
lmis = lmis + (F <= -eye(3*nx)*eps);
for i = 1:nz
    li = L(i,:);
    lmis = lmis + ([1, li;li', P] >= eps*(eye(nx+1)));
end
lmis = lmis + multiplier_constraint;
lmis = lmis + (P>=eps*eye(nx));

sol = optimize(lmis, [], sdpsettings('solver','mosek','verbose', 0))
max(real(eig(double(F))))

% verify solution
Pinv = inv(double(P));
X = Pinv;
H = double(L)*Pinv;
Lambda = diag(1./diag(double(M)));
lmi = [-X (C2d'+H')*Lambda Ad'*X; ...
    Lambda*(C2d+H) -2*Lambda B2d'*X; ...
    X*Ad X*B2d -X];
disp('max real EV of LMI:'); max(real(eig(lmi)))

X, H, Lambda

for i =1:nz
    hi = H(i,:);
    assert(max(real(eig([1 hi;hi' X])))>=0)
end


%% plot the ellipsoid, any initial condition in this ellipsoid is guaranteed to be exponentially stable
% thus this is the invariant set under the system dynamics
% starting from this initial condition the state will never leave the set
min_ = -1; max_ = -min_; num_samples = 200;

theta = linspace(0, 2*pi, num_samples); % angle values
unit_circle = [cos(theta); sin(theta)]; % points on unit circle

% compute X^(-1/2)
[V, D] = eig(X);
X_half_inv = V * diag(1 ./ sqrt(diag(D))) * V';

ellipse = X_half_inv * unit_circle; % transform the circle
figure; hold on; axis equal;
plot(ellipse(1,:), ellipse(2,:), 'b-', 'LineWidth', 2);
xlabel('x_1'); ylabel('x_2'); grid on, hold on
legend('$x^T P x < 0$', 'Interpreter', 'latex', 'Location', 'Best');

% Generate all sign combinations for s ∈ {-1,1}^2
S = [-1 -1; -1 1; 1 1; 1 -1];

% Compute vertices
X_ = (H \ S')';  % equivalent to inv(H)*s'

% Close the polygon by repeating the first vertex
X_ = [X_; X_(1,:)];

% Plot the edges
plot(X_(:,1), X_(:,2), 'r-', 'LineWidth', 2);
% scatter(X(:,1), X(:,2), 60, 'r', 'filled');
xlabel('x_1'); ylabel('x_2');


counter  = 0; M = 500;
% lets plot some trajectories
for i=1:M
    % Generate a random initial condition within the range [-5, 5]
    x0 = -max_ + (max_ - min_) * rand(nx, 1);

    N = 1000;
    d = zeros(nd,N); % no input
    
    % Call the simulation function
    [e, x] = simulate_system(dsys_, x0, d);
    
    if x0' * X * x0 > 1
        plot(x0(1,1), x0(2,1), 'x')
        plot(x(1,1:5), x(2,1:5))
        counter = counter +1;
        continue
    end
    plot(x(1,:), x(2,:))

end
fprintf('from %i samples %i where not feasible\n', M, counter)
legend({'$x^T P x < 0$', '$\|H x\|_\infty < 1$'}, 'Interpreter', 'latex', 'Location', 'northeast');


% Export the figure to a PDF
print(gcf, './matlab/plots/invariance.pdf', '-dpdf');



