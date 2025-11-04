clear all, close all
% load system
run('shared.m')

%% simulation
N = 1000;
d = zeros(nd,N); % no input

% Call the simulation function
[e, x] = simulate_system(dsys_, [0;0], d);

t = linspace(0,(N-1)*dt, N);
% plot(t,e)

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
lmi = L1' * [-X, zeros(nx,nx); zeros(nx,nx), X] * L1 + ...
    L3' * Pm(Lambda) * L3;
% lmi = [-X C2d'*Lambda Ad'*X;
%     Lambda*C2d -2*Lambda B2d'*X;
%     X*Ad X*B2d -X];
lmis = lmis + (lmi <= -eye(2*nx)*eps);
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

sol = optimize(lmis, -trace(P), sdpsettings('solver','mosek','verbose', 0))
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
xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 18); ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 18);

feasible_ic_and_inputs = {};
infeasible_ic_and_input = {};
counter  = 0; M = 1000;N = 1000;
% lets plot some trajectories
for i=1:M
    % Generate a random initial condition within the range [-5, 5]
    x0 = -max_ + (max_ - min_) * rand(nx, 1);

    d = zeros(nd,N); % no input
    
    % Call the simulation function
    [e, x] = simulate_system(dsys_, x0, d);
    
    if x0' * X * x0 > 1
        plot(x0(1,1), x0(2,1), 'x')
        plot(x(1,1:5), x(2,1:5), 'LineWidth', 1.5)
        counter = counter +1;
        infeasible_ic_and_input{end+1} = struct('d', d, 'x0', x0, 'e', e, 'x', x);
        continue
    else
        feasible_ic_and_inputs{end+1} = struct('d', d, 'x0', x0, 'e', e, 'x', x);
    end
    % plot(x0(1,1), x0(2,1), 'x')
    % plot(x(1,1:5), x(2,1:5))
    plot(x(1,:), x(2,:), 'LineWidth', 1.5)

end
fprintf('from %i samples %i where not feasible\n', M, counter)
legend({'$x^T P x < 0$', '$\|H x\|_\infty < 1$'}, 'Interpreter', 'latex', 'Location', 'northeast', 'FontSize', 18);


% Export the figure to a PDF
% print(gcf, './matlab/plots/no external inputs/invariance.png', '-dpng');
exportgraphics(gca, './matlab/plots/no external inputs/invariance.pdf');
save('./matlab/data/no external inputs/invarinace.mat', "feasible_ic_and_inputs","infeasible_ic_and_input")


%% lets look at some simulations in the time domain
% first the stable ones
K = 10;
t = linspace(0,(N-1)*dt, N);
figure; hold on; grid on% Create a new figure for plotting
for i = 1:K
    plot(t, feasible_ic_and_inputs{i}.e, 'LineWidth', 1.5)
end
xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('Output $e$', 'Interpreter', 'latex', 'FontSize', 18);
title('Outputs from feasible input and initial condition')
% print(gcf, './matlab/plots/no external inputs/outputs-for-feasible-inputs.png', '-dpng');
exportgraphics(gca, './matlab/plots/no external inputs/outputs-for-feasible-inputs.pdf');

% then the unstable ones
figure; hold on; grid on % Create a new figure for plotting
for i = 1:K
    plot(t, infeasible_ic_and_input{i}.e, 'LineWidth', 1.5)
end
xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('Output e', 'Interpreter', 'latex', 'FontSize', 18);
title('Outputs from infeasible input or initial condition')
% print(gcf, './matlab/plots/no external inputs/outputs-for-infeasible-inputs.png', '-dpng');
exportgraphics(gca, './matlab/plots/no external inputs/outputs-for-infeasible-inputs.pdf');
