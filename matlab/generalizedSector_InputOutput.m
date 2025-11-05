clc, clear, close all
% load system
run('shared.m')

% Same setup as proposition 2 LaBella, but with an input
% analysis
% standard sector
eps = 1e-5;
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

L1 = [eye(nx), zeros(nx,nd), zeros(nx,nw);
    Ad, Bd, B2d];
L2 = [zeros(ne,nx), eye(nd), zeros(ne,nw);
    Cd, Dd, D12d];
L3 = [zeros(nz,nx), zeros(nz,nd), eye(nw);
    C2d, D21d, D22d]; 
lmis = [];
lmi = L1' * [-X, zeros(nx,nx); zeros(nx,nx), X] * L1 + ...
    L2' * [-eye(nd), zeros(nd,ne); zeros(ne,nd), zeros(ne,ne)] * L2 + ...
    L3' * Pm(Lambda) * L3;
% lmi = [-X C2d'*Lambda Ad'*X;
%     Lambda*C2d -2*Lambda B2d'*X;
%     X*Ad X*B2d -X];
lmis = lmis + (lmi <= -eye(nx+nd+nw)*eps);
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
lambda = sdpvar(1,1);
% S_hat = 1;
S_hat = sdpvar(1,1);
for i=1:nw
    multiplier_constraint=[multiplier_constraint;m(i,1)>=eps];
end
M = diag(m);

lmis = [];
alpha = 0.925;
F = [-alpha*P zeros(nx,nd) P*C2d' + L' P*Ad';
    zeros(nd,nx) -eye(nd) D21d' Bd';
    C2d*P+L D21d -2*M M*B2d';
    Ad*P Bd B2d*M -P];
lmis = lmis + (F <= -eye(nx+nd+nw+nx)*eps);
for i = 1:nz
    li = L(i,:);
    lmis = lmis + ([S_hat, li;li', P] >= eps*(eye(nx+1)));
end
lmis = lmis + multiplier_constraint;
lmis = lmis + (P>=eps*eye(nx));

% sol = optimize(lmis, [], sdpsettings('solver','mosek','verbose', 0))
sol = optimize(lmis, S_hat, sdpsettings('solver','mosek','verbose', 0))

s = sqrt(1/double(S_hat))
% s = 1
disp('max real EV of F:'); max(real(eig(double(F))))

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
min_ = -7; max_ = -min_;

theta = linspace(0, 2*pi, 200); % angle values
unit_circle = [cos(theta); sin(theta)]; % points on unit circle

% compute X^(-1/2)
[V, D] = eig(1/s^2*X);
X_half_inv = V * diag(1 ./ sqrt(diag(D))) * V';

ellipse = X_half_inv * unit_circle; % transform the circle
figure; hold on; grid on
% axis equal;
plot(ellipse(1,:), ellipse(2,:), 'b-', 'LineWidth', 2);
xlabel('x_1'); ylabel('x_2'); grid on, hold on

% Generate all sign combinations for s ∈ {-1,1}^2
% S = [-1 -1; -1 1; 1 1; 1 -1];
% 
% % Compute vertices
% X_ = (H \ S')';  % equivalent to inv(H)*s'
% 
% % Close the polygon by repeating the first vertex
% X_ = [X_; X_(1,:)];
% 
% % Plot the edges
% plot(X_(:,1), X_(:,2), 'r-', 'LineWidth', 2);

feasible_ic_and_inputs = {};
infeasible_ic_and_input = {};
gs = {}; ws = {}; zs={};xs={};
counter  = 0; M = 1000; N = 100; b_nonlinear = false;
t = linspace(0,(N-1)*dt, N);
% lets plot some trajectories
for i=1:M
    % Generate a random initial condition within the range [-5, 5]
    x0 = -max_ + (max_ - min_) * rand(nx, 1);

    d = sqrt(s^2*(1-alpha))*sin(linspace(0,(N-1)*dt,N));
    % d = sqrt(s^2*(1-alpha))*2*(rand(nd,N) - 0.5);
    % d = sqrt(s^2*(1-alpha))*ones(nd,N);
   
    % Call the simulation function
    [e, x, z, w] = simulate_system(dsys_, x0, d);
    xs{end+1} = x; ws{end+1} = w; zs{end+1} = z;
    
    % if norm(d,'inf') > s^2
    if x0'*X*x0  > s^2 || norm(d.^2,'inf') > (1-alpha)*s^2
        plot(x0(1,1), x0(2,1), 'x', 'LineWidth', 1.5)
        plot(x(1,1:5), x(2,1:5), 'LineWidth', 1.5)
        counter = counter +1;
        infeasible_ic_and_input{end+1} = struct('d', d, 'x0', x0, 'e', e, 'x', x, 'w', w, 'z', z);
        continue
    else
        feasible_ic_and_inputs{end+1} = struct('d', d, 'x0', x0, 'e', e, 'x', x, 'w', w, 'z', z);
    end
    plot(x0(1,1), x0(2,1), 'o', 'LineWidth', 1.5)
    plot(x(1,:), x(2,:), 'LineWidth', 1.5)
    
    gs{end+1} = H*x;

end
fprintf('from %i samples %i where not feasible\n', M, counter)
legend({'$x^T P x < s^2$', '$\|H x\|_\infty < 1$'}, 'Interpreter', 'latex', 'Location', 'northeast', 'FontSize', 18);

% print(gcf, './matlab/plots/with external inputs/invariance.png', '-dpng');
% exportgraphics(gcf, './matlab/plots/with external inputs/three-differnt-init-cond-x-w-z.png');
save('./matlab/data/init_rand-input_sin.mat', "feasible_ic_and_inputs","s","dsys_", 'Lambda', 'X', 'H', 't', 'dt')

count2 = 0;
for i = 1:length(feasible_ic_and_inputs)
    z = feasible_ic_and_inputs{i}.z;
    if any(any(abs(z)>g))
        count2=count2+1;
        % figure, hold on, grid on
        % plot(t,feasible_ic_and_inputs{i}.w(1,:), 'LineWidth', 1.5)
        % plot(t,feasible_ic_and_inputs{i}.w(2,:), 'LineWidth', 1.5)
    end
end
fprintf("Found %i trajectories with |z|>1 from %i\n", count2, length(feasible_ic_and_inputs))

%% this one is useful for analyzing z and w



% figure, grid on, hold on
% for g = gs
%     g = g{:};
%     plot(g(1,:), g(2,:), 'LineWidth', 1.5)
%     xlabel('$Hx(1,:)$', 'Interpreter', 'latex', 'FontSize', 18)
%     ylabel('$Hx(2,:)$', 'Interpreter', 'latex', 'FontSize', 18)
% end
% legend({'$(x)_1$','$(x)_2$','$(x)_3$' }, 'Interpreter', 'latex', 'Location', 'northeast', 'FontSize', 18);
% 
% figure
% N1 = 100;
% for i = 1:length(zs)
%     z = zs{i}; w = ws{i}; x = xs{i};
%     subplot(3,1,1), hold on, grid on
%     plot(t(1:N1), x(1,1:N1), 'LineWidth', 1.5), plot(t(1:N1),x(2,1:N1), 'LineWidth', 1.5)
%     ylabel('$x$', 'Interpreter', 'latex', 'FontSize', 18)
%     subplot(3,1,2), hold on, grid on
%     plot(t(1:N1), w(1,1:N1), 'LineWidth', 1.5), plot(t(1:N1),w(2,1:N1), 'LineWidth', 1.5)
%     ylabel('$w$', 'Interpreter', 'latex', 'FontSize', 18)
%     subplot(3,1,3), hold on, grid on
%     plot(t(1:N1), z(1,1:N1), 'LineWidth', 1.5), plot(t(1:N1),z(2,1:N1), 'LineWidth', 1.5)
%     ylabel('$z$', 'Interpreter', 'latex', 'FontSize', 18)
% end
    


%% lets look at some simulations in the time domain
% first the stable ones
figure; hold on; grid on% Create a new figure for plotting
for i = 1:length(feasible_ic_and_inputs)
    plot(t, feasible_ic_and_inputs{i}.d, 'LineWidth', 1.5)
end
xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('Output $d$', 'Interpreter', 'latex', 'FontSize', 18);
title('Feasible inputs')
% print(gcf, './matlab/plots/with external inputs/feasible-inputs.png', '-dpng');
% exportgraphics(gca, './matlab/plots/with external inputs/feasible-inputs.pdf');

figure; hold on; grid on% Create a new figure for plotting
for i = 1:length(feasible_ic_and_inputs)
    plot(t, feasible_ic_and_inputs{i}.e, 'LineWidth', 1.5)
end
xlabel('Time (s)','Interpreter', 'latex', 'FontSize', 18);
ylabel('Output $e$', 'Interpreter', 'latex', 'FontSize', 18);
title('Outputs from feasible input and initial condition')
% print(gca, './matlab/plots/with external inputs/outputs-for-feasible-inputs.png', '-dpng');
% exportgraphics(gca, './matlab/plots/with external inputs/outputs-for-feasible-inputs.pdf');

% then the unstable ones
K = 10;
figure; hold on; grid on% Create a new figure for plotting
for i = 1:K
    plot(t, infeasible_ic_and_input{i}.d, 'LineWidth', 1.5)
end
xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('Output $d$', 'Interpreter', 'latex', 'FontSize', 18);
title('Infeasible inputs')
% exportgraphics(gca, './matlab/plots/with external inputs/infeasible-inputs.pdf');

figure; hold on; grid on % Create a new figure for plotting
for i = 1:K
    plot(t, infeasible_ic_and_input{i}.e, 'LineWidth', 1.5)
end
xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('Output $e$', 'Interpreter', 'latex', 'FontSize', 18);
title('Outputs from infeasible input or initial condition')
print(gcf, './matlab/plots/with external inputs/outputs-for-infeasible-inputs.png', '-dpng');
exportgraphics(gca, './matlab/plots/with external inputs/outputs-for-infeasible-inputs.pdf');


