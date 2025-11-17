clear, close all

% folder_name = '~/genSecSysId-Data/models/crnn/44a34327b5e14d92baa101236e7d8aee';
% folder_name = '~/genSecSysId-Data/models/crnn/76e61a76f426478b917b05015e19b0a1';
folder_name = '~/genSecSysId-Data/models/crnn/823bbdefe4064bfbbeff8b770cdef7c4';
% st

model_filename = 'best_model_params.mat';

load(fullfile(folder_name,model_filename))

ne = size(C,1);
nx = size(A, 1); % Number of states
nd = size(B, 2); % Number of inputs
nz = size(C2,1);nw=nz;
dt = 0.1; % should actually be part of the parameters that are loaded
g = 1; % Set the bound for the deadzone
sat = @(x) max(min(x, g), -g);
dzn = @(x) x-sat(x);

X_l = inv(P);
H_l = L*X_l;
Lambda_l = diag(la);
lmi = [-alpha^2*X_l (C2'+H_l')*Lambda_l A'*X_l; ...
    Lambda_l*(C2+H_l) -2*Lambda_l B2'*X_l; ...
    X_l*A X_l*B2 -X_l];
disp('max real EV of LMI:'); max(real(eig(lmi)))

X_l, H_l, Lambda_l

for i =1:nz
    hi_l = H_l(i,:);
    assert(max(real(eig([1/s^2 hi_l;hi_l' X_l])))>=0)
end

%%
L = sdpvar(nz,nx);
P = sdpvar(nx,nx);
m = sdpvar(nz,1);
lambda = sdpvar(1,1);
% S_hat = 1;
multiplier_constraint = [];
S_hat = sdpvar(1,1);
for i=1:nw
    multiplier_constraint=[multiplier_constraint;m(i,1)>=eps];
end
M = diag(m);

lmis = [];
F = [-alpha^2*P zeros(nx,nd) P*C2' + L' P*A';
    zeros(nd,nx) -eye(nd) D21' B';
    C2*P+L D21 -2*M M*B2';
    A*P B B2*M -P];
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

X, H, Lambda


dsys_l_ = struct('Ad', A, 'Bd', B, 'B2d', B2, ...
    'Cd', C, 'Dd', D, 'D12d', D12, ...
    'C2d', C2, 'D21d', D21, 'D22d', D22, 'ne', ne, 'nl', dzn, 'dt', dt);


%%
run('shared.m')

%%
min_ = -6; max_ = -min_;

theta = linspace(0, 2*pi, 200); % angle values
unit_circle = [cos(theta); sin(theta)]; % points on unit circle

% compute X^(-1/2)
L = chol(1/s^2*X); X_half_inv = inv(L');
ellipse = X_half_inv * unit_circle; % transform the circle

% Construct inequalities A*x <= b
[n, m] = size(H);
A = [H; -H];
b = ones(2*n, 1);
% Compute all intersection points
V = []; % vertices
for i = 1:size(A,1)
    for j = i+1:size(A,1)
        % Solve A([i,j],:) * x = b([i,j])
        M = A([i,j],:);
        if abs(det(M)) > 1e-10
            x = M \ b([i,j]);
            % Keep if satisfies all inequalities
            if all(A*x <= b + 1e-9)
                V = [V, x];
            end
        end
    end
end

% Sort vertices counterclockwise
V = unique(V','rows')';
center = mean(V,2);
angles = atan2(V(2,:) - center(2), V(1,:) - center(1));
[~, idx] = sort(angles);
V = V(:, idx);

figure; hold on; grid on
fill(V(1,:), V(2,:), [0.8 0.9 1], 'EdgeColor','r','LineWidth',1.5);
% plot(V(1,:), V(2,:),'r','LineWidth',1.5);
plot(ellipse(1,:), ellipse(2,:), 'b-', 'LineWidth', 2);
counter  = 0; M = 200; N = 50; b_nonlinear = false;
t = linspace(0,(N-1)*dt, N);
% lets plot some trajectories
for i=1:M
    % Generate a random initial condition within the range [-5, 5]
    x0 = -max_ + (max_ - min_) * rand(nx, 1);

    % d = sqrt(s^2*(1-alpha))*sin(linspace(0,(N-1)*dt,N));
    % d = sqrt(s^2*(1-alpha))*2*(rand(nd,N) - 0.5);
    d = 0.5*2*(rand(nd,N) - 0.5);
    % d = sqrt(s^2*(1-alpha))*ones(nd,N);
   
    % Call the simulation function
    [e, x, z, w] = simulate_system(dsys_, x0, d);
    [e_trained, x_trained, ~, ~] = simulate_system(dsys_l_, x0, d);
    
    plot(x0(1,1), x0(2,1), 'o', 'LineWidth', 1.5)
    plot(x(1,:), x(2,:),'--', 'LineWidth', 1.5)
    plot(x_trained(1,:), x_trained(2,:), 'LineWidth', 1.5)
    

end
