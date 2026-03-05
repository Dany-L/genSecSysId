clc, clear, close all
% load system
run('./systems/pendulum.m')

nd = 1;
tspan = [0 20];
dt = 0.1;
N = tspan(2)/dt;

t_u = 0:0.1:tspan(2);
u_u = zeros(length(t_u),1);

N_1 = tspan(2)/2/dt;
t_u_1 = 0:dt:(N_1-1)*dt; 
% u_u(1:size(t_u_1,2)) =  5* 2*(rand(nd,N_1) -0.5);
% u_u(1:size(t_u_1,2)) = sin(t_u_1);


% x0 = [2; 6];           % initial angle and velocity
x0 = [0;0];

% Generate white noise
white_noise = randn(length(t_u), 1);

% Design low-pass filter
fs = 1/dt; % Sampling frequency
fc = 1;    % Cut-off frequency
[b, a] = butter(4, fc/(fs/2)); % 4th order Butterworth filter

% Apply low-pass filter to white noise
filtered_noise = filter(b, a, white_noise);

% Replace the torque signal with the filtered noise
u_u(1:size(t_u_1,2)) = filtered_noise(1:size(t_u_1,2));

% Solve ODE
[t, x] = ode45(@(t,x) pendulum_forced(t, x, t_u, u_u, params), tspan, x0);

% Plot results
figure; hold on, grid on,
plot(x0(1),x0(2), 'x', 'MarkerSize',5)
plot(x(:,1), x(:,2))

% Plot results
figure; 
subplot(2,1,1); plot(t, x(:,1)); ylabel('\theta (rad)');
subplot(2,1,2); plot(t, x(:,2)); ylabel('\theta dot (rad/s)'); xlabel('t (s)');
% s = 1;
% alpha = 1;
% 
% io_data = {};
% ws = {}; zs={};xs={};
% counter  = 0; M = 300; N = 50;
% t = linspace(0,(N-1)*dt, N);
% min_ = -6; max_ = -min_;
% % lets plot some trajectories
% for i=1:M
%     % Generate a random initial condition within the range [-5, 5]
%     x0 = -max_ + (max_ - min_) * rand(nx, 1);
%     % x0 = [0;0];
% 
%     % d = sqrt(s^2*(1-alpha))*sin(linspace(0,(N-1)*dt,N));
%     d = sqrt(s^2*(1-alpha))*2*(rand(nd,N) - 0.5);
%     % d = sqrt(s^2*(1-alpha))*ones(nd,N);
% 
%     % Call the simulation function
%     [e, x, z, w] = simulate_system(dsys_, x0, d);
%     xs{end+1} = x; ws{end+1} = w; zs{end+1} = z;
%     io_data{end+1} = struct('d', d, 'x0', x0, 'e', e, 'x', x, 'w', w, 'z', z);
% 
% 
% end

% save('./data/init_rand-input_rand.mat', "io_data","s","dsys_", 'Lambda', 'X', 'H', 't', 'dt','alpha','sys','min_')
