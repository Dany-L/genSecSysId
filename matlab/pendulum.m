%% Nonlinear oscillator with softening spring
clear; clc; close all;

% Parameters
c  = 0.4;
k1 = 1.0;
k3 = 0.2;
u  = 0;  % input (set nonzero later if you want to see forced response)

% Dynamics
f = @(x)[x(2);
         -c*x(2) - k1*x(1) + k3*x(1)^3 + u];

% Grid for vector field
[x1, x2] = meshgrid(linspace(-3,3,30), linspace(-3,3,30));
dx1 = zeros(size(x1)); dx2 = zeros(size(x2));
for i = 1:numel(x1)
    dx = f([x1(i); x2(i)]);
    dx1(i) = dx(1);
    dx2(i) = dx(2);
end

% Normalize vectors for display
L = sqrt(dx1.^2 + dx2.^2);
dx1n = dx1 ./ (L + eps);
dx2n = dx2 ./ (L + eps);

figure; hold on; box on;
quiver(x1, x2, dx1n, dx2n, 0.6, 'b');
xlabel('x_1 (position)');
ylabel('x_2 (velocity)');
title('Phase Portrait: Nonlinear Oscillator with Softening Spring');
axis equal; grid on;

% Simulate trajectories
f_ode = @(t,x) [x(2);
                -c*x(2) - k1*x(1) + k3*x(1)^3 + u];

x0s = [-2.5 0; -2 0; -1.5 0; 1.5 0; 2 0; 2.5 0];
colors = lines(size(x0s,1));
for i = 1:size(x0s,1)
    [t,x] = ode45(f_ode, [0 1], x0s(i,:)');
    plot(x(:,1), x(:,2), 'Color', colors(i,:), 'LineWidth', 1.4);
    plot(x(1,1), x(1,2), 'o', 'Color', colors(i,:), 'MarkerFaceColor', colors(i,:));
end

plot(0,0,'ko','MarkerFaceColor','k','MarkerSize',6); % equilibrium
legend('Vector field','Trajectories');
