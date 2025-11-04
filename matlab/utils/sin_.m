function [d] = sin_(N, max_range)
%SIN_ Summary of this function goes here
%   Detailed explanation goes here
%DECAYING_SINUS Summary of this function goes here
%   Detailed explanation goes here
% Parameters
% N = 1000;        % number of samples
% s = 10.0;        % L2 budget
% r = 0.99;        % exponential decay rate (0 < r < 1)
omega = 2*pi*5/100;  % angular frequency, e.g. 5 cycles per 100 samples
max_=max_range;min_=-max_;
a = -max_ + (max_ - min_) * rand(1, 1);
phi = 0;

% phi = -2 + rand() * 1.0;  % Generate a random number between -0.5 and 0.5
% phi = 0.3;       % phase offset [radians]

% Compute envelope scaling constant
% Ensures sum_{k=0}^{N-1} (a_k)^2 = s  (upper bound)
% c = sqrt(s^2 * (1 - r^2) / (1 - r^(2*N)));


% Generate discrete-time vector
k = 0:(N-1);

% Exponential envelope
% c = -max_ + (max_ - min_) * rand(1, 1);  % Generate a random number between -.5 and .5
% a = c * (r.^k);
% a = 1;

% Sine signal with exponential decay
d = a .* sin(omega*k + phi);

% fprintf('norm of d %.2f\n', norm(d,2))
% fprintf('Envelope constant c = %.6f\n', c);
% fprintf('Sum of squares (energy) = %.6f\n', energy);

% Plot signal
% figure;
% plot(k, d, 'LineWidth', 1.2);
% xlabel('k'); ylabel('d_k');
% title('Exponentially decaying sine signal');
% grid on;