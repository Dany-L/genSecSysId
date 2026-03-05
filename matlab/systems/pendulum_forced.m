function dx = pendulum_forced(t, x, t_u, u_u, params)

    % unpack parameters
    g = params.g;
    L = params.L;
    m = params.m;
    c = params.c;

    % interpolate input u(t)
    u = interp1(t_u, u_u, t, 'linear', 'extrap');

    % states
    theta = x(1);
    theta_dot = x(2);

    % dynamics
    dx = zeros(2,1);
    dx(1) = theta_dot;
    dx(2) = -(g/L)*sin(theta) - c*theta_dot + u/(m*L^2);

end