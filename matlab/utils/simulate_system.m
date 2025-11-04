function [e, x, z, w] = simulate_system(sys,x0, d)
    nx = size(x0,1); ne = sys.ne; nw = size(sys.C2d,1);
    N = length(d);
    e = zeros(ne, N);
    x = zeros(nx, N+1);
    w = zeros(nw,N);
    z = zeros(nw,N);
    x(:,1) = x0; % initial condition
    for k = 1:N
        d_k = d(:,k); x_k = x(:,k);
        z(:, k) = sys.C2d*x_k + sys.D21d*d_k; % needs to be changed if D22 is not zero
        w(:, k) = sys.nl(z(:,k));
        e(:,k) = sys.Cd*x_k + sys.Dd*d_k + sys.D12d*w(:,k);
        x(:,k+1) = sys.Ad*x_k + sys.Bd*d_k + sys.B2d*w(:,k);
    end
end