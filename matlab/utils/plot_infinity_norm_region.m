function plot_infinity_norm_region(H)
% PLOT_INFINITY_NORM_REGION plots the boundary of the set {x : ||H*x||_inf < 1}
% where H is an n-by-2 matrix.

    [n, m] = size(H);
    if m ~= 2
        error('H must have exactly 2 columns.');
    end

    % Construct inequalities A*x <= b
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

    if isempty(V)
        error('No feasible region found.');
    end

    % Sort vertices counterclockwise
    V = unique(V','rows')';
    center = mean(V,2);
    angles = atan2(V(2,:) - center(2), V(1,:) - center(1));
    [~, idx] = sort(angles);
    V = V(:, idx);

    % Plot
    fill(V(1,:), V(2,:), [0.8 0.9 1], 'EdgeColor','b','LineWidth',1.5);
    axis equal
    xlabel('x_1'); ylabel('x_2');
    title('\{x : ||H x||_\infty < 1\}');
    grid on;

end
