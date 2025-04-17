% Numerical Non-Convexity Check

clear; clc; close all;

x_start = 0; y_start = 0;
x_goal = 10; y_goal = 10;
x_min = -2; x_max = 12;
y_min = -2; y_max = 12;
enable_obstacle = true; % Must be true to show non-convexity
obs_x = 3; obs_y = 3;
obs_r = 1.5;
robot_r = 0.5;
min_dist_sq = (obs_r + robot_r)^2;

constraint_func = @(x_int, y_int) (x_int - obs_x)^2 + (y_int - obs_y)^2 >= min_dist_sq;

is_feasible = @(x, y) x >= x_min && x <= x_max && y >= y_min && y <= y_max && ...
                      (~enable_obstacle || constraint_func(x, y));


rng('default'); % For reproducibility
num_checks = 5; % Number of pairs to check
num_samples_on_line = 20; % Points to check between pairs

figure; hold on; grid on; axis equal;
axis([x_min x_max y_min y_max]);
title('Numerical Check for Non-Convexity');
xlabel('X_{int}'); ylabel('Y_{int}');
% Plot obstacle boundary
if enable_obstacle
    theta = linspace(0, 2*pi, 100);
    plot(obs_x + (obs_r + robot_r)*cos(theta), obs_y + (obs_r + robot_r)*sin(theta), 'r-', 'LineWidth', 1.5);
    plot(obs_x, obs_y, 'rx');
    text(obs_x, obs_y, '  Obstacle', 'Color', 'red');
end

fprintf('Checking line segments between feasible points:\n');

for i = 1:num_checks
    % Find two random feasible points (p1, p2)
    p1 = []; p2 = [];
    while isempty(p1) || ~is_feasible(p1(1), p1(2))
        p1 = [x_min + (x_max - x_min)*rand(), y_min + (y_max - y_min)*rand()];
    end
    while isempty(p2) || ~is_feasible(p2(1), p2(2)) || norm(p1-p2) < 1 % Ensure points are distinct
        p2 = [x_min + (x_max - x_min)*rand(), y_min + (y_max - y_min)*rand()];
    end

    plot(p1(1), p1(2), 'bo', 'MarkerFaceColor', 'b');
    plot(p2(1), p2(2), 'bo', 'MarkerFaceColor', 'b');

    segment_is_convex = true;
    for lambda = linspace(0, 1, num_samples_on_line)
        p_inter = (1 - lambda) * p1 + lambda * p2; % Point on line segment
        if ~is_feasible(p_inter(1), p_inter(2))
            plot(p_inter(1), p_inter(2), 'kx', 'MarkerSize', 8, 'LineWidth', 1.5); % Mark infeasible point
            segment_is_convex = false;
        else
             plot(p_inter(1), p_inter(2), 'g.'); % Mark feasible point
        end
    end

    if ~segment_is_convex
        plot([p1(1), p2(1)], [p1(2), p2(2)], 'k--', 'LineWidth', 1.5); % Draw line where infeasibility found
        fprintf('  - Found INFEASIBLE point on segment between (%.2f, %.2f) and (%.2f, %.2f)\n', p1(1), p1(2), p2(1), p2(2));
    else
         plot([p1(1), p2(1)], [p1(2), p2(2)], 'g-'); % Draw line if all feasible
         fprintf('  - All points FEASIBLE on segment between (%.2f, %.2f) and (%.2f, %.2f)\n', p1(1), p1(2), p2(1), p2(2));
    end
end
legend({'Obstacle Boundary','Obstacle Center','Feasible Points (Endpoints)','Infeasible Mid-Point','Feasible Mid-Point','Infeasible Segment','Feasible Segment'}, 'Location', 'best');
hold off;

% Observation: If enable_obstacle is true, you should see black crosses 'kx'
% and dashed lines 'k--', indicating that segments connecting two feasible points
% can pass through the infeasible obstacle region, proving non-convexity.