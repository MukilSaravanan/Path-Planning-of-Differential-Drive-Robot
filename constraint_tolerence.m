% MATLAB Code for Numerical Noise (Constraint Tolerance) Demo

clear; clc; close all;

% --- Problem Parameters ---
x_min = -2; x_max = 12;
y_min = -2; y_max = 12;
obs_x = 5; obs_y = 5;
obs_r = 1.5;
robot_r = 0.5;
min_dist = obs_r + robot_r; % 2.0
min_dist_sq = min_dist^2;   % 4.0

% Points to check - choose some near the boundary
point_A = [3.0, 5.0]; % Distance = sqrt((3-5)^2 + (5-5)^2) = 2.0 (exactly on boundary)
point_B = [5.0, 3.0]; % Distance = 2.0 (exactly on boundary)
point_C = [3.5, 3.5]; % Distance = sqrt((3.5-5)^2+(3.5-5)^2) = sqrt(1.5^2+1.5^2) = sqrt(2*2.25) = sqrt(4.5) approx 2.12 (outside)
point_D = [4.0, 4.0]; % Distance = sqrt((4-5)^2+(4-5)^2) = sqrt(1+1) = sqrt(2) approx 1.41 (inside - infeasible)

points = [point_A; point_B; point_C; point_D];
point_names = {'A (Dist=2.0)', 'B (Dist=2.0)', 'C (Dist=2.12)', 'D (Dist=1.41)'};

% --- Feasibility Check Function with Tolerance ---
% Checks if squared distance >= min_dist_sq - tolerance
is_feasible_tol = @(x, y, tol) ...
    x >= x_min && x <= x_max && y >= y_min && y <= y_max && ...
    ((x - obs_x)^2 + (y - obs_y)^2 >= min_dist_sq - tol);

% --- Test with Different Tolerances ---
tolerances_to_test = [0, 1e-1, 1e-3, 1e-6]; % Test zero, loose, recommended[cite: 59], tight

fprintf('Effect of Constraint Tolerance on Feasibility:\n');
fprintf('%-15s |', 'Point');
for tol = tolerances_to_test
    fprintf(' Tol=%.1e |', tol);
end
fprintf('\n');
fprintf('----------------|-----------|-----------|-----------|-----------|\n');

for i = 1:size(points, 1)
    fprintf('%-15s |', point_names{i});
    p = points(i,:);
    for tol = tolerances_to_test
        feasible = is_feasible_tol(p(1), p(2), tol);
        if feasible
            fprintf(' Feasible  |');
        else
            fprintf(' INFEASIBLE|');
        end
    end
    fprintf('\n');
end

% Observation:
% - Point A & B (exactly on boundary): May be feasible/infeasible with tol=0 due to float precision.
%   Should be feasible with positive tolerance.
% - Point C (outside): Should always be feasible.
% - Point D (inside): Should always be infeasible unless tolerance is very large (>= 4.0 - 2.0 = 2.0).
% - A very small tolerance (e.g., 1e-16, not tested here) might incorrectly classify A or B as infeasible.
% - A large tolerance (e.g., 1e-1) might incorrectly classify points slightly inside the circle as feasible.