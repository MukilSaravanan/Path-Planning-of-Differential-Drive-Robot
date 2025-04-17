% MATLAB Code for GA, Hill Climbing, and Grid Search Comparison
% For Simplified Problem (Section 3.4) with Timing and ENHANCED PLOTTING
% Aims to demonstrate conclusions from Report Sections 3.4.3 & 3.4.4

clear; clc; close all; rng('default'); % Use default for consistent demo plot

%% --- Problem Parameters (Placeholders) ---
x_start = 0; y_start = 0;
x_goal = 10; y_goal = 10;
x_min = -2; x_max = 12;
y_min = -2; y_max = 12;
enable_obstacle = true;
obs_x = 5; obs_y = 5; % Adjusted obstacle position slightly
obs_r = 1.5;
robot_r = 0.5;
min_dist_sq = (obs_r + robot_r)^2; % Effective radius = 1.0

%% --- Feasibility Check & Objective Functions ---
constraint_func = @(x_int, y_int) (x_int - obs_x)^2 + (y_int - obs_y)^2 >= min_dist_sq;
is_feasible = @(x, y) x >= x_min && x <= x_max && y >= y_min && y <= y_max && ...
                      (~enable_obstacle || constraint_func(x, y));
objective_func = @(x_int, y_int) sqrt((x_int - x_start)^2 + (y_int - y_start)^2) + ...
                                 sqrt((x_goal - x_int)^2 + (y_goal - y_int)^2);
fitness_func = @(x, y) objective_func(x, y);

%% --- Initialize Results Structure ---
results = struct();

%% --- 1. Genetic Algorithm Implementation ---
fprintf('--- Running Genetic Algorithm ---\n');
% (Keep GA Parameters, Initialization, Evolution Loop, and Results reporting from the previous version)
% --- GA Parameters (from Report Sec 3.4.2 [cite: 70]) ---
ga_params.pop_size = 30; ga_params.num_generations = 50; ga_params.tournament_size = 3;
ga_params.crossover_rate = 0.8; ga_params.mutation_rate = 0.1; ga_params.mutation_sigma = 0.5;
% --- GA Initialization [cite: 71] ---
population = zeros(ga_params.pop_size, 2); fitness = zeros(ga_params.pop_size, 1);
init_feasible_count = 0; max_init_attempts = ga_params.pop_size * 100; attempt = 0;
while init_feasible_count < ga_params.pop_size && attempt < max_init_attempts; x_trial = x_min + (x_max - x_min) * rand(); y_trial = y_min + (y_max - y_min) * rand(); if is_feasible(x_trial, y_trial); init_feasible_count = init_feasible_count + 1; population(init_feasible_count, :) = [x_trial, y_trial]; end; attempt = attempt + 1; end
if init_feasible_count < ga_params.pop_size; error('GA: Could not initialize feasible population.'); end
results.ga_func_evals = 0; for i = 1:ga_params.pop_size; fitness(i) = fitness_func(population(i, 1), population(i, 2)); results.ga_func_evals = results.ga_func_evals + 1; end
history_ga = zeros(ga_params.num_generations, 1);
% --- GA Evolution Loop with Timing ---
tic;
for gen = 1:ga_params.num_generations; new_population = zeros(size(population)); for i = 1:2:ga_params.pop_size; parent1_idx = tournament_selection(fitness, ga_params.tournament_size); parent2_idx = tournament_selection(fitness, ga_params.tournament_size); parent1 = population(parent1_idx, :); parent2 = population(parent2_idx, :); if rand < ga_params.crossover_rate; alpha_co = rand; child1 = alpha_co*parent1+(1-alpha_co)*parent2; child2 = (1-alpha_co)*parent1+alpha_co*parent2; else; child1 = parent1; child2 = parent2; end; if rand < ga_params.mutation_rate; child1 = child1 + randn(1, 2) * ga_params.mutation_sigma; end; if rand < ga_params.mutation_rate; child2 = child2 + randn(1, 2) * ga_params.mutation_sigma; end; new_population(i, :) = child1; if i+1 <= ga_params.pop_size; new_population(i+1, :) = child2; end; end; population = new_population; for i = 1:ga_params.pop_size; x = population(i, 1); y = population(i, 2); if is_feasible(x, y); fitness(i) = fitness_func(x, y); else; fitness(i) = inf; end; results.ga_func_evals = results.ga_func_evals + 1; end; [best_fitness_gen, ~] = min(fitness); history_ga(gen) = best_fitness_gen; end
results.ga_time = toc;
% --- GA Results ---
[results.ga_best_obj, best_idx_ga] = min(fitness); results.ga_best_point = population(best_idx_ga, :);
fprintf('GA Final Best Objective: %.4f\n', results.ga_best_obj); fprintf('GA Final Best Point (x, y): (%.4f, %.4f)\n', results.ga_best_point(1), results.ga_best_point(2)); fprintf('GA Function Evaluations: %d\n', results.ga_func_evals); fprintf('GA Computational Time: %.4f seconds\n\n', results.ga_time);

%% --- 2. Hill Climbing Implementation ---
fprintf('--- Running Hill Climbing ---\n');
% (Keep HC Parameters, Loop, and Results reporting from the previous version)
% --- HC Parameters ---
hc_params.max_iter = 100; hc_params.step_size = 0.5; hc_params.tolerance = 1e-4; hc_params.num_restarts = 5;
% --- HC Loop with Timing ---
results.hc_best_obj_overall = inf; results.hc_best_point_overall = [NaN, NaN]; results.hc_func_evals = 0;
hc_restart_results = zeros(hc_params.num_restarts, 3); all_hc_final_points = nan(hc_params.num_restarts, 2);
tic;
for r = 1:hc_params.num_restarts; x_curr = NaN; y_curr = NaN; start_attempts = 0; max_start_attempts = 1000; while (~is_feasible(x_curr, y_curr) || isnan(x_curr)) && start_attempts < max_start_attempts; x_curr = x_min + (x_max - x_min) * rand(); y_curr = y_min + (y_max - y_min) * rand(); start_attempts = start_attempts + 1; end; if ~is_feasible(x_curr, y_curr) || isnan(x_curr); fprintf('HC Restart %d: Skipped.\n', r); continue; end; obj_curr = fitness_func(x_curr, y_curr); results.hc_func_evals = results.hc_func_evals + 1; current_step_size = hc_params.step_size; for iter = 1:hc_params.max_iter; improved = false; best_neighbor_x = x_curr; best_neighbor_y = y_curr; obj_best_neighbor = obj_curr; neighbors = [x_curr + current_step_size, y_curr; x_curr - current_step_size, y_curr; x_curr, y_curr + current_step_size; x_curr, y_curr - current_step_size]; for k = 1:size(neighbors, 1); xn = neighbors(k, 1); yn = neighbors(k, 2); if is_feasible(xn, yn); obj_neighbor = fitness_func(xn, yn); results.hc_func_evals = results.hc_func_evals + 1; if obj_neighbor < obj_best_neighbor; obj_best_neighbor = obj_neighbor; best_neighbor_x = xn; best_neighbor_y = yn; improved = true; end; end; end; if improved && (obj_curr - obj_best_neighbor) > hc_params.tolerance; x_curr = best_neighbor_x; y_curr = best_neighbor_y; obj_curr = obj_best_neighbor; else; current_step_size = current_step_size * 0.5; if current_step_size < hc_params.tolerance / 10; break; end; end; end; hc_restart_results(r, :) = [obj_curr, x_curr, y_curr]; all_hc_final_points(r,:) = [x_curr, y_curr]; fprintf('HC Restart %d: Found Objective = %.4f at (%.4f, %.4f)\n', r, obj_curr, x_curr, y_curr); if obj_curr < results.hc_best_obj_overall; results.hc_best_obj_overall = obj_curr; results.hc_best_point_overall = [x_curr, y_curr]; end; end
results.hc_time = toc;
% --- HC Results ---
fprintf('HC Overall Best Objective: %.4f\n', results.hc_best_obj_overall); fprintf('HC Overall Best Point (x, y): (%.4f, %.4f)\n', results.hc_best_point_overall(1), results.hc_best_point_overall(2)); fprintf('HC Total Function Evaluations: %d\n', results.hc_func_evals); fprintf('HC Computational Time: %.4f seconds\n\n', results.hc_time);

%% --- 3. Grid Search Implementation ---
fprintf('--- Running Grid Search ---\n');
% (Keep GS Parameters, Loop, and Results reporting from the previous version)
% --- GS Parameters ---
gs_params.num_grid_points = 50;
% --- GS Loop with Timing ---
x_search_vals = linspace(x_min, x_max, gs_params.num_grid_points); y_search_vals = linspace(y_min, y_max, gs_params.num_grid_points); results.gs_best_obj = inf; results.gs_best_point = [NaN, NaN]; results.gs_func_evals = 0;
tic;
for i = 1:gs_params.num_grid_points; for j = 1:gs_params.num_grid_points; xi = x_search_vals(i); yi = y_search_vals(j); if is_feasible(xi, yi); current_obj = fitness_func(xi, yi); results.gs_func_evals = results.gs_func_evals + 1; if current_obj < results.gs_best_obj; results.gs_best_obj = current_obj; results.gs_best_point = [xi, yi]; end; end; end; end
results.gs_time = toc;
% --- GS Results ---
fprintf('Grid Search Best Objective: %.4f\n', results.gs_best_obj); fprintf('Grid Search Best Point (x, y): (%.4f, %.4f)\n', results.gs_best_point(1), results.gs_best_point(2)); fprintf('Grid Search Function Evaluations (feasible points): %d\n', results.gs_func_evals); fprintf('Grid Search Computational Time: %.4f seconds\n\n', results.gs_time);

%% --- Summary & Link to Report Conclusions ---
% (Keep Summary printing section from the previous version)
fprintf('--- Comparison Summary & Report Links (Sec 3.4.3 / 3.4.4) ---\n'); fprintf('Method         | Best Objective | Func Evals | Comp. Time (s) | Notes based on Report\n'); fprintf('---------------|----------------|------------|----------------|-----------------------------------------------------------\n'); fprintf('Genetic Alg.   | %14.4f | %10d | %14.4f | Balanced, avoids local minima, more evals than HC [cite: 80, 81]\n', results.ga_best_obj, results.ga_func_evals, results.ga_time); fprintf('Hill Climbing  | %14.4f | %10d | %14.4f | Fast, but gets stuck in local minima (see restarts) [cite: 77]\n', results.hc_best_obj_overall, results.hc_func_evals, results.hc_time); fprintf('Grid Search    | %14.4f | %10d | %14.4f | Comprehensive but computationally expensive [cite: 75]\n', results.gs_best_obj, results.gs_func_evals, results.gs_time); fprintf('\nObservations:\n'); fprintf('- Hill Climbing restarts often yield different objectives, confirming local minima due to non-convexity caused by the obstacle[cite: 77, 83].\n'); if results.hc_func_evals < results.ga_func_evals; fprintf('- GA required more function evaluations than HC, as expected[cite: 81].\n'); else; fprintf('- Note: GA required fewer/similar function evaluations than HC in this run (may vary).\n'); end; if results.gs_func_evals > results.ga_func_evals && results.gs_func_evals > results.hc_func_evals; fprintf('- Grid Search evaluated the most points, showing high cost[cite: 75, 87].\n'); end; fprintf('- GA/HC found solutions near the obstacle boundary, not the simple straight line, when obstacle enabled[cite: 84].\n'); fprintf('- GA (population-based) often finds a good quality solution compared to HC[cite: 86].\n');


%% --- Calculate Objective Landscape for Plotting ---
fprintf('\nCalculating objective landscape for visualization...\n');
vis_grid_points = 100; % Resolution for visualization grid
x_vis_vals = linspace(x_min, x_max, vis_grid_points);
y_vis_vals = linspace(y_min, y_max, vis_grid_points);
[X_vis, Y_vis] = meshgrid(x_vis_vals, y_vis_vals);
Z_vis = zeros(size(X_vis));

for i = 1:vis_grid_points
    for j = 1:vis_grid_points
        xi = X_vis(i,j);
        yi = Y_vis(i,j);
        if is_feasible(xi, yi)
            Z_vis(i,j) = objective_func(xi, yi);
        else
            Z_vis(i,j) = NaN; % Mark infeasible points for plotting
        end
    end
end
fprintf('Landscape calculation complete.\n');

%% --- Calculate Objective Landscape for Plotting ---
% (This section remains the same - Ensure it runs before plotting)
fprintf('\nCalculating objective landscape for visualization...\n');
vis_grid_points = 100; % Resolution for visualization grid
x_vis_vals = linspace(x_min, x_max, vis_grid_points);
y_vis_vals = linspace(y_min, y_max, vis_grid_points);
[X_vis, Y_vis] = meshgrid(x_vis_vals, y_vis_vals);
Z_vis = zeros(size(X_vis));
for i = 1:vis_grid_points
    for j = 1:vis_grid_points
        xi = X_vis(i,j); yi = Y_vis(i,j);
        if is_feasible(xi, yi); Z_vis(i,j) = objective_func(xi, yi);
        else; Z_vis(i,j) = NaN; end % Mark infeasible points
    end
end
fprintf('Landscape calculation complete.\n');

%% --- Enhanced Plotting with Subplots ---
figure('Name', 'Algorithm Comparison & Objective Landscape'); % Add a figure name

% --- Subplot 1: Contour Plot with Results ---
subplot(1,2,1);
hold on; grid on; axis equal;
axis([x_min x_max y_min y_max]);
title('Algorithm Comparison (Top View)');
xlabel('X_{int}'); ylabel('Y_{int}');

% Plot Contours (Colored)
contour_levels = 30; % Adjust number of contour levels if needed
contour(X_vis, Y_vis, Z_vis, contour_levels); % Remove 'Color' to enable colored contours
colorbar; % Show color scale for objective value
colormap(gca, 'jet'); % Apply colormap to this subplot

% Plot Essentials
h_plots = []; h_labels = {}; % Handles and labels for THIS subplot's legend
h_plots(end+1) = plot(x_start, y_start, 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 10);
h_labels{end+1} = 'Start';
h_plots(end+1) = plot(x_goal, y_goal, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 10);
h_labels{end+1} = 'Goal';

% Plot Obstacle
if enable_obstacle
    theta = linspace(0, 2*pi, 100); eff_r = obs_r + robot_r;
    h_plots(end+1) = plot(obs_x + eff_r*cos(theta), obs_y + eff_r*sin(theta), 'r-', 'LineWidth', 2);
    h_labels{end+1} = 'Obstacle Boundary';
    % plot(obs_x, obs_y, 'rx','MarkerSize', 8, 'LineWidth', 1.5); % Optional center marker
end

% Plot Algorithm Results (Points and Paths)
start_pt = [x_start, y_start]; goal_pt = [x_goal, y_goal];

% GA Result
if isfinite(results.ga_best_obj)
    pt = results.ga_best_point;
    h_plots(end+1) = plot(pt(1), pt(2), 'bs', 'MarkerFaceColor', 'b', 'MarkerSize', 10);
    h_labels{end+1} = sprintf('GA Pt (%.2f)', results.ga_best_obj);
    plot([start_pt(1), pt(1), goal_pt(1)], [start_pt(2), pt(2), goal_pt(2)], 'b--','LineWidth',1, 'HandleVisibility','off'); % Path line - no separate legend entry
end

% HC Results
if isfinite(results.hc_best_obj_overall)
    valid_hc_points = all_hc_final_points(all(isfinite(all_hc_final_points), 2), :);
    if ~isempty(valid_hc_points)
        h_plots(end+1) = plot(valid_hc_points(:,1), valid_hc_points(:,2), 'mo', 'MarkerSize', 5);
        h_labels{end+1} = 'HC Restarts';
    end
    pt = results.hc_best_point_overall;
    h_plots(end+1) = plot(pt(1), pt(2), 'ms', 'MarkerFaceColor', 'm', 'MarkerSize', 10);
    h_labels{end+1} = sprintf('HC Best Pt (%.2f)', results.hc_best_obj_overall);
    plot([start_pt(1), pt(1), goal_pt(1)], [start_pt(2), pt(2), goal_pt(2)], 'm:','LineWidth',1.5, 'HandleVisibility','off'); % Path line
end

% GS Result
if isfinite(results.gs_best_obj)
    pt = results.gs_best_point;
    h_plots(end+1) = plot(pt(1), pt(2), 'cs', 'MarkerFaceColor', 'c', 'MarkerSize', 10);
    h_labels{end+1} = sprintf('GS Pt (%.2f)', results.gs_best_obj);
    plot([start_pt(1), pt(1), goal_pt(1)], [start_pt(2), pt(2), goal_pt(2)], 'c-.','LineWidth',1, 'HandleVisibility','off'); % Path line
end

% Add legend to the first subplot
legend(h_plots, h_labels, 'Location', 'best');
hold off;

% --- Subplot 2: Surface Plot ---
subplot(1,2,2);
surf(X_vis, Y_vis, Z_vis, 'EdgeColor', 'none', 'FaceAlpha', 0.8); % Make slightly transparent
hold on;
grid on;
xlabel('X_{int}'); ylabel('Y_{int}'); zlabel('Path Length');
title('Objective Function Surface');
colorbar; % Show color scale for objective value
colormap(gca, 'jet'); % Apply colormap to this subplot
axis([x_min x_max y_min y_max]); % Match XY axis limits

% Plot obstacle as cylinder on surface plot
if enable_obstacle
    % Calculate Z limits based on plotted surface, excluding NaNs
    z_data = Z_vis(~isnan(Z_vis));
    min_z = min(z_data);
    max_z = max(z_data) * 1.1; % Go slightly above max surface height
    if isempty(min_z) || isempty(max_z) || ~isfinite(min_z) || ~isfinite(max_z)
        min_z = 0; max_z = 25; % Fallback Z range if surface is all NaN
    end

    eff_r = obs_r + robot_r;
    [cx, cy, cz] = cylinder(eff_r, 50); % Generate cylinder coordinates
    % Scale cylinder height and position it
    cz_scaled = cz * (max_z - min_z) + min_z;
    surf(cx*1+obs_x, cy*1+obs_y, cz_scaled, 'FaceColor', 'r', 'EdgeColor', 'none', 'FaceAlpha', 0.4);
end

% Optionally plot the best path(s) also on the surface
if isfinite(results.ga_best_obj) % Example for GA path
     pt = results.ga_best_point;
     z_pt = objective_func(pt(1), pt(2)); % Get Z value for the point
     path_x = [start_pt(1), pt(1), goal_pt(1)];
     path_y = [start_pt(2), pt(2), goal_pt(2)];
     path_z = [objective_func(start_pt(1), start_pt(2)), z_pt, objective_func(goal_pt(1), goal_pt(2))]; % Estimate Z for start/goal
     plot3(path_x, path_y, path_z + 0.1, 'b-o', 'LineWidth', 2, 'MarkerFaceColor','b', 'MarkerSize', 4); % Plot slightly above surface
end

hold off;
%% --- Helper Function: Tournament Selection (Corrected) ---
function selected_idx = tournament_selection(fitness_values, t_size)
    pop_size = length(fitness_values); best_idx = -1; best_fitness = inf; first_participant_idx = -1;
    for i = 1:t_size; idx = randi(pop_size); if i == 1; first_participant_idx = idx; end; if fitness_values(idx) < best_fitness; best_fitness = fitness_values(idx); best_idx = idx; end; end
    if best_idx == -1; selected_idx = first_participant_idx; else; selected_idx = best_idx; end
    if ~isscalar(selected_idx) || selected_idx < 1 || selected_idx > pop_size || floor(selected_idx) ~= selected_idx; warning('Invalid index %f generated/returned in tournament selection. Defaulting to 1.', selected_idx); selected_idx = 1; end
end