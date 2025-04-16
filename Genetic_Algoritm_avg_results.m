% MATLAB Code for GA Analysis (Section 5) -
% Runs the Improved GA multiple times for a specific scenario
% Optimizing waypoint positions only, minimizing path length.

clear; clc; close all; rng('shuffle'); % Use shuffle for different results each time

%% --- GA Function Definition ---
% Encapsulating the improved GA into a function
function results = run_ga_pos_experiment(problem, ga_params, penalty, run_id)
    fprintf('\n--- Starting GA Position-Only Run %d ---\n', run_id);
    % --- Initialization ---
    population = zeros(ga_params.pop_size, problem.n_vars);
    fitness = inf(ga_params.pop_size, 1);
    objective = inf(ga_params.pop_size, 1);
    penalty_val = inf(ga_params.pop_size, 1);
    init_feasible_count = 0; max_init_attempts = ga_params.pop_size * 100; attempt = 0;
    while init_feasible_count < ga_params.pop_size && attempt < max_init_attempts
        coords = problem.x_min + (problem.x_max - problem.x_min) * rand(1, problem.n_vars); % Chromosome is now just coords
        chromosome = coords; % Simplify variable name
        [temp_fitness, ~, temp_penalty] = evaluate_solution_pos(chromosome, problem, penalty);
        if temp_penalty < 1e6
             init_feasible_count = init_feasible_count + 1;
             population(init_feasible_count, :) = chromosome;
             fitness(init_feasible_count) = temp_fitness;
        end
        attempt = attempt + 1;
    end
    if init_feasible_count < ga_params.pop_size; error('GA Run %d: Could not initialize enough feasible individuals.', run_id); end
    if init_feasible_count < ga_params.pop_size % Fill remaining
        for i=(init_feasible_count+1):ga_params.pop_size; population(i,:) = population(mod(i-1, init_feasible_count)+1, :); fitness(i) = fitness(mod(i-1, init_feasible_count)+1); end
    end
    for i = 1:ga_params.pop_size % Evaluate full initial population
        [fitness(i), objective(i), penalty_val(i)] = evaluate_solution_pos(population(i,:), problem, penalty);
    end

    [best_fitness_overall, best_idx_overall] = min(fitness);
    best_solution_overall = population(best_idx_overall, :);
    generations_no_improvement = 0;
    history_ga = zeros(ga_params.num_generations, 1); history_ga(1) = best_fitness_overall;

    % --- Evolution Loop ---
    tic; % Start timer
    gen = 0;
    for gen = 1:ga_params.num_generations
        % Adaptive Mutation
        current_gen_fraction = gen / ga_params.num_generations;
        current_mutation_rate = ga_params.mutation_initial_rate - current_gen_fraction * (ga_params.mutation_initial_rate - ga_params.mutation_final_rate);
        current_mutation_sigma = ga_params.mutation_initial_sigma - current_gen_fraction * (ga_params.mutation_initial_sigma - ga_params.mutation_final_sigma);

        new_population = zeros(size(population));
        new_fitness = inf(size(fitness));
        new_objective = inf(size(objective));
        new_penalty_val = inf(size(penalty_val));

        % Elitism
        [~, sort_idx] = sort(fitness);
        elite_indices = sort_idx(1:ga_params.elitism_count);
        new_population(1:ga_params.elitism_count, :) = population(elite_indices, :);
        new_fitness(1:ga_params.elitism_count) = fitness(elite_indices);
        new_objective(1:ga_params.elitism_count) = objective(elite_indices);
        new_penalty_val(1:ga_params.elitism_count) = penalty_val(elite_indices);

        % Generate remaining population
        for i = (ga_params.elitism_count + 1) : 2 : ga_params.pop_size
            parent1_idx = tournament_selection(fitness, ga_params.tournament_size);
            parent2_idx = tournament_selection(fitness, ga_params.tournament_size);
            while parent2_idx == parent1_idx; parent2_idx = tournament_selection(fitness, ga_params.tournament_size); end
            parent1 = population(parent1_idx, :); parent2 = population(parent2_idx, :);
            if rand < ga_params.crossover_prob; alpha_co = rand; child1 = alpha_co*parent1+(1-alpha_co)*parent2; child2 = (1-alpha_co)*parent1+alpha_co*parent2; else; child1 = parent1; child2 = parent2; end
            if rand < current_mutation_rate; child1 = child1 + randn(1, problem.n_vars) * current_mutation_sigma; end
            if (i + 1 <= ga_params.pop_size) && (rand < current_mutation_rate); child2 = child2 + randn(1, problem.n_vars) * current_mutation_sigma; end
            [new_fitness(i), new_objective(i), new_penalty_val(i)] = evaluate_solution_pos(child1, problem, penalty); new_population(i, :) = child1;
            if i + 1 <= ga_params.pop_size; [new_fitness(i+1), new_objective(i+1), new_penalty_val(i+1)] = evaluate_solution_pos(child2, problem, penalty); new_population(i+1, :) = child2; end
        end
        population = new_population; fitness = new_fitness; objective = new_objective; penalty_val = new_penalty_val;

        % Local Search (Optional)
        if ga_params.use_local_search
            [current_best_fitness_ls, current_best_idx_ls] = min(fitness);
            if isfinite(current_best_fitness_ls)
                best_individual_ls = population(current_best_idx_ls,:);
                improved_individual_ls = basic_local_search_pos(best_individual_ls, problem, penalty);
                [improved_fitness_ls, imp_obj, imp_pen] = evaluate_solution_pos(improved_individual_ls, problem, penalty);
                if improved_fitness_ls < current_best_fitness_ls
                    population(current_best_idx_ls,:) = improved_individual_ls; fitness(current_best_idx_ls) = improved_fitness_ls; objective(current_best_idx_ls) = imp_obj; penalty_val(current_best_idx_ls) = imp_pen;
                end
            end
        end

        % Update overall best and check termination
        [best_fitness_gen, ~] = min(fitness);
        if isinf(best_fitness_gen) && any(isfinite(fitness)); best_fitness_gen = min(fitness(isfinite(fitness))); elseif isinf(best_fitness_gen); best_fitness_gen = history_ga(max(1, gen-1)); end
        history_ga(gen) = best_fitness_gen;
        if best_fitness_gen < best_fitness_overall; best_fitness_overall = best_fitness_gen; best_idx_temp = find(fitness == best_fitness_gen, 1); best_solution_overall = population(best_idx_temp, :); generations_no_improvement = 0; else; generations_no_improvement = generations_no_improvement + 1; end
        if generations_no_improvement >= ga_params.no_improvement_limit; fprintf('Run %d terminated early at Gen %d (no improvement).\n', run_id, gen); history_ga = history_ga(1:gen); break; end
    end
    run_time = toc;

    % Final evaluation
    [final_fitness, final_objective, final_penalty] = evaluate_solution_pos(best_solution_overall, problem, penalty);

    % Store results
    results.fitness = final_fitness;
    results.objective = final_objective; % Path length only
    results.penalty = final_penalty;
    results.time = run_time;
    results.generations = gen;
    results.solution = best_solution_overall;
    results.history = history_ga;

    fprintf('Run %d Finished. Fitness=%.4f (PathLen=%.4f, Pen=%.4f), Gens=%d, Time=%.2fs\n', ...
            run_id, results.fitness, results.objective, results.penalty, results.generations, results.time);
end % End of run_ga_pos_experiment function

%% --- Define Scenarios  ---
% Scenario 1: Simple Environment
problem_simple = define_problem_structure_pos();
problem_simple.obstacles = [ 7, 6, 1.0 ]; % One central obstacle

% Scenario 2: Moderate Environment
problem_moderate = define_problem_structure_pos();
problem_moderate.obstacles = [  7, 6, 1.0; 4, 1, 1.0];

% Scenario 3: Complex Environment
problem_complex = define_problem_structure_pos();
problem_complex.obstacles = [   7, 6, 1.0; 4, 1, 1.0; 12, 10, 1;];

%% --- Setup GA Parameters & Penalties ---
ga_params = struct();
ga_params.pop_size = 50;
ga_params.num_generations = 300;
ga_params.elitism_count = 2;
ga_params.tournament_size = 3;
ga_params.crossover_prob = 0.8;
ga_params.mutation_initial_rate = 0.1;
ga_params.mutation_final_rate = 0.01;
ga_params.mutation_initial_sigma = 1; % Tune
ga_params.mutation_final_sigma = 0.01;  % Tune
ga_params.no_improvement_limit = 50;
ga_params.use_local_search = false; % For local search 
penalty = struct();
penalty.K_boundary = 500.0;         % Keep
penalty.K_obstacle_waypoint = 1000.0; % Keep
penalty.K_obstacle_segment = 1500.0; % Keep

%% --- Perform Multiple Runs for One Scenario (e.g., Moderate) ---
num_runs = 10;
target_scenario = problem_moderate; % Choose scenario
scenario_name = 'Moderate';

fprintf('\n=== Running %d GA experiments for %s Scenario ===\n', num_runs, scenario_name);

run_results = cell(num_runs, 1);
all_path_lengths = nan(num_runs, 1);
all_penalties = nan(num_runs, 1);
all_times = nan(num_runs, 1);
all_generations = nan(num_runs, 1);
success_count = 0;
success_threshold = 1e-4;

for r = 1:num_runs
    run_results{r} = run_ga_pos_experiment(target_scenario, ga_params, penalty, r);
    all_path_lengths(r) = run_results{r}.objective; % Objective is now just path length
    all_penalties(r) = run_results{r}.penalty;
    all_times(r) = run_results{r}.time;
    all_generations(r) = run_results{r}.generations;
    if run_results{r}.penalty <= success_threshold
        success_count = success_count + 1;
    end
end

%% --- Calculate and Display Statistics ---
valid_indices = isfinite(all_path_lengths) & isfinite(all_penalties);
valid_path_lengths = all_path_lengths(valid_indices);

avg_path_length = mean(valid_path_lengths);
std_dev_path_length = std(valid_path_lengths);
avg_time = mean(all_times);
avg_generations = mean(all_generations);
success_rate = (success_count / num_runs) * 100;

fprintf('\n=== GA Position-Only Performance Summary for %s (%d Runs) ===\n', scenario_name, num_runs);
fprintf('Avg. Path Length: %.4f\n', avg_path_length);
fprintf('Std. Deviation (Path Length): %.4f\n', std_dev_path_length);
fprintf('Avg. Convergence Gen: %.1f\n', avg_generations);
fprintf('Success Rate (Penalty < %.1e): %.1f %%\n', success_threshold, success_rate);
fprintf('Avg. Computation Time: %.4f seconds\n', avg_time);

%% --- Plotting Example: Best Run Result & Convergence Histories ---
best_run_idx = -1; best_run_objective = inf;
for r = 1:num_runs
    if isfinite(run_results{r}.objective) && run_results{r}.penalty <= success_threshold
        if run_results{r}.objective < best_run_objective; best_run_objective = run_results{r}.objective; best_run_idx = r; end
    end
end

figure;
% Plot convergence
subplot(1,2,1); hold on;
for r = 1:num_runs; plot(1:run_results{r}.generations, run_results{r}.history, 'Color', [0.7 0.7 1.0]); end
if best_run_idx > 0; plot(1:run_results{best_run_idx}.generations, run_results{best_run_idx}.history, 'b-', 'LineWidth', 1.5); title(sprintf('GA Conv. (%s) - Best Run', scenario_name));
else; title(sprintf('GA Conv. (%s) - No success', scenario_name)); end
xlabel('Generation'); ylabel('Best Fitness (PathLen + Pen)'); grid on; set(gca, 'YScale', 'log'); hold off;

% Plot path
subplot(1,2,2); hold on; grid on; axis equal;
title(sprintf('Best Path Found (%s)', scenario_name)); xlabel('X'); ylabel('Y');
xlim([target_scenario.x_min, target_scenario.x_max]); ylim([target_scenario.y_min, target_scenario.y_max]);
if ~isempty(target_scenario.obstacles); theta = linspace(0, 2*pi, 100); for k = 1:size(target_scenario.obstacles, 1); obs_x = target_scenario.obstacles(k, 1); obs_y = target_scenario.obstacles(k, 2); obs_r = target_scenario.obstacles(k, 3); eff_r = obs_r + target_scenario.robot_radius; plot(obs_x + eff_r*cos(theta), obs_y + eff_r*sin(theta), 'r-', 'LineWidth', 1.5); plot(obs_x, obs_y, 'rx'); end; end
if best_run_idx > 0
    [waypoints_x, waypoints_y] = decode_chromosome_pos(run_results{best_run_idx}.solution, target_scenario);
    plot(waypoints_x, waypoints_y, 'b-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'b', 'MarkerSize', 6);
    plot(target_scenario.x_start, target_scenario.y_start, 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 10);
    plot(target_scenario.x_goal, target_scenario.y_goal, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 10);
    subtitle(sprintf('Best Run PathLen: %.2f, Pen: %.2f', run_results{best_run_idx}.objective, run_results{best_run_idx}.penalty));
else; text(mean(xlim), mean(ylim), 'No successful run.', 'HorizontalAlignment', 'center'); end
hold off;


%% --- Helper Functions ---

% --- define_problem_structure_pos Function ---
function prob = define_problem_structure_pos()
    % Defines the structure for position-only optimization
    prob.n_waypoints = 8;
    prob.x_start = 0; prob.y_start = 0;
    prob.x_goal = 15; prob.y_goal = 15;
    prob.x_min = -2; prob.x_max = 17;
    prob.y_min = -2; prob.y_max = 17;
    prob.robot_radius = 0.5;
    prob.obstacles = [];
    prob.alpha = 1.0; % Weight for path length (beta = 0 implicitly)
    prob.n_vars = 2 * (prob.n_waypoints - 2); 
end

% --- evaluate_solution_pos Function ---
function [fitness, objective, total_penalty] = evaluate_solution_pos(chromosome, problem, penalty)
    % Evaluates fitness for position-only optimization
    [waypoints_x, waypoints_y] = decode_chromosome_pos(chromosome, problem);
    n_wp = problem.n_waypoints; waypoints = [waypoints_x, waypoints_y];

    % 1. Objective (Path Length Only)
    path_len = sum(sqrt(sum(diff(waypoints).^2, 2)));
    objective = problem.alpha * path_len; % Alpha is typically 1 here

    % 2. Penalties (Position-based only)
    total_penalty = 0;
    % a) Boundary
    x_coords = waypoints(2:n_wp-1, 1); y_coords = waypoints(2:n_wp-1, 2);
    total_penalty = total_penalty + penalty.K_boundary * sum(max(0, problem.x_min - x_coords).^2);
    total_penalty = total_penalty + penalty.K_boundary * sum(max(0, x_coords - problem.x_max).^2);
    total_penalty = total_penalty + penalty.K_boundary * sum(max(0, problem.y_min - y_coords).^2);
    total_penalty = total_penalty + penalty.K_boundary * sum(max(0, y_coords - problem.y_max).^2);
    % b) Obstacles (Waypoints & Segments)
    if ~isempty(problem.obstacles)
        min_obs_dist_sq_wp = (problem.obstacles(:,3) + problem.robot_radius).^2;
        min_obs_dist_sq_seg = min_obs_dist_sq_wp; % Use same buffer
        for i = 1:n_wp % Waypoint check
            for k = 1:size(problem.obstacles, 1)
                dist_sq = (waypoints(i, 1) - problem.obstacles(k, 1))^2 + (waypoints(i, 2) - problem.obstacles(k, 2))^2;
                total_penalty = total_penalty + penalty.K_obstacle_waypoint * max(0, min_obs_dist_sq_wp(k) - dist_sq)^2;
            end
        end
        for i = 1:(n_wp - 1) % Segment midpoint check
            mid_x = (waypoints(i, 1) + waypoints(i+1, 1)) / 2;
            mid_y = (waypoints(i, 2) + waypoints(i+1, 2)) / 2;
            for k = 1:size(problem.obstacles, 1)
                 dist_sq = (mid_x - problem.obstacles(k, 1))^2 + (mid_y - problem.obstacles(k, 2))^2;
                 total_penalty = total_penalty + penalty.K_obstacle_segment * max(0, min_obs_dist_sq_seg(k) - dist_sq)^2;
            end
        end
    end

    % 3. Combine
    fitness = objective + total_penalty;
    if isnan(fitness) || isinf(fitness) || ~isreal(fitness); fitness = inf; objective = inf; total_penalty = inf; end
end

% --- decode_chromosome_pos Function ---
function [waypoints_x, waypoints_y] = decode_chromosome_pos(chromosome, problem)
    % Decodes position-only chromosome
    n_wp = problem.n_waypoints;
    n_vars = problem.n_vars; % = 2 * (n_wp - 2)
    intermediate_coords = reshape(chromosome(1:n_vars), [n_wp - 2, 2]);
    waypoints_x = [problem.x_start; intermediate_coords(:,1); problem.x_goal];
    waypoints_y = [problem.y_start; intermediate_coords(:,2); problem.y_goal];
end

% --- tournament_selection Function  ---
function selected_idx = tournament_selection(fitness_values, t_size)
    pop_size = length(fitness_values); best_idx = -1; best_fitness = inf; first_participant_idx = -1;
    for i = 1:t_size; idx = randi(pop_size); if i == 1; first_participant_idx = idx; end; if isfinite(fitness_values(idx)) && fitness_values(idx) < best_fitness; best_fitness = fitness_values(idx); best_idx = idx; elseif isinf(fitness_values(idx)) && isinf(best_fitness) && best_idx == -1; best_idx = idx; end; end
    if best_idx == -1; selected_idx = first_participant_idx; else; selected_idx = best_idx; end
    if ~isscalar(selected_idx) || selected_idx < 1 || selected_idx > pop_size || floor(selected_idx) ~= selected_idx; warning('Invalid index %f generated/returned in tournament selection. Defaulting to 1.', selected_idx); selected_idx = 1; end
end

% --- basic_local_search_pos Function ---
function improved_solution = basic_local_search_pos(solution, problem, penalty)
    % Local search for position-only variables
    max_ls_iter = 10; step_size = 0.1; current_solution = solution;
    [current_fitness, ~, ~] = evaluate_solution_pos(current_solution, problem, penalty);
    if isinf(current_fitness); improved_solution = solution; return; end

    for iter = 1:max_ls_iter
        best_neighbor = current_solution; best_neighbor_fitness = current_fitness; improved = false;
        for j = 1:length(solution) % Perturb each variable (x or y coordinate)
            original_val = current_solution(j);
            % Try positive step
            neighbor_plus = current_solution; neighbor_plus(j) = neighbor_plus(j) + step_size;
            [fitness_plus, ~, ~] = evaluate_solution_pos(neighbor_plus, problem, penalty);
            if fitness_plus < best_neighbor_fitness; best_neighbor_fitness = fitness_plus; best_neighbor = neighbor_plus; improved = true; end
            % Try negative step
            neighbor_minus = current_solution; neighbor_minus(j) = neighbor_minus(j) - step_size;
            [fitness_minus, ~, ~] = evaluate_solution_pos(neighbor_minus, problem, penalty);
             if fitness_minus < best_neighbor_fitness; best_neighbor_fitness = fitness_minus; best_neighbor = neighbor_minus; improved = true; end
        end
        if improved
            current_solution = best_neighbor; current_fitness = best_neighbor_fitness;
        else
            step_size = step_size * 0.7; if step_size < 1e-5; break; end
        end
    end
    improved_solution = current_solution;
end