% =========================================================================
% MATLAB Script for Alpha Sensitivity Analysis (Sec 3.2 Table)
% =========================================================================

clear; clc; close all; rng('default'); % Use 'default' for reproducible results

%% --- Main Script Section ---

% --- Problem Setup ---
% Choose a scenario consistent with the report's Section 3.2 analysis
problem_base = define_problem_structure_pos();
problem_base.obstacles = [  7, 6, 1.0; 4, 1, 1.0]; % Moderate environment obstacles
scenario_name = 'Moderate'; % Scenario name for titles/logging

% --- Baseline GA Parameters & Penalties ---
% Use parameters consistent with your report/previous runs
ga_params = struct();
ga_params.pop_size = 50;
ga_params.num_generations = 250; % Adjust based on convergence speed
ga_params.elitism_count = 2;
ga_params.tournament_size = 3;
ga_params.crossover_prob = 0.8;
ga_params.mutation_initial_rate = 0.1;
ga_params.mutation_final_rate = 0.01;
ga_params.mutation_initial_sigma = 0.5; % Standard deviation for Gaussian mutation
ga_params.mutation_final_sigma = 0.01;
ga_params.no_improvement_limit = 40; % Generations to wait before early stopping
ga_params.use_local_search = false; % Set to true to enable basic_local_search

penalty = struct();
penalty.K_boundary = 500.0;         % Penalty weight for boundary violations
penalty.K_obstacle_waypoint = 1000.0; % Penalty weight for waypoint-obstacle collision
penalty.K_obstacle_segment = 1500.0;  % Penalty weight for segment-obstacle collision (midpoint check)

% --- Alpha Value Testing ---
alpha_values_to_test = [1.0, 0.8, 0.5, 0.2, 0.0]; % Alpha values from report Sec 3.2 [cite: 45]
num_alphas = length(alpha_values_to_test);
num_runs_per_alpha = 10; % Number of GA runs for each alpha value (for statistical significance)
success_threshold = 1e-3; % Penalty value below which a run is considered successful (feasible)

% --- Results Storage ---
results_table = table('Size', [num_alphas, 4], ...
                      'VariableTypes', {'double', 'double', 'double', 'double'}, ...
                      'VariableNames', {'Alpha', 'AvgPathLength', 'StdDevPathLength', 'SuccessRate'});

fprintf('\n===== Running GA for Alpha Sensitivity (%s Scenario) =====\n', scenario_name);
fprintf('Number of runs per alpha: %d\n', num_runs_per_alpha);
fprintf('Success Threshold (Max Penalty): %.1e\n', success_threshold);

% --- Main Loop for Alpha Testing ---
for i = 1:num_alphas
    current_alpha = alpha_values_to_test(i);
    fprintf('\n------------------------------------------\n');
    fprintf('--- Testing Alpha = %.2f ---\n', current_alpha);
    fprintf('------------------------------------------\n');

    % Set the current alpha value in the problem structure
    current_problem = problem_base;
    current_problem.alpha = current_alpha;

    % Store results for this alpha value across multiple runs
    run_path_lengths = nan(num_runs_per_alpha, 1);
    run_penalties = nan(num_runs_per_alpha, 1);
    run_times = nan(num_runs_per_alpha, 1);
    successful_run_count = 0;

    for r = 1:num_runs_per_alpha
        fprintf('--- Run %d/%d for alpha=%.2f starting... ---\n', r, num_runs_per_alpha, current_alpha);
         try
             run_id = (i-1)*num_runs_per_alpha + r; % Unique ID for the run
             % Call the main GA function
             result = run_ga_pos_experiment(current_problem, ga_params, penalty, run_id);

             run_penalties(r) = result.penalty; % Store final penalty
             run_times(r) = result.time;     % Store run time

             % Check for success (finite path length and low penalty)
             if isfinite(result.path_length) && result.penalty <= success_threshold
                 run_path_lengths(r) = result.path_length; % Store raw path length for successful run
                 successful_run_count = successful_run_count + 1;
             else
                 run_path_lengths(r) = NaN; % Mark unsuccessful run's path length as NaN
                 fprintf('Run %d (alpha=%.2f, run %d) FAILED or INFEASIBLE (Penalty: %.4f, PathLen: %.4f).\n', run_id, current_alpha, r, result.penalty, result.path_length);
             end
         catch ME
             fprintf('ERROR during GA run %d (alpha = %.2f, run %d): %s\n', run_id, current_alpha, r, ME.message);
             fprintf('Error details: %s\n', ME.getReport('basic'));
             run_path_lengths(r) = NaN;
             run_penalties(r) = NaN;
             run_times(r) = NaN;
         end
    end

    % --- Calculate Statistics for this Alpha Value ---
    valid_indices = ~isnan(run_path_lengths); % Indices of successful runs
    if any(valid_indices)
        avg_path = mean(run_path_lengths(valid_indices));
        std_path = std(run_path_lengths(valid_indices));
    else
        avg_path = NaN; % Indicate no successful runs found
        std_path = NaN;
    end
    success_rate = (successful_run_count / num_runs_per_alpha) * 100;
    avg_time = mean(run_times(~isnan(run_times))); % Avg time over completed runs

    % --- Store Aggregate Results ---
    results_table.Alpha(i) = current_alpha;
    results_table.AvgPathLength(i) = avg_path;
    results_table.StdDevPathLength(i) = std_path;
    results_table.SuccessRate(i) = success_rate;

    fprintf('\n--- RESULTS for Alpha = %.2f ---\n', current_alpha);
    fprintf('Avg Path Length (Successful Runs): %.4f (Std Dev: %.4f)\n', avg_path, std_path);
    fprintf('Success Rate (Penalty <= %.1e): %.1f%%\n', success_threshold, success_rate);
    fprintf('Average Run Time: %.4f seconds\n', avg_time);
    fprintf('------------------------------------------\n');

end % End loop over alpha values

%% --- Display Final Results Table ---
fprintf('\n\n============================================================\n');
fprintf('      Alpha Sensitivity Results Summary (%s Scenario)\n', scenario_name);
fprintf('============================================================\n');
disp(results_table);
fprintf('============================================================\n');

%% --- Plot Results ---
figure('Name', ['Alpha Sensitivity - ' scenario_name]);
errorbar(results_table.Alpha, results_table.AvgPathLength, results_table.StdDevPathLength, ...
         '-o', 'LineWidth', 1.5, 'MarkerSize', 8, 'CapSize', 5, 'MarkerFaceColor', 'b');
title(['Avg. Path Length vs. Alpha Weight (' scenario_name ')']);
xlabel('Alpha (Weight for Path Length in Objective Function)');
ylabel('Average Path Length (Successful Runs)');
grid on;
set(gca, 'XDir','reverse'); % Match report table order (high alpha on left)
xticks(sort(alpha_values_to_test)); % Ensure all tested alphas are labeled
xlim([min(alpha_values_to_test)-0.1, max(alpha_values_to_test)+0.1]); % Add padding

% Add success rate labels to the plot
hold on;
for i = 1:num_alphas
    if isfinite(results_table.AvgPathLength(i)) % Only label points with valid data
        text(results_table.Alpha(i), results_table.AvgPathLength(i), ...
            sprintf('  SR: %.0f%%', results_table.SuccessRate(i)), ...
            'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', ...
            'FontSize', 9, 'Color', [0.1 0.1 0.8]);
    end
end
hold off;

% Adjust y-axis limits if needed
valid_avg_paths = results_table.AvgPathLength(isfinite(results_table.AvgPathLength));
if ~isempty(valid_avg_paths)
    ylim([min(valid_avg_paths)*0.95, max(valid_avg_paths)*1.05]);
else
    ylim([0, 1]); % Default if no data
end


disp(' ');
disp('Script finished successfully.');
disp('See the generated table in the command window and the plot figure.');

% =========================================================================
%                       HELPER FUNCTION DEFINITIONS
% =========================================================================

%% --- GA Function Definition ---
% Runs the Genetic Algorithm for one configuration
function results = run_ga_pos_experiment(problem, ga_params, penalty, run_id)
    % Display start message with alpha
    fprintf('Run %d: Starting GA (alpha=%.2f). Pop=%d, Gens=%d\n', ...
            run_id, problem.alpha, ga_params.pop_size, ga_params.num_generations);

    % --- Initialization ---
    population = zeros(ga_params.pop_size, problem.n_vars);
    fitness = inf(ga_params.pop_size, 1);
    objective = inf(ga_params.pop_size, 1); % Weighted objective part
    penalty_val = inf(ga_params.pop_size, 1);
    path_len = inf(ga_params.pop_size, 1); % Raw path length

    init_feasible_count = 0;
    max_init_attempts = ga_params.pop_size * 200; % Max attempts to find feasible individuals
    attempt = 0;
    init_start_time = tic; % Time initialization

    while init_feasible_count < ga_params.pop_size && attempt < max_init_attempts
        % Generate random coordinates within bounds
        coords = problem.x_min + (problem.x_max - problem.x_min) * rand(1, problem.n_vars);
        chromosome = coords;

        % Check feasibility using penalty from evaluation function
        [~, ~, temp_penalty, ~] = evaluate_solution_pos(chromosome, problem, penalty);

        % Use a slightly relaxed penalty threshold for initial population generation
        init_feasibility_threshold = 1e-1; % Relaxed from success_threshold
        if temp_penalty <= init_feasibility_threshold
             init_feasible_count = init_feasible_count + 1;
             population(init_feasible_count, :) = chromosome;
             % Full evaluation (fitness, objective, path_len) will happen below
        end
        attempt = attempt + 1;
    end
    init_time = toc(init_start_time);

    % --- Handle Initialization Outcome ---
    if init_feasible_count == 0
        error('GA Run %d: Failed to initialize ANY feasible individuals after %d attempts (%.2f s). Check constraints, penalties, or initialization bounds.', ...
              run_id, attempt, init_time);
    elseif init_feasible_count < ga_params.pop_size
         warning('GA Run %d: Initialized only %d/%d feasible individuals after %d attempts (%.2f s). Filling rest by copying.', ...
                 run_id, init_feasible_count, ga_params.pop_size, attempt, init_time);
         % Fill remaining population slots by copying the feasible individuals found
         for i = (init_feasible_count + 1) : ga_params.pop_size
             copy_idx = mod(i - 1, init_feasible_count) + 1;
             population(i, :) = population(copy_idx, :);
         end
    else
        fprintf('Run %d: Successfully initialized %d feasible individuals (%d attempts, %.2f s).\n', ...
                run_id, init_feasible_count, attempt, init_time);
    end

    % --- Evaluate Full Initial Population ---
    for i = 1:ga_params.pop_size
        [fitness(i), objective(i), penalty_val(i), path_len(i)] = evaluate_solution_pos(population(i,:), problem, penalty);
    end

    % --- Find Initial Best ---
    [best_fitness_overall, best_idx_overall] = min(fitness);
    if ~isempty(best_idx_overall) && isfinite(best_fitness_overall)
      best_idx_overall = best_idx_overall(1); % Take first if multiple minima
      best_solution_overall = population(best_idx_overall, :);
      best_path_len_overall = path_len(best_idx_overall); % Store initial best path length
    else
        % Handle case where all initial fitnesses might be Inf
        best_fitness_overall = Inf;
        best_solution_overall = population(1, :); % Assign arbitrarily
        best_path_len_overall = Inf;
        warning('Run %d: No finite fitness found in initial population.', run_id);
    end

    generations_no_improvement = 0;
    history_ga = nan(ga_params.num_generations, 1); % Use NaN for logging gaps
    if isfinite(best_fitness_overall); history_ga(1) = best_fitness_overall; end

    % --- Evolution Loop ---
    fprintf('Run %d: Starting evolution...\n', run_id);
    evolution_start_time = tic;
    gen = 0; % Initialize generation counter
    termination_reason = 'Max Generations Reached'; % Default reason

    for gen = 1:ga_params.num_generations
        % --- Adaptive Mutation Parameters ---
        current_gen_fraction = gen / ga_params.num_generations;
        current_mutation_rate = ga_params.mutation_initial_rate - current_gen_fraction * (ga_params.mutation_initial_rate - ga_params.mutation_final_rate);
        current_mutation_sigma = ga_params.mutation_initial_sigma - current_gen_fraction * (ga_params.mutation_initial_sigma - ga_params.mutation_final_sigma);

        % --- Preallocate New Generation ---
        new_population = zeros(size(population));
        new_fitness = inf(size(fitness));
        new_objective = inf(size(objective));
        new_penalty_val = inf(size(penalty_val));
        new_path_len = inf(size(path_len));

        % --- Elitism ---
        num_elites_copied = 0;
        valid_fitness_indices = find(isfinite(fitness));
        if ~isempty(valid_fitness_indices)
            [~, sorted_valid_original_indices] = sort(fitness(valid_fitness_indices));
            num_elites_to_select = min(ga_params.elitism_count, length(valid_fitness_indices));
            elite_indices_in_valid = sorted_valid_original_indices(1:num_elites_to_select);
            elite_indices = valid_fitness_indices(elite_indices_in_valid);
            num_elites_copied = length(elite_indices);

            if num_elites_copied > 0
                new_population(1:num_elites_copied, :) = population(elite_indices, :);
                new_fitness(1:num_elites_copied) = fitness(elite_indices);
                new_objective(1:num_elites_copied) = objective(elite_indices);
                new_penalty_val(1:num_elites_copied) = penalty_val(elite_indices);
                new_path_len(1:num_elites_copied) = path_len(elite_indices); % Copy elite path lengths
            end
        end
        % If no elites found (e.g., all Inf fitness), num_elites_copied remains 0

        % --- Generate Offspring (Crossover and Mutation) ---
        start_index_for_children = num_elites_copied + 1;
        for i = start_index_for_children : 2 : ga_params.pop_size % Step by 2 for pairs
            % --- Selection ---
            parent1_idx = tournament_selection(fitness, ga_params.tournament_size);
            parent2_idx = tournament_selection(fitness, ga_params.tournament_size);
            % Ensure distinct parents
            while parent2_idx == parent1_idx
                parent2_idx = tournament_selection(fitness, ga_params.tournament_size);
            end
            parent1 = population(parent1_idx, :);
            parent2 = population(parent2_idx, :);

            % --- Crossover (Blend/Average Crossover) ---
            if rand < ga_params.crossover_prob
                alpha_co = rand; % Blend factor
                child1 = alpha_co*parent1 + (1-alpha_co)*parent2;
                child2 = (1-alpha_co)*parent1 + alpha_co*parent2;
            else
                % No crossover, children are clones of parents
                child1 = parent1;
                child2 = parent2;
            end

            % --- Mutation ---
            if rand < current_mutation_rate
                 mutation_noise1 = randn(1, problem.n_vars) * current_mutation_sigma;
                 child1 = child1 + mutation_noise1;
                 % Simple bounds enforcement after mutation
                 child1 = max(problem.x_min, min(problem.x_max, child1));
            end
            % Mutate child2 only if it exists in this iteration
            if (i + 1 <= ga_params.pop_size) && (rand < current_mutation_rate)
                 mutation_noise2 = randn(1, problem.n_vars) * current_mutation_sigma;
                 child2 = child2 + mutation_noise2;
                 child2 = max(problem.x_min, min(problem.x_max, child2));
            end

            % --- Evaluate Offspring ---
            [new_fitness(i), new_objective(i), new_penalty_val(i), new_path_len(i)] = evaluate_solution_pos(child1, problem, penalty);
            new_population(i, :) = child1;

            if i + 1 <= ga_params.pop_size % Evaluate child2 if it exists
                [new_fitness(i+1), new_objective(i+1), new_penalty_val(i+1), new_path_len(i+1)] = evaluate_solution_pos(child2, problem, penalty);
                new_population(i+1, :) = child2;
            end
        end % End offspring generation loop

        % --- Replace Old Population ---
        population = new_population;
        fitness = new_fitness;
        objective = new_objective;
        penalty_val = new_penalty_val;
        path_len = new_path_len; % Update path lengths for the new generation

        % --- Local Search (Optional) ---
        if ga_params.use_local_search
            [current_best_fitness_ls, current_best_idx_ls] = min(fitness);
             if isfinite(current_best_fitness_ls) && ~isempty(current_best_idx_ls)
                best_idx_ls = current_best_idx_ls(1); % Take first if multiple
                best_individual_ls = population(best_idx_ls,:);
                % Run local search
                improved_individual_ls = basic_local_search_pos(best_individual_ls, problem, penalty);
                % Evaluate the improved individual
                [improved_fitness_ls, imp_obj, imp_pen, imp_path] = evaluate_solution_pos(improved_individual_ls, problem, penalty);
                % Replace if better
                if improved_fitness_ls < current_best_fitness_ls
                    population(best_idx_ls,:) = improved_individual_ls;
                    fitness(best_idx_ls) = improved_fitness_ls;
                    objective(best_idx_ls) = imp_obj;
                    penalty_val(best_idx_ls) = imp_pen;
                    path_len(best_idx_ls) = imp_path; % Update path length too
                    % fprintf('LS Improved Gen %d: %.4f -> %.4f\n', gen, current_best_fitness_ls, improved_fitness_ls);
                end
            end
        end

        % --- Update Overall Best and Check Termination ---
        [best_fitness_gen, best_idx_gen] = min(fitness);
        best_path_len_gen = Inf; % Path length of the best individual in current gen

        % Handle cases where min fitness might still be Inf
        if isinf(best_fitness_gen) && any(isfinite(fitness))
             finite_indices = find(isfinite(fitness));
             [best_fitness_gen, idx_in_finite] = min(fitness(finite_indices)); % Find min among finite
             best_idx_gen = finite_indices(idx_in_finite(1)); % Get original index
             if ~isempty(best_idx_gen); best_path_len_gen = path_len(best_idx_gen(1)); end
        elseif isfinite(best_fitness_gen) && ~isempty(best_idx_gen)
            best_idx_gen = best_idx_gen(1); % Take first if multiple
            best_path_len_gen = path_len(best_idx_gen);
        else
             % All Inf case: best_fitness_gen remains Inf, best_idx_gen is empty
             best_fitness_gen = Inf;
             best_idx_gen = [];
             best_path_len_gen = Inf;
        end

        % --- Log History ---
        if isfinite(best_fitness_gen); history_ga(gen) = best_fitness_gen; else; history_ga(gen) = NaN; end

        % --- Update Overall Best Solution Found So Far ---
        if best_fitness_gen < best_fitness_overall
            best_fitness_overall = best_fitness_gen;
            if ~isempty(best_idx_gen) % Ensure we have a valid index
                best_solution_overall = population(best_idx_gen, :);
                best_path_len_overall = path_len(best_idx_gen); % Update best raw path length
            end
            generations_no_improvement = 0; % Reset counter
        else
            generations_no_improvement = generations_no_improvement + 1;
        end

        % --- Termination Condition Check ---
        if generations_no_improvement >= ga_params.no_improvement_limit
            termination_reason = 'No Improvement Limit Reached';
            fprintf('Run %d terminated early at Gen %d (%s).\n', run_id, gen, termination_reason);
            break; % Exit generation loop
        end

        % --- Optional: Print Progress ---
         if mod(gen, 50) == 0 || gen == 1
             fprintf('Run %d Gen %d: Best Fitness=%.4f (PathLen=%.4f) (Overall BestFit=%.4f)\n', ...
                     run_id, gen, best_fitness_gen, best_path_len_gen, best_fitness_overall);
         end
    end % End of generation loop

    evolution_time = toc(evolution_start_time);

    % --- Final Evaluation & Results ---
    % Use the tracked best values
    final_fitness = best_fitness_overall;
    final_path_length = best_path_len_overall;
    final_solution = best_solution_overall;

    % Re-evaluate final solution to get consistent penalty/objective parts if needed
    [~, final_objective_weighted, final_penalty, ~] = evaluate_solution_pos(final_solution, problem, penalty);

    % --- Store Results ---
    results.fitness = final_fitness; % Best fitness found
    results.objective_weighted = final_objective_weighted; % Weighted obj part of best solution
    results.penalty = final_penalty;         % Penalty of best solution
    results.path_length = final_path_length;   % Raw path length of best solution
    results.time = init_time + evolution_time; % Total time
    results.generations = gen;                 % Actual number of generations run
    results.solution = final_solution;       % Best chromosome found
    results.history = history_ga(1:gen);     % Trimmed fitness history
    results.termination_reason = termination_reason;

    fprintf('Run %d Finished (%s). Fitness=%.4f (PathLen=%.4f, Pen=%.4f), Gens=%d, Time=%.2fs\n', ...
            run_id, results.termination_reason, results.fitness, results.path_length, results.penalty, results.generations, results.time);

end % End of run_ga_pos_experiment function


%% --- define_problem_structure_pos Function ---
% Defines the optimization problem structure
function prob = define_problem_structure_pos()
    prob.n_waypoints = 8; % Number of waypoints including start and end
    prob.x_start = 0; prob.y_start = 0; % Start position
    prob.x_goal = 15; prob.y_goal = 15; % Goal position
    % Workspace boundaries
    prob.x_min = -2; prob.x_max = 17;
    prob.y_min = -2; prob.y_max = 17;
    prob.robot_radius = 0.5; % Robot radius for collision checking
    prob.obstacles = []; % Obstacles defined as [x_center, y_center, radius]
                         % This will be set by the main script section
    prob.alpha = 1.0;    % Weight for path length (objective term)
                         % This will be overwritten in the main loop
    % Number of variables to optimize: (n_waypoints - 2) x/y coordinates
    prob.n_vars = 2 * (prob.n_waypoints - 2);
end % End of define_problem_structure_pos function


%% --- evaluate_solution_pos Function ---
% Evaluates the fitness of a given chromosome (waypoint coordinates)
% Returns: fitness, weighted_objective, penalty, raw_path_length
function [fitness, objective_component, total_penalty, path_len] = evaluate_solution_pos(chromosome, problem, penalty_weights)
    % Decode chromosome into waypoints
    [waypoints_x, waypoints_y] = decode_chromosome_pos(chromosome, problem);
    n_wp = problem.n_waypoints;
    waypoints = [waypoints_x, waypoints_y];

    % --- Input Validation ---
    if any(isnan(chromosome)) || any(isinf(chromosome)) || ~isreal(chromosome)
        % fprintf('Invalid chromosome detected.\n');
        fitness = inf; objective_component = inf; total_penalty = inf; path_len = inf;
        return;
    end

    % --- 1. Calculate Raw Path Length ---
    segment_vectors = diff(waypoints, 1, 1); % Vectors between consecutive points
    segment_lengths = sqrt(sum(segment_vectors.^2, 2)); % Euclidean distance for each segment
    path_len = sum(segment_lengths); % Total raw path length

    % --- 2. Calculate Weighted Objective Component ---
    % Only consider path length weighted by alpha here
    objective_component = problem.alpha * path_len;

    % --- 3. Calculate Penalties ---
    total_penalty = 0;

    % a) Boundary Penalties (apply only to intermediate waypoints)
    intermediate_waypoints = waypoints(2:n_wp-1, :);
    x_coords = intermediate_waypoints(:, 1);
    y_coords = intermediate_waypoints(:, 2);

    % Squared violation distance from boundaries
    penalty_x_min = sum(max(0, problem.x_min - x_coords).^2);
    penalty_x_max = sum(max(0, x_coords - problem.x_max).^2);
    penalty_y_min = sum(max(0, problem.y_min - y_coords).^2);
    penalty_y_max = sum(max(0, y_coords - problem.y_max).^2);
    total_penalty = total_penalty + penalty_weights.K_boundary * (penalty_x_min + penalty_x_max + penalty_y_min + penalty_y_max);

    % b) Obstacle Penalties
    if ~isempty(problem.obstacles)
        num_obstacles = size(problem.obstacles, 1);
        % Effective radius = obstacle radius + robot radius
        effective_radii = problem.obstacles(:, 3) + problem.robot_radius;
        min_obs_dist_sq_wp = effective_radii.^2; % Squared distance threshold for waypoints
        min_obs_dist_sq_seg = min_obs_dist_sq_wp; % Use same buffer for segments (simplification)

        % Waypoint-Obstacle Penalties (check ALL waypoints for safety)
        for i = 1:n_wp
            wp = waypoints(i, :); % Current waypoint coordinates
            for k = 1:num_obstacles
                obs_center = problem.obstacles(k, 1:2);
                dist_sq = sum((wp - obs_center).^2); % Squared distance to obstacle center
                % Squared violation depth (how much the minimum distance squared is violated)
                violation_sq = max(0, min_obs_dist_sq_wp(k) - dist_sq);
                % Penalty increases quadratically with violation depth
                total_penalty = total_penalty + penalty_weights.K_obstacle_waypoint * violation_sq; % Quadratic penalty
            end
        end

         % Segment-Obstacle Penalties (Simplified: Check midpoint distance)
         % More accurate check would involve point-segment distance calculation
         for i = 1:(n_wp - 1) % For each segment
             p1 = waypoints(i, :);
             p2 = waypoints(i + 1, :);
             mid_point = (p1 + p2) / 2; % Segment midpoint
             for k = 1:num_obstacles
                 obs_center = problem.obstacles(k, 1:2);
                 dist_sq = sum((mid_point - obs_center).^2); % Squared distance from midpoint to obstacle center
                 % Squared violation depth for segment midpoint
                 violation_sq = max(0, min_obs_dist_sq_seg(k) - dist_sq);
                 total_penalty = total_penalty + penalty_weights.K_obstacle_segment * violation_sq; % Quadratic penalty
             end
         end
    end

    % --- 4. Calculate Final Fitness ---
    % Fitness = weighted objective + total penalty
    fitness = objective_component + total_penalty;

    % --- Output Validation ---
    % Ensure all returned values are real and finite, default to Inf otherwise
    if isnan(fitness) || ~isreal(fitness); fitness = inf; end
    if isnan(objective_component) || ~isreal(objective_component); objective_component = inf; end
    if isnan(total_penalty) || ~isreal(total_penalty); total_penalty = inf; end
    if isnan(path_len) || ~isreal(path_len); path_len = inf; end

     % Re-ensure fitness is Inf if any component is Inf (important for selection)
     if isinf(objective_component) || isinf(total_penalty)
         fitness = inf;
     end

end % End of evaluate_solution_pos function


%% --- decode_chromosome_pos Function ---
% Converts the 1D chromosome vector into 2D waypoint coordinates
function [waypoints_x, waypoints_y] = decode_chromosome_pos(chromosome, problem)
    n_wp = problem.n_waypoints;
    n_vars = problem.n_vars; % Expected number of variables = 2 * (n_wp - 2)

    % --- Input Validation ---
    if numel(chromosome) ~= n_vars
        error('Decode Error: Chromosome length (%d) does not match expected n_vars (%d = 2*(%d-2)).', ...
              numel(chromosome), n_vars, n_wp);
    end
     if any(isnan(chromosome)) || any(isinf(chromosome)) || ~isreal(chromosome)
         error('Decode Error: Chromosome contains NaN, Inf, or non-real values.');
     end


    % Reshape the chromosome into (n_waypoints - 2) rows and 2 columns (x, y)
    intermediate_coords = reshape(chromosome(:), [n_wp - 2, 2]); % Ensure column vector then reshape

    % Construct the full list of waypoints including fixed start and goal
    waypoints_x = [problem.x_start; intermediate_coords(:,1); problem.x_goal];
    waypoints_y = [problem.y_start; intermediate_coords(:,2); problem.y_goal];

end % End of decode_chromosome_pos function


%% --- tournament_selection Function ---
% Selects an individual index using tournament selection
function selected_idx = tournament_selection(fitness_values, tournament_size)
    pop_size = length(fitness_values);
    if pop_size == 0; error('Tournament Selection Error: Population size is zero.'); end

    % Randomly select 'tournament_size' participants from the population
    candidate_indices = randi(pop_size, 1, tournament_size);

    best_idx_in_tournament = -1;          % Index of the winner in the original population
    best_fitness_in_tournament = inf;   % Fitness of the winner

    % Find the best individual among the participants
    for i = 1:tournament_size
        current_idx = candidate_indices(i);
        % Basic check for index validity (should not happen with randi if pop_size > 0)
        if current_idx < 1 || current_idx > pop_size
             warning('Tournament generated invalid index %d (Pop size %d)', current_idx, pop_size);
             continue; % Skip this participant
        end

        current_fitness = fitness_values(current_idx);

        % Update winner if current participant is better (lower fitness is better)
        if isfinite(current_fitness) && current_fitness < best_fitness_in_tournament
            best_fitness_in_tournament = current_fitness;
            best_idx_in_tournament = current_idx;
        elseif isinf(current_fitness) && isinf(best_fitness_in_tournament) && best_idx_in_tournament == -1
             % Handle case where the first few participants are Inf
             best_idx_in_tournament = current_idx;
        end
    end

    % --- Handle Outcome ---
    if best_idx_in_tournament ~= -1
        % A winner (potentially with Inf fitness if all were Inf) was found
        selected_idx = best_idx_in_tournament;
    else
        % This should ideally not happen if tournament_size > 0 and pop_size > 0
        % Fallback: select the first participant if no valid winner identified
        selected_idx = candidate_indices(1);
        warning('Tournament selection failed to find a best index. Defaulting to first participant %d.', selected_idx);
    end

    % --- Final Sanity Check ---
    if ~isscalar(selected_idx) || selected_idx < 1 || selected_idx > pop_size || floor(selected_idx) ~= selected_idx
        warning('Invalid index %.2f selected in tournament. Defaulting to 1.', selected_idx);
        selected_idx = 1; % Force valid index as last resort
    end

end % End of tournament_selection function


%% --- basic_local_search_pos Function ---
% Simple Hill Climbing local search to refine a solution (optional)
function improved_solution = basic_local_search_pos(solution, problem, penalty_weights)
    % --- Parameters for Local Search ---
    max_ls_iter = 10;       % Max iterations for local search refinement
    step_size = 0.1;        % Initial step size for perturbations
    step_reduction = 0.7;   % Factor to reduce step size if no improvement found
    min_step_size = 1e-5;   % Minimum step size before stopping LS

    current_solution = solution;
    % Evaluate the starting point
    [current_fitness, ~, ~, ~] = evaluate_solution_pos(current_solution, problem, penalty_weights);

    % Only proceed if the initial fitness is finite
    if isinf(current_fitness)
        improved_solution = solution; % Return original if starting fitness is Inf
        return;
    end

    % --- Iterative Refinement ---
    for iter = 1:max_ls_iter
        best_neighbor = current_solution;       % Best neighbor found in this iteration
        best_neighbor_fitness = current_fitness;% Fitness of the best neighbor
        improved_in_iteration = false;        % Flag if improvement was found

        % Iterate through each variable (coordinate) in the solution vector
        for j = 1:length(solution)
            % Try positive step perturbation
            neighbor_plus = current_solution;
            neighbor_plus(j) = neighbor_plus(j) + step_size;
            % Optional: Enforce bounds strictly? evaluate_solution_pos handles via penalty
            [fitness_plus, ~, ~, ~] = evaluate_solution_pos(neighbor_plus, problem, penalty_weights);

            if fitness_plus < best_neighbor_fitness
                best_neighbor_fitness = fitness_plus;
                best_neighbor = neighbor_plus;
                improved_in_iteration = true;
            end

            % Try negative step perturbation
            neighbor_minus = current_solution;
            neighbor_minus(j) = neighbor_minus(j) - step_size;
            % Optional: Enforce bounds strictly?
            [fitness_minus, ~, ~, ~] = evaluate_solution_pos(neighbor_minus, problem, penalty_weights);

            if fitness_minus < best_neighbor_fitness
                best_neighbor_fitness = fitness_minus;
                best_neighbor = neighbor_minus;
                improved_in_iteration = true;
            end
        end % End loop through variables

        % --- Update Based on Iteration Outcome ---
        if improved_in_iteration
            % Move to the best neighbor found
            current_solution = best_neighbor;
            current_fitness = best_neighbor_fitness;
            % Optional: Could reset step size or slightly increase it upon improvement
        else
            % No improvement found by perturbing any variable, reduce step size
            step_size = step_size * step_reduction;
            if step_size < min_step_size
                % Stop local search if step size becomes too small
                break;
            end
        end
    end % End local search iterations

    improved_solution = current_solution; % Return the best solution found by LS

end % End of basic_local_search_pos function

% ======================== END OF FILE ============================