% author : ursy

close all
clear all

set(0, 'defaulttextinterpreter', 'latex')
set(0, 'defaultAxesFontname', 'Times New Roman')
set(0, 'defaultAxesFontSize', 22)
set(groot, 'defaultAxesTicklabelInterpreter', 'latex')
set(groot, 'defaultLegendInterpreter', 'latex')

% load output file
load ./output/run.output_positions.csv
load ./output/run.output_times.csv

% set some entries
% fiber bead radius
a_bead_fib = 1e-03;
% number of fiber beads
Nb = 20;
% length
LS = Nb * 2.2 * a_bead_fib;
% viscosity
eta = 1.0;
% weight per unit length
W = 1.0;
% number of fibers
n_fibers = 1;

% settling time
unit_time_gravity = LS * eta / W;

% plot setup
save_movie = true;
plot_steps = 30;                        % plot every n steps.
plot_now = plot_steps - 1;
figure('Renderer', 'painters', 'Position', [10 10 600 600])
Lim = 0.5;                              % axis limits.

% movie setup
if save_movie
    movie = VideoWriter(['output/' 'Movie'  '.avi']); 
    movie.FrameRate = 5;  % How many frames per second.
    open(movie);
    framecount = 1;
end

size_positions = size(run_output_positions);
% total number of beads
number_of_beads = fix(size_positions(2) / 3);
number_of_beads_per_fiber = fix(number_of_beads / n_fibers);
% set a radius array
rad_array = (a_bead_fib/LS) * ones(1, number_of_beads);
% total number of steps
n_steps = size_positions(1);

for step = 1:n_steps-1  % loop over steps
    plot_now = plot_now + 1;
    
    position_i = run_output_positions(step, :);    
    all_x = position_i(1:3:size_positions(2));
    all_y = position_i(2:3:size_positions(2));
    all_z = position_i(3:3:size_positions(2));
    
    x_m = mean(all_x);
    y_m = mean(all_y);
    z_m = mean(all_z);
    
    centers = [((all_x - x_m)/LS)' ((all_z - z_m)/LS)'];
    
    all_x = reshape(all_x, [number_of_beads_per_fiber, n_fibers]);
    all_y = reshape(all_y, [number_of_beads_per_fiber, n_fibers]);
    all_z = reshape(all_z, [number_of_beads_per_fiber, n_fibers]);
    
    if plot_now == plot_steps
        for k = 1:n_fibers
            plot((all_x(:,k) - x_m)/LS, (all_z(:,k) - z_m)/LS, '-', 'Color', uint8([17 17 17]));
            hold on
        end
        viscircles(centers, rad_array, 'Color', uint8([17 17 17]));
        
        title(['$t/T = \;$'  num2str(fix(run_output_times(step) / unit_time_gravity))]);
        
        pbaspect([1 1 1])
        xlabel('$(x - x_{m})/L$');
        ylabel('$(z - z_{m})/L$');
        xticks([-Lim 0 Lim]);
        yticks([-Lim 0 Lim]);
        xlim([-Lim-0.01 Lim+0.01]);
        ylim([-Lim-0.01 Lim+0.01]);
        axis equal
        hold off
        
        if save_movie == true
            frame = getframe(gcf);
            writeVideo(movie,frame);
            framecount=framecount+1;
        end
        pause(0.1);
    end
    if plot_now == plot_steps
        plot_now = 0;
    end
end

if save_movie
    close(movie);
end

