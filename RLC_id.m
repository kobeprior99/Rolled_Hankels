%% ========================================================================
%  Project:      RLC Circuit System Identification Using Hankel Matrices
%
%  Author:       Kobe Prior
%  Date:         November 13, 2026
%
%  Description:
%      This project investigates data-driven system identification of a
%      second-order RLC circuit using Hankel matrix methods. A dynamical
%      system model is reconstructed from sampled input-output measurements
%      using Hankel matrix decomposition and realization techniques.
%
%  Objectives:
%      1. Generate or obtain the input-output response of a second-order
%         RLC circuit.
%      2. Construct Hankel matrices from the measured system response.
%      3. Perform system identification to recover a reduced-order
%         state-space representation (A, B, C, D).
%      4. Compare the identified model against the known RLC system model.
%      5. Evaluate identification accuracy through time-domain response
%         comparisons and reconstruction error metrics.
%
%  Methodology:
%      Hankel Matrix Construction
%      Singular Value Decomposition (SVD)
%      Model Order Selection
%      State-Space Realization
%      Time-Domain Response Validation
%
%  ========================================================================

clc;
close all;

% threshold for truncating eigenvalues of SVD for system order
threshold = 1e-9;
% timestep
dt = 1e-5;

% change these values to change expected system
R = 10;
L = .1;
C = 10e-6;

% Expected SS parameter
Ac = [0 1; -1/(L*C) -R/L];
Bc = [0; 1/(L*C)];
Cc = [1 0];
Dc = 0;

% construct system and impulse response
sys_original = ss(Ac, Bc, Cc, Dc);
sys_original_discrete = c2d(sys_original, dt);
[h,t] = impulse(sys_original_discrete);
% magnitude scaling of impulse response
h = h * dt;
N = length(h);

[A_est, B_est, C_est, D_est, order] = estimateSS(h, N, threshold);


% pass in a unit step for testing
% I basically just copied Kobe's testing code from the powerpoint
u = ones(size(t));

sys_est = ss(A_est, B_est, C_est, D_est, dt);
y_original = lsim(sys_original, u, t);
y_era = lsim(sys_est, u, t);

figure;
plot(t, y_original, 'LineWidth', 1.5);
hold on;
plot(t, y_era, '--', 'LineWidth',1.5);
xlabel("Time (s)");
ylabel('Capacitor Voltage (V)');
legend('Original RCL', 'ERA estimate');

relative_error = norm(y_original-y_era) / norm(y_original);
mse = mean((y_original-y_era).^2);
fprintf('Relative Error %.6e\n', relative_error);
fprintf('MSE: %.6e\n', mse);



function [A_d, B_d, C_d, D_d, order] = estimateSS(h, N, threshold)
    % create hankel matrix
    [H0, H1] = hankelfy(h, N);
    % solve SVD for system order
    [~, S1, ~] = svd(H0);

    % calculate order by looking at significant eigenvalues
    order = sum(S1 > threshold, "all");
    % compute truncated SVD
    [U2, S2, V2] = svds(H0, order);
    fprintf("Estimated order: %d\n", order);

    sqrt_S = sqrt(S2);
    % calculate C and O parameters
    % math is taken from Kobe's powerpoint
    C = sqrt_S * V2';
    O = U2 * sqrt_S;

    A_d = (sqrt_S \ (U2' * H1 * V2)) / sqrt_S;
    B_d = C(:,1);
    C_d = O(1,:);
    D_d = h(1);

end

% create hankel matrixs for SID
function [H0, H1] = hankelfy(h, N)
    n = floor((N-1)/2);
    H0 = zeros(n, n);
    H1 = zeros(n, n);
    for i = 1:n
        for j = 1:n
        H0(i,j) = h(j+i);
        H1(i,j) = h(j+i+1);
        end
    end
end