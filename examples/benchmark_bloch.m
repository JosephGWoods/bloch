function benchmark_bloch(varargin)
%BENCHMARK_BLOCH Benchmark bloch() across OpenMP thread counts.
%   benchmark_bloch
%   benchmark_bloch('threads', [1 4 8])
%   benchmark_bloch('repeats', 5, 'threads', [1 4 8])
%
% If no thread counts are supplied, queries the current OpenMP thread count
% via bloch('threads') and benchmarks 1, current/2, and current threads.
% The original thread count is restored at the end.

    parser = inputParser;
    parser.FunctionName = 'benchmark_bloch';
    parser.addParameter('repeats', 3, @(x) isnumeric(x) && isscalar(x) && x > 0 && x == floor(x));
    parser.addParameter('threads', [], @(x) isempty(x) || (isnumeric(x) && all(x > 0) && all(floor(x) == x)));
    parser.parse(varargin{:});

    repeats = parser.Results.repeats;
    requestedThreads = parser.Results.threads;

    originalThreads = max(1, round(bloch('threads')));

    if isempty(requestedThreads)
        requestedThreads = unique([1, max(1, round(originalThreads / 2)), originalThreads]);
    end

    [b1, gz, dt, dp, df, dv] = build_problem();
    fprintf('Problem size: dp=%d, df=%d, dv=%d, total=%d\n', ...
        numel(dp), numel(df), numel(dv), numel(dp)*numel(df)*numel(dv));
    fprintf('Runs per thread count: %d\n', repeats);

    cleanupObj = onCleanup(@() restore_threads(originalThreads)); % restores threads even on error

    for ii = 1:numel(requestedThreads)
        threads = requestedThreads(ii);
        bloch('threads', threads);

        times = zeros(repeats, 1);
        for jj = 1:repeats
            t0 = tic;
            [~, ~, ~] = bloch(b1, gz, dt, [], [], df, dp, dv, 0);
            times(jj) = toc(t0);
        end

        fprintf('=== OpenMP threads = %d ===\n', threads);
        fprintf('Benchmark run time results: mean=%.3f s, best=%.3f s, worst=%.3f s\n\n', ...
            mean(times), min(times), max(times));
    end
end

function [b1, gz, dt, dp, df, dv] = build_problem()
    % Slice-selective windowed sinc pulse (same in benchmark_bloch.m and benchmark_bloch.py).
    dur    = 2e-3;                                          % pulse duration (s)
    dt     = 10e-6;                                         % 10 us per time step
    ntime  = round(dur / dt);                               % number of time steps
    flip   = pi / 2;                                        % 90 degree flip angle
    slthk  = 5e-3;                                           % slice thickness (m)
    TBWP   = 8;                                              % time-bandwidth product
    t      = ((1:ntime).' - 0.5) * dt - (dur / 2);          % time points centered on zero
    alpha  = 0.5;                                            % apodization factor (Hann window)
    window = (1 - alpha) + alpha * cos(2 * pi * t / dur);   % Hann window
    b1     = window .* sinc(t * TBWP / dur);                % normalized sinc pulse
    b1     = b1 * flip / (2 * pi * sum(b1) * dt);           % B1+ waveform (Hz)
    b1     = [b1; zeros(floor(ntime/2), 1)];                % pad RF with zeros during gradient rewinder
    gz     = [ones(ntime, 1)          *  TBWP / dur / slthk;    % slice-selective gradient waveform (Hz/m)
              ones(floor(ntime/2), 1) * -TBWP / dur / slthk];   % slice rewinding gradient waveform (Hz/m)

    dp = linspace(-slthk, slthk, 100); % (m)
    df = linspace(-100, 100, 100);     % (Hz)
    dv = linspace(-0.5, 0.5, 100);     % (m/s)
end

function restore_threads(originalThreads)
    fprintf('Restoring original thread count (threads = %d)\n', originalThreads);
    bloch('threads', originalThreads);
end

