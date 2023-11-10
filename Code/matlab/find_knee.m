function knee_index = find_knee(y, window_size)
dy = diff(y);
smoothed_dy = smooth(dy, window_size);
n = numel(smoothed_dy);
deviations = zeros(1, n);

for i = 2:n-1
    % Calculate deviation from a line connecting the first and last points
    deviations(i) = abs((smoothed_dy(i+1) - smoothed_dy(i)) * i - (n - 1) * (smoothed_dy(n-1) - smoothed_dy(1)));
end

[~, knee_index] = max(deviations);
if  knee_index>0
    knee_index = knee_index -1;
end
end