function secs = clock2secs(c)
%%
% Function converts a vector created by the 'clock' function to seconds
% Note: only works up to days.

%% Do seconds
multiplier = 1 * 60 * 60 * 24;

secs = 0;

%extract the days
secs = secs + c(3) * multiplier;
multiplier = multiplier / 24;

%extract the hours
secs = secs + c(4) * multiplier;
multiplier = multiplier / 60;

%extract the minutes
secs = secs + c(5) * multiplier;

%extract the seconds
secs = secs + c(6);

end