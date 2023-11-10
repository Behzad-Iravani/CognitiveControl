% -*- UTF-8 -*-
% Prepare data for DCNN!
% Author:
%       Neda Kaboodvand & Behzad Iravani
%       n.kaboodvand@gmail.com
%       behzadiravani@gmail.com
%
% This script prepares the source reconsted timeseire 
% of the primary dataset used in "Predicting Conflict Errors:
% Integrating Constitutional Neural Networks and Whole-Brain
% Modeling of Preparatory Electrophysiological Activity" for training CNN
%
% Copyright (C) Neda Kaboodvand & Behzad Iravani
% Department of Neurosurgery, University of Pennsylvania. Philadelphia 
% Department of Neurology and Neurological sciences, Stanford University, Palo Alto 
% 
% May 29, 2023---
% -------------------------------------------

disp('Prepare data for DCNN!')
% clear memory 
clear 
clc
%% Add filedtrip topplbox 
restoredefaultpath
addpath('F:\Matlab_ToolBoxes\fieldtrip-master');
ft_defaults()
%% Experimntal conditions 
CONFtypes = {'con','incon'}; % stimulus: congurent or incongurent 
ERRtypes  = {'hit','err'};    % response: Hit or error 
%%
data = struct('signalPreP',[],'signalAft',[], 'chanlabel', [], 'time', []); % initiazlie data structure
s    = dir("source\*_timeseries.mat"); % get the time_series files'name 
for is = 1:numel(s) % loop over the data fiels 
    % load timeseries 
    load(fullfile(s(is).folder, s(is).name))
    % extrect the participant ID
    S_ = regexp(s(is).name,'\w+-\w+(?=_eLORETA)','match');
    disp(S_{:})
    % ------------ Select incongurent trials -----------------
    clear tmp tmpPreP tmpAft
    cfg        = [];
    cfg.trials = (EEGSource.trialinfo.type == 'incon' .... )
        & EEGSource.trialinfo.stimRT~=0);
    cfg.latency = [-1.5, -.05];

    tmp        = rmfield(EEGSource, 'elec');
    [n,x]      = hist(categorical(tmp.label));
    for ch = x(n>1)
        for ii = 1:n(strcmp(x,ch{:}))
             newL{ii} = sprintf('%s_%d',ch{:},ii);
        end
        tmp.label(categorical(tmp.label) == ch) = newL;
    end

    tmpPreP     = ft_selectdata(cfg, tmp);
    cfg.latency = [-.05, 1.5];
    tmpAft      = ft_selectdata(cfg, tmp);
    % -------------------------------------------------------
    L = size(tmpPreP.trial, 1);  % get the number of trials 
    H = size(data.signalPreP,1); % get the size of the signal
    % storing the data for the particpant 
    data.signalPreP(H+1:H+L,:,:) = tmpPreP.trial;
    data.signalAft(H+1:H+L,:,:) = tmpAft.trial;
    data.label(H+1:H+L, 1)   = cellstr(tmpPreP.trialinfo.accur);
    data.sub(H+1:H+L,1)      = repmat(S_,L,1);
    data.RT(H+1:H+L,1)       = tmpPreP.trialinfo.stimRT;
    data.LogRT(H+1:H+L,1)    = log10(tmpPreP.trialinfo.stimRT);
end
data.chanlabel = tmp.label;
data.time = tmp.time;
disp('saving...')
save data_incong_hit_err.mat data
disp('$End')
% $END



