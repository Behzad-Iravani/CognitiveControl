% -*- UTF-8 -*-
% Author:
%       Neda Kaboodvand & Behzad Iravani
%       n.kaboodvand@gmail.com
%       behzadiravani@gmail.com
%
% This script carries out the source reconstruction of the fMRI localized
% sources for the primary dataset used in "Predicting Conflict Errors:
% Integrating Constitutional Neural Networks and Whole-Brain
% Modeling of Preparatory Electrophysiological Activity"
%
% Copyright (C) Neda Kaboodvand & Behzad Iravani
% Department of Neurosurgery, University of Pennsylvania. Philadelphia 
% Department of Neurology and Neurological sciences, Stanford University, Palo Alto 
% 
% May 27, 2023---
% -------------------------------------------

% clear memory 
clc
clear 
disp('Source reconstruction using eLORTE...')
%% ------------
restoredefaultpath; % restore the path 
addpath('F:\Matlab_ToolBoxes\fieldtrip-master'); % add fieldtrip to the matlab path 
ft_defaults()
%% load souce location defined by fMRI
Nodes = readtable('DS1_fMRI_Loalized_ROIs.tsv', 'FileType', 'text');% load the MNI coordinates 
% create a template source model
cfg                   = [];
cfg.sourcemodel.pos   = [Nodes.x, Nodes.y, Nodes.z];
template_source_model = ft_prepare_sourcemodel(cfg);
%% load eeg data
F = dir("Analyzed\EEG\ICA\*_clean.mat");
for fi = 1:numel(F) % loop over the clean EEG --- ICA was used to remove the scanner and other noisy components  
    
    S = regexp(F(fi).name,'\w+-\w+(?=_data)','match'); % find the participants name 
    % load the manually realigned electrodes 
    fprintf('load %s mri\n', S{:})
    elec_source = load(['sensor\', S{:} '_realign_elec.mat'], 'elec_new', 'vol', 'bnd');
    % load the participants' MRI 
    mri = ft_read_mri(fullfile('data_fMRI', S{:}, 'anat', ['c',S{:},'_T1w.nii']));
    %% Build the source model
    cfg           = [];
    cfg.warpmni   = 'yes';
    cfg.template  = template_source_model;
    cfg.nonlinear = 'yes';
    cfg.mri       = mri;
    cfg.unit      ='cm';
    source_model          = ft_prepare_sourcemodel(cfg);
    % load the EEG data 
    load(fullfile(F(fi).folder, F(fi).name))
    % redefine trials
    cfg            = [];
    cfg.latency    = [-1.500, 1.500];
    cfg.channel    = {'all', '-ECG'};

    prepratory     = ft_selectdata(cfg, data);
    % ----- computed the covariance for source reconstruction  --------% 
    cfg            = [];
    cfg.covariance ='yes';
    prepratoryA     = ft_timelockanalysis(cfg, prepratory);
    % save the results to the sensor folder  
    save(['sensor\', S{:} '_trial_timelock.mat'], 'prepratoryA', 'preparatory')
    % estimate the common filter for source reconstruction 
    cfg                    = [];
    cfg.method             = 'eloreta';
    cfg.sourcemodel        = source_model;
    cfg.headmodel          = elec_source.vol;
    cfg.elec               = elec_source.elec_new;
    cfg.keeptrials         = 'no';
    cfg.eloreta.keepfilter = 'yes';
    cfg.eloreta.lambda     = .1;

    eLORETA= ft_sourceanalysis(cfg, prepratoryA);

    save(['source\', S{:} '_eLORETA_filter.mat'], 'eLORETA', 'source_model')
    % Extract trials maximum projection
        clear timeseries  timeseriesmaxproj
        for itrial=1:length(prepratory.trial)
            timeseries(:,itrial) = cellfun(@(x) x*prepratory.trial{itrial},eLORETA.avg.filter(eLORETA.inside),'Uni',0);
        end

        for ich=1:size(timeseries,1) %voxels
            [u, s, v] = svd(cat(2, timeseries{ich,:}), 'econ');
            timeseriesmaxproj(ich,:) = u(:,1)' * cat(2, timeseries{ich,:});
        end

        % save source data
        EEGSource           = [];
        EEGSource.label     = Nodes.label;
        EEGSource.elec.pos  = [Nodes.x, Nodes.y, Nodes.z];
        EEGSource.time      = prepratory.time{1};
        EEGSource.trial     = permute(reshape(timeseriesmaxproj, size(timeseries,1), length(prepratory.time{1}), length(prepratory.trial)),[3,1,2]);
        EEGSource.trialinfo = prepratory.trialinfo;

        save(['source\', S{:} '_eLORETA_maxPro_timeseries.mat'], 'EEGSource')
end
disp('$END')
% $END
