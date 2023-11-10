% -*- UTF-8 -*-
% Author:
%       Neda Kaboodvand & Behzad Iravani
%       n.kaboodvand@gmail.com
%       behzadiravani@gmail.com
%
% This script carries out the preptrocessing of EEG data for the primary dataset used in "Predicting Conflict Errors:
% Integrating Constitutional Neural Networks and Whole-Brain
% Modeling of Preparatory Electrophysiological Activity"
%
% Copyright (C) Neda Kaboodvand & Behzad Iravani
% Department of Neurosurgery, University of Pennsylvania. Philadelphia 
% Department of Neurology and Neurological sciences, Stanford University, Palo Alto 
% 
% May 29, 2023---
% -------------------------------------------
%% Clear Memory 
clear
clc
%% Add fieldTrip toolbox
restoredefaultpath
addpath('C:\MatlabToolboxes\fieldtrip-master\fieldtrip-master')
ft_defaults()
%% Paths
EEG_path = 'data_eeg\'; % path to eeg
fMRI_path = 'data_fMRI\Preprocessed\TCs'; % path to fMRI TCs
%% load behavioral tabel
load beh_sef.mat % load behavioral mat file 
number_subj = unique(T.Subject); % get the unique number of participants 
%% Preprocessing
for subj = 1:max(number_subj) % loop over participants 
     if exist(['Analyzed\EEG\ICA\' sprintf('sub-%02d_data_orig.mat', subj)],'file') && ... 
        exist(['Analyzed\EEG\ICA\' sprintf('sub-%02d_ICA_comp.mat', subj)],'file') && ...
        exist(['Analyzed\EEG\ICA\' sprintf('sub-%02d_data_clean.mat', subj)],'file') % checks if already processed
         fprintf('The analysis for sub-%02d is compeleted, moving on...\n ', subj)
         continue % the preprocessing has already been taken care of, move to the next subject  
     end % if
    
    if ~exist(['Analyzed\EEG\ICA\' sprintf('sub-%02d_data_orig.mat', subj)],'file') % is propressing done

        cfg                     = [];
        cfg.dataset             = [EEG_path, sprintf('Expsef_%d_1.vhdr', subj)];
        cfg.trialdef.eventtype  = {'Stimulus'};
        cfg.trialdef.eventvalue = {'S 13'};
        cfg.trialdef.prestim    = 2;
        cfg.trialdef.poststim   = 2;
        [cfg]                   = ft_definetrial(cfg);

        data = ft_preprocessing(cfg);

        cfg               = [];
        cfg.lpfilter      = 'yes';
        cfg.dftfilter     = 'yes';
        cfg.lpfreq        = [200];
        cfg.dftfreq       = [50, 100, 150];
        cfg.reref         =  'yes';
        cfg.refchannel    = 'all';

        data              = ft_preprocessing(cfg, data);
        save(['Analyzed\EEG\ICA\' sprintf('sub-%02d_data_orig.mat', subj)], 'data')
    else
        fprintf('The orignal data exists, loading ...')
        load(['Analyzed\EEG\ICA\' sprintf('sub-%02d_data_orig.mat', subj)], 'data')
    end
    if ~any(T.Subject == subj)
        continue
    end
    data.trialinfo = T(T.Subject == subj,:);


    % ICA
    if  ~exist(['Analyzed\EEG\ICA\' sprintf('sub-%02d_ICA_comp.mat', subj) ],'file')
        % resample
        cfg =[];
        cfg.resamplefs      = 128;
        rsDat = ft_resampledata(cfg, data);

        cfg        = [];
        cfg.method = 'fastica';

        comp = ft_componentanalysis(cfg, rsDat);
        save(['Analyzed\EEG\ICA\' sprintf('sub-%02d_ICA_comp.mat', subj)], 'comp')
    else
        fprintf('The ICA component exists, loading ...')
        load(['Analyzed\EEG\ICA\' sprintf('sub-%02d_ICA_comp.mat', subj)], 'comp')
    end

    if ~exist(['Analyzed\EEG\ICA\' sprintf('sub-%02d_data_clean.mat', subj)],'file')
        f = figure;
        cfg = [];
        cfg.component = 1:20;       % specify the component(s) that should be plotted
        cfg.layout    = 'biosemi64'; % specify the layout file that should be used for plotting
        cfg.comment   = 'no';
      

        cfg.viewmode                = 'component';
        ft_databrowser(cfg, comp)
        drawnow()
        rmComp = input('Please enter ICA components to remove? ');


        cfg = [];
        cfg.component  = rmComp;
        cfg.demean     = 'yes';
        [data] = ft_rejectcomponent(cfg, comp, data);


        save(['Analyzed\EEG\ICA\' sprintf('sub-%02d_data_clean.mat', subj)], 'data')
        close(f)
    end
end
disp('$End')
% $END

