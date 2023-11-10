clc
clear
load('data\source\python_simulation\C\simulated_82.mat')
dat = [];
dat.label  = arrayfun(@(j) sprintf('%d', j), 1:19, 'uni', false)';
dat.trial  = {squeeze(dat_downsample)'};
dat.time   = {linspace(-1.5,0, size(dat_downsample,2))};
dat.dimord = 'chan_time';
cfg =[];
cfg.method = 'mtmfft';
cfg.foi = 2:2:200;
cfg.tapsmofrq  = 4;
Notstim = ft_freqanalysis(cfg, dat)
% d = dir('data\source\python_simulation\S*')
dii = 0;
for di =  linspace(.5,5, 20)
    dii = dii +1;
    load(['data\source\python_simulation\Stimulation_ROI 12-6\_Stimulation_search_freq127052.6_' sprintf('%1.1f', di) '.mat'])
    dat = [];
    dat.label  = arrayfun(@(j) sprintf('%d', j), 1:19, 'uni', false)';
    dat.trial  = {squeeze(dat_downsample)'};
    dat.time   = {linspace(-1.5,0, size(dat_downsample,2))};
    dat.dimord = 'chan_time';
    cfg =[];
    cfg.method = 'mtmfft';
    cfg.foi = 2:2:200;
    cfg.tapsmofrq  = 4;
    stim{dii} = ft_freqanalysis(cfg, dat);
    cfg = [];
    cfg.parameter = 'powspctrm';
    cfg.operation = 'log10(x1/x2)';
    stim{dii} = ft_math(cfg, stim{dii}, Notstim);

    HIT(dii) = output(2);
end
%%
cfg = [];
cfg.keepindividual = 'yes';
gs = ft_freqgrandaverage(cfg,  stim{:});
%%
cfg = [];
cfg.method            = 'montecarlo';           % use the Monte Carlo Method to calculate the significance probability
cfg.statistic         = 'ft_statfun_correlationT';        % use the independent samples T-statistic as a measure to evaluate the effect at the sample level
cfg.correctm          = 'cluster';
cfg.clusteralpha      = 0.5;                   % alpha level of the sample-specific test statistic that will be used for thresholding
cfg.clustertail       = 0;
cfg.clusterstatistic  = 'maxsum';               % test statistic that will be evaluated under the permutation distribution.
cfg.tail              = 0;                      % -1, 1 or 0 (default = 0); one-sided or two-sided test
cfg.correcttail       = 'prob';                 % the two-sided test implies that we do non-parametric two tests
cfg.alpha             = 0.05;                   % alpha level of the permutation test
cfg.numrandomization  = 1000;                   % number of draws from the permutation distribution
cfg.design            = double(HIT);%2* ones(1,19); % design matrix, note the transpose
% cfg.design(1,12:13)   = 1;
cfg.ivar              = 1;                      % the index of the independent variable in the design matrix;
cfg.neighbours        = [];
stat = ft_freqstatistics(cfg, gs);
%%
label = readtable('data\DS01_fMRI_Loalized_ROIs.tsv', 'FileType','text');
load OrderMAE.mat
close all
sig = smooth(mean(stat.stat), 30, 'sgolay');

p = patch([stat.freq, fliplr(stat.freq)], [sig'-mean(stat.mask), fliplr(sig'+mean(stat.mask))],'k');
p.FaceAlpha = .7;
ax= gca();
ax.FontName = 'Arial Nova Cond';
ax.FontSize = 12;
ax.LineWidth =1.2;
ax.XAxisLocation = 'origin';
ylabel('T-value', 'FontSize', 12)
xlabel('Frequency', 'FontSize', 12)
% % for i = ix
% %     subplot(4,5,i)
% cfg=[];
% cfg.parameter     = 'stat';
% % cfg.maskparameter = 'mask';
% % cfg.maskstyle     = 'box';% 'thickness' or 'saturation' (default = 'box')
% % cfg.maskfacealpha = .6;
% 
% % cfg.channel  = i;
% cfg.title = label.RegionLabel(i);
% figure
% cfg = ft_singleplotER(cfg,stat)
% end

