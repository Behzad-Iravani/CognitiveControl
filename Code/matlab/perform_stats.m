function [stat] = perform_stats(TFR1,TFR2)
%  Perfomes statisitcal TFR analysis
%   Detailed explanation goes here

cfg = [];
cfg.method          = 'motecarlo';%'analytic';
cfg.statistic       = 'ft_statfun_indepsamplesT';
cfg.alpha           = 0.05;
cfg.correctm        = 'cluster';
n1 = size(TFR1.powspctrm,1);
n2 = size(TFR2.powspctrm,1);
design  = zeros(2,n1+n2);
design(1,1:n1) = 1;
design(1,n1+1:n1+n2) = 2;
design(2,1:n1) = [1:n1];
design(2,n1+1:n1+n2) = [1:n2];

cfg.design   = design;
cfg.ivar     = 1;
% % % cfg.uvar     = 2;
stat = ft_freqstatistics(cfg, TFR1, TFR2);
end