function  plot_stats(stat, whites)
% Smooths and plots stat TFRs
%   Detailed explanation goes here
% stat.stat2 = -stat.stat;
stat.stat2 = conv2(squeeze(stat.stat), ones(5)/25,'same');
stat.stat2 = permute(stat.stat2,[3,1,2]);
stat.mask = stat.stat2>0 & ...
    permute(conv2(squeeze(stat.prob), ones(5)/25,'same'),[3,1,2])<=.05;
cfg = [];
cfg.parameter = 'stat2';
cfg.maskparameter  = 'mask';
cfg.maskstyle      ='outline';
cfg.renderer       = 'painters';
cfg.maskalpha      = .1;
cfg.masknans       = 'no';% or 'no' (default = 'yes')
cfg.zlim     = [0,3];
v = viridis2(8);
cfg.colormap = interp1(linspace(1, whites+3, whites+3),...
    [ones(whites,3);v([1,4,8],:)],linspace(1, whites+3, 128));
ft_singleplotTFR(cfg, stat)
end