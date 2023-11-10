function [x1, x2] = compute_inference(comp)
load data\interp\matlab\Grad-CAM_stats.mat


x1 = nan(size(H.hit.powspctrm,1),1);
x2 = nan(size(H.error.powspctrm,1),1);

for i=1:length(x1)
  % construct a 3-dimensional Boolean array to select the data from this participant
  sel3d        = false(size(squeeze(H.hit.powspctrm(i,:,:,:))));
  sel3d = squeeze(stat.mask);

  % select the FIC data in the cluster for this participant, represent it as a vector
  tmp = squeeze(H.hit.powspctrm(i,:,:,:));
  tmp = tmp(comp);
  % compute the average over the cluster
  x1(i) = mean(tmp);
end

for i=1:length(x2)
  % construct a 3-dimensional Boolean array to select the data from this participant
  sel3d        = false(size(squeeze(H.error.powspctrm(i,:,:,:))));
  sel3d = squeeze(stat.mask);

  % select the FIC data in the cluster for this participant, represent it as a vector
  tmp = squeeze(H.error.powspctrm(i,:,:,:));
  tmp = tmp(comp);
  % compute the average over the cluster
  x2(i) = mean(tmp);
end



end