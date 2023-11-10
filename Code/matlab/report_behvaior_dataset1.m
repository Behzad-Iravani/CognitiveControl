% -*- code: 'UTF-8' -*-

function BHV = report_behvaior_dataset1()
% report_behvaior_dataset1 extract the number of Hit vs Error and reaction
% time per individual.

load data\source\data_incong_hit_err.mat
BHV = table();
is = 0;
for s = unique(data.sub)' % loop over subjects
    is = is +1;
    BHV.subj{is} = s{:};
    clear x n
    [n,x] = hist(categorical(data.label(strcmp(data.sub, s{:}))));
    for ix = 1:length(x)
        BHV.(x{ix})(is) = n(ix);
    end
    BHV.rtErr(is)      = nanmean(data.RT(strcmp(data.sub, s{:}) & strcmp(data.label, 'err')));
    BHV.LowrtErr(is)   = min(data.RT(strcmp(data.sub, s{:}) & strcmp(data.label, 'err')));
    BHV.UpperrtErr(is) = max(data.RT(strcmp(data.sub, s{:}) & strcmp(data.label, 'err')));

    BHV.rtHit(is)      = nanmean(data.RT(strcmp(data.sub, s{:}) & strcmp(data.label, 'hit')));
    BHV.LowrtHit(is)   = min(data.RT(strcmp(data.sub, s{:}) & strcmp(data.label, 'hit')));
    BHV.UpperrtHit(is) = max(data.RT(strcmp(data.sub, s{:}) & strcmp(data.label, 'hit')));
end % end loop
fprintf('total number of individuals: %d \n', height(BHV));

fprintf('number of Hit : %1.2f +/- %1.2f \n', mean(BHV.hit), std(BHV.hit));
fprintf('mean rt of Hit : %1.2f +/- %1.2f \n', mean(BHV.rtHit), std(BHV.rtHit));


fprintf('number of Error : %1.2f +/- %1.2f \n', mean(BHV.err), std(BHV.err));
fprintf('mean rt of Error : %1.2f +/- %1.2f \n', mean(BHV.rtErr), std(BHV.rtErr));


end