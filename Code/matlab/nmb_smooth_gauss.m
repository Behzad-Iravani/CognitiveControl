function [sdat]= nmb_smooth_gauss(dat,sigma)
% This function is part of Novel measurment of olfcatory bulb
% Author: Behzad Iravani
% -----------------------------------------------------------
% input:
% dat   ---> fieldtrip data structure 
% sigma ---> standard deviation of gauss function
% output:
% sdta  ---> soomthed fieldtrip data structure 
% -----------------------------------------------------------

xc = 0;
yc = 0;
[x,y] = meshgrid([-3*sigma:.5:3*sigma]);

exponent = ((x-xc).^2 + (y-yc).^2)./(2*sigma^2);
amplitude = 1 / (sigma * sqrt(2*pi));  
% The above is very much different than Alan's "1./2*pi*sigma^2"
% which is the same as pi*Sigma^2 / 2.
W     = amplitude  * exp(-exponent);

sdat=dat;
try
    s=size(dat.powspctrm);
    if numel(s)==4
        for sub=1:size(dat.powspctrm,1)
            
            for ch=1:numel(dat.label)
                
                
                sdat.powspctrm(sub,ch,:,:)=conv2(squeeze(dat.powspctrm(sub,ch,:,:)),(1/sum(W(:)))*W,'same');
                
                
            end
        end
    else
        for ch=1:numel(dat.label)
            
            sdat.powspctrm(ch,:,:)=conv2(squeeze(dat.powspctrm(ch,:,:)),(1/sum(W(:)))*W,'same');
        end
    end

catch
    s=size(dat);
    W_new =(1/max(W(:)))*W;
    if numel(s) == 2
    sdat=conv2(squeeze(dat),(1/sum(W_new(:)))*W_new,'same');
    elseif numel(s)==3
        
        for i3 = 1:size(dat,1) 
        sdat(i3,:,:)=conv2(squeeze(dat(i3,:,:)),(1/sum(W_new(:)))*W_new,'same');
        end
    end

end



end

