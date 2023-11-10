function lfilter = phase_linearlization(filter)
phase = linspace(pi,-pi, length(filter));

lfilter = real(ifft(abs(fft(filter)).*exp(1j*phase')));

end