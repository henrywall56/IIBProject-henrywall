function [coefficent z]=norderline(order,frequency,data);
% this is to find the nth order least squre lineof the input data
coefficent=polyfit(frequency,data,order);

linedata=zeros(1,length(data));

for i=1:(order+1)
    linedata=linedata+coefficent(i)*frequency.^(order+1-i);
end
% plot(t,data);
% hold on; grid on;
% plot(t,data,'*');
% plot(t,linedata,'r-');

% error = linedata-data;
% figure(3);
% plot(frequency,error);hold on;grid on;plot(frequency,error,'*');
% variation=0;
% for i=1:length(error)
%     variation=variation+error(i)^2;
% end
% 
% fprintf('the error variation is:');
% variation=sqrt(variation/length(error))
% fprintf('\n');
% 
% fprintf('the error percent is:');
% errorpercent=mean(abs(error./linedata))
% fprintf('\n');

z=linedata;