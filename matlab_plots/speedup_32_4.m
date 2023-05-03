
%[size,time] = readvars("1024x64_runs.txt")
[size(:,1),time(:,1)] = readvars("times_32_4.txt")
[size(:,2),time(:,2)] = readvars("non_pipelined_32_4.txt")
[size2,time2] = readvars("serial_avg.txt")

time2 = time2*1000;
speedup = time2./time

%newStr = extractAfter(size,5)
%newStr = extractAfter(size,8)
%size = categorical(newStr)
size2 = categorical(size2)
size2 = reordercats(size2,string(size2));
%newStr = substr(size,4,length)


bar(size2, speedup);
%title('1024x64 Average execution time') %1024x64 4096x64 1024x128 2048x128
%4096x128
title('Speedup, serial vs non-pipelined (block,grid) = (32,4)')
xlabel('Matrix size')
ylabel('Speedup factor')