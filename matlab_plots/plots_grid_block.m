
%[size,time] = readvars("non_pipelined_32_4.txt")
[size,time] = readvars("top_times.txt")

%time = time*1000;

%newStr = extractAfter(size,5)
%newStr = extractAfter(size,8)
%size = categorical(newStr)
size = categorical(size)
size = reordercats(size,string(size));
%newStr = substr(size,4,length)

bar(size, time);
%title('1024x64 Average execution time') %1024x64 4096x64 1024x128 2048x128
%4096x128
title('Best average execution times')
xlabel('size (block,grid)')
ylabel('Average execution time [milliseconds]')