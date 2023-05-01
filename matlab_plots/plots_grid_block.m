
%[size,time] = readvars("1024x64_runs.txt")
[size,time] = readvars("4096x128_runs.txt")

%time = time*1000;

%newStr = extractAfter(size,5)
newStr = extractAfter(size,8)
size = categorical(newStr)
size = reordercats(size,string(size));
%newStr = substr(size,4,length)

bar(size, time);
%title('1024x64 Average execution time') %1024x64 4096x64 1024x128 2048x128
%4096x128
title('4096x128 Average execution time')
xlabel('size (block,grid)')
ylabel('Average execution time [milliseconds]')