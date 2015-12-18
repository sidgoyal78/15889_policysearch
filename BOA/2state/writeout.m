function writeout(prefix, problem, arr)
    f1 = strcat(prefix, '_' , problem, '_normal.txt'  );
    f2 = strcat(prefix, '_', problem, '_max.txt');
    csvwrite(f1, arr);
    mvala = givemaxtillnow(arr);
    csvwrite(f2, mvala);
end