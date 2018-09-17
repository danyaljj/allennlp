#!/usr/bin/perl
binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");
# compute mean of lines in a file
$numlines = 0;
$sum = 0.0;
while(<STDIN>)
{    
    $numlines += 1;
    $line = $_;
    chomp($line);
    $sum += $line;
}
print "total = ".$numlines."\n";
print "sum = ".$sum."\n";
$mean = $sum / $numlines;
print "mean = ".$mean."\n";

