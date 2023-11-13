#!/bin/bash

TRJ="trajectory.xyz"
COL="../all_cluster_IDs_xyz.dat"

# Ntot=$(wc -l < "$TRJ" | awk '{print $1}')	# number of lines in the file
N=$(head -n 1 "$TRJ" | tr -d ' ')			# number of particles in the system
Nl=$(expr "$N" + 2)							# lines for each frame in the file
cp -r $TRJ tmp1.xyz

# remove the last Nl lines
echo "Removing last lines..."
for i in {0..5}; do
	Ntot=$(wc -l < "tmp1.xyz" | awk '{print $1}')
	# echo $Ntot
	d=$(expr "$Ntot" - "$Nl")
	head -n $d tmp1.xyz	> tmp2.xyz
	mv tmp2.xyz tmp1.xyz
done

# remove the first Nl lines
echo "Removing first lines..."
for i in {0..5}; do
	sed "1,${Nl}d" < tmp1.xyz > tmp2.xyz
	mv tmp2.xyz tmp1.xyz
	# wc -l tmp1.xyz | awk '{print $1}'
done

echo "The following data length must be equal:"
wc -l tmp1.xyz $COL

paste tmp1.xyz $COL > tmp2.xyz			# create a single file with all the information

# now create a file with the correct formatting
awk -v N=$Nl 'NR%N==1 {print N-2} NR%N==1 {print "Properties=species:S:1:pos:R:3"} NF==4 {print $4, $1,$2,$3}' tmp2.xyz > colored_trj.xyz

rm -f tmp*.xyz