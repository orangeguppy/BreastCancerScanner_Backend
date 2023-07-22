#!/bin/bash
#author = Fabio Alexandre Spanhol
#email = faspanhol@gmail.com
#revision = 07-24-2014

# I modified the script by the original author to count the number of samples for each tumour type and magnification

# pass in the directory to search on the command line, use $PWD if not arg received
rdir=${1:-$(pwd)}

# if $rdir is a file, get it's directory
if [ -f $rdir ]; then
    rdir=$(dirname $rdir)
fi

# declare an array to store the sample count for each kind of tumour and magnification
declare -A count_for_each_class=( 
    ["adenosis"]=0 ["fibroadenoma"]=0 ["phyllodes_tumor"]=0 ["tubular_adenoma"]=0 
    ["ductal_carcinoma"]=0 ["lobular_carcinoma"]=0 ["mucinous_carcinoma"]=0 ["papillary_carcinoma"]=0

    ["adenosis40X"]=0 ["adenosis100X"]=0 ["adenosis200X"]=0 ["adenosis400X"]=0
    ["fibroadenoma40X"]=0 ["fibroadenoma100X"]=0 ["fibroadenoma200X"]=0 ["fibroadenoma400X"]=0
    ["phyllodes_tumor40X"]=0 ["phyllodes_tumor100X"]=0 ["phyllodes_tumor200X"]=0 ["phyllodes_tumor400X"]=0
    ["tubular_adenoma40X"]=0 ["tubular_adenoma"]=0 ["tubular_adenoma200X"]=0 ["tubular_adenoma400X"]=0

    ["ductal_carcinoma40X"]=0 ["ductal_carcinoma100X"]=0 ["ductal_carcinoma200X"]=0 ["ductal_carcinoma400X"]=0
    ["lobular_carcinoma40X"]=0 ["lobular_carcinoma100X"]=0 ["lobular_carcinoma200X"]=0 ["lobular_carcinoma400X"]=0
    ["mucinous_carcinoma40X"]=0 ["mucinous_carcinoma100X"]=0 ["mucinous_carcinoma200X"]=0 ["mucinous_carcinoma400X"]=0
    ["papillary_carcinoma40X"]=0 ["papillary_carcinoma100X"]=0 ["papillary_carcinoma200X"]=0 ["papillary_carcinoma400X"]=0
)

for key in "${!count_for_each_class[@]}"; do
  echo "Key: $key, Value: ${count_for_each_class[$key]}"
done

# first, find our tree of directories
for dir in $( find $rdir -type d -print ); do
    # get a count of directories within $dir.
    sdirs=$( find $dir -maxdepth 1 -type d | wc -l );

    # only proceed if sdirs is less than 2 ( 1 = self ).
    if (( $sdirs < 2 )); then 
        # get a count of all the files in $dir, but not in subdirs of $dir)
        files=$( find $dir -maxdepth 1 -type f | wc -l ); 
        echo "$dir : $files"; 

        # get the type of tumour
        tumour_type=$(basename "$(dirname "$(dirname "$dir")")")
        echo "Type of tumour: $tumour_type"

        # get the magnification
        magnification=$(find "$dir" -type d -exec basename {} \; | sort -u | head -n 1)
        echo $magnification

        # get the keyname
        key_name="$tumour_type$magnification"

        # Update the count
        ((count_for_each_class[$tumour_type] += $files))
        ((count_for_each_class[$key_name] += $files))
    fi
done

read -rp "Scripts counted, Press any key to exit..."
for key in "${!count_for_each_class[@]}"; do
  echo "Key: $key, Value: ${count_for_each_class[$key]}"
done
read -rp "Press any key to exit..."