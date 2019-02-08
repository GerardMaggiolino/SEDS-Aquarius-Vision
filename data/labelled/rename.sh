# Renames all .tiff files in current directory to {0, 1, ..., n}.
x=0
for file in *.tiff; do 
    mv $file "${x}.tiff"
    x=$((x+1))
done
