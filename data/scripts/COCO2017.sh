#! /bin/bash

start=`data +%s`

# handle optional download dir
if [ -z "$1"]
    then
      # navigate to ~/data
      echo "navigating to ../data/ ..."
      cd ../
      mkdir -p ./coco
      cd ./coco
      mkdir -p ./images
      mkdir -p ./annotations
    else
      # check if specified dir is valid
      if [ ! -d $1]; then
          echo $1 " is not a valid directory "
      fi
      echo "navigating to " $1 " ..."
      cd $1
fi

if [ ! -d images]
  then
    mkdir -p ./images
fi

# Download the image data
cd ./images
echo "Downloading MSCOCO train images ..."
curl -LO http://images.cocodataset.org/zips/train2017.zip
echo "Downloading MSCOCO test images ..."
curl -LO http://images.cocodataset.org/zips/test2015.zip

cd ../
if [ ! -d annotations]
  then
    mkdir -p ./annotations
fi

# Download the annotation data
cd ./annotations
echo "Downloading MSCOCO train/val annotations ..."
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
echo "Finished downloading. Now extracting ..."

# Unzip data
echo "Extracting train images ..."
unzip ../images/train2017.zip -d ../images
echo "Extracting test images ..."
unzip ../images/test2015.zip -d ../images
echo "Extracting annotations ..."
unzip ./annotations_trainval2017.zip

echo "Removing zip files ..."
rm ../images/trainval2017.zip
rm ../images/test2015.zip
rm ./annotations_trainval2017.zip

end=`date +%s`
runtime=$((end-start))

echo "Completed in " $runtime " seconds"

