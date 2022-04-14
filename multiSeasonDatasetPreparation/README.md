From the Sen12MS dataset (https://github.com/schmitt-muc/SEN12MS), first download the folders ROIs1868_summer_s2, ROIs1158_spring_s2, ROIs1970_fall_s2, ROIs2017_winter_s2 into this directory.

Then run
$ python multiseasonDatasetMaker.py

As an output, you will get a directory called "multiseasonDatset" which will have four directories with images (fall, spring, summer, and winter) and four .txt files (fall.txt, spring.txt, summer.txt, winter.txt).
