structure of files:

Training file (Vaihingen3D_Traininig.pts)
X Y Z Intensity return_number number_of_returns label

Evaluation (Vaihingen3D_EVAL_WITHOUT_REF.pts)
X Y Z Intensity return_number number_of_returns 

The label means

0 Powerline
1 Low vegetation
2 Impervious surfaces
3 Car
4 Fence/Hedge
5 Roof
6 Facade
7 Shrub
8 Tree


In this folder you also find two screenshots showing the two areas. The color coding in the image showing the training area is

R   G   B   label
===================
255 255 125 Powerline
0 255 255 Low vegetation
255 255 255 Impervious surfaces
255 255 0 Car
0 255 125 Fence/Hedge
0 0 255 Roof
0 125 255 Facade
125 255 0 Shrub
0 255 0 Tree


SUBMISSION
Participants are expected to deliver for each point in Vaihingen3D_EVAL_WITHOUT_REF.pts a list of XYZ coordinates and for each point a label assigned. The sequence of points can be different compared to the provided file, but the precision of points should not be altered too much (to enable a reliable closest point search to our reference).

Example:

497074.9300 5419772.0400 266.0400 1 
497074.9300 5419772.0800 266.0200 5 
497074.9300 5419772.0900 266.0500 0 

A simple pts file like this should be sent (ZIPPED, i.e. in a size of max 5MB) to the address indicated on the website.

OMISSION OF DATA
In contrast to the 2D labeling challenge we allow participants to provide only partial data, i.e. to omit classes and points in the delivered data. If you for instance only classify vegetation it is allowed to just provide the points you labeled with ID 1,7,8. Of course, during evaluation false positive and false negative decisions will be revealed and considered, as well.


