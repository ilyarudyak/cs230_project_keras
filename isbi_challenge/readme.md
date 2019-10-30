## plans

1) finish with the model - batch norm, adam, modified loss on border 
2) try to improve data - transformations, elastic deformation, retrain on all images
if no improvement lets use approach from klibisz;
3) read projects from Stanford;

basically, there are only few things left from klibisz:
- his approach to data;
- retraining on all data;
- modified loss on border;

and that's probably it; if you do something wrong we'll figure it out later;
let's then start another project;

## questions
- what `Adam` params does he use? standard parameters, just changes lr;
- what is `he_normal`? he is for `relu` activation that we use, default is glorot - suited for tanh; so we correctly use it; we may use normal or uniform - klibisz uses normal;
