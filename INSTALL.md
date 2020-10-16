# resolve reference to _C
cd detectron2_repo
python setup.py build develop
ln -s ~/data/datasets datasets
