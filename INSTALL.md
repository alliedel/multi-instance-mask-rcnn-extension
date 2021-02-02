# resolve reference to _C
cd detectron2_repo
python setup.py build develop
ln -s ~/data/datasets datasets

# docker build <build_dir> -t <image>:<image_tag> --build-arg UID=<uid-value> --build-arg GID=<gid-value>
