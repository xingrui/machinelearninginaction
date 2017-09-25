basedir=$PWD
for dir in Ch*
do
    echo $basedir/$dir
    cd $basedir/$dir
    /bin/python demo.py
    if [ $? -ne 0 ]
    then
        echo "Error when run test in $dir"
        exit 1
    fi
done
