python_bin=/bin/python

function tests
{
    export M=$(cat left.txt.org | wc -l)
    export N=$(cat right.txt.org | wc -l)
    export P=$(head -1 right.txt.org | wc -w)
    cat -n left.txt.org > left.txt
    cat -n right.txt.org > right.txt 
    ${python_bin} mrMatrixDot.py left.txt right.txt
    rm left.txt right.txt
}

tests

