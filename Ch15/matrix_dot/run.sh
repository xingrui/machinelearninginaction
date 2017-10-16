python_bin=/bin/python

function tests
{
    left_file=left.txt.org
    right_file=right.txt.org
    export M=$(cat ${left_file} | wc -l)
    export N=$(cat ${right_file} | wc -l)
    export P=$(head -1 ${right_file} | wc -w)
    cat -n ${left_file} > left.txt
    cat -n ${right_file} > right.txt 
    ${python_bin} mrMatrixDot.py left.txt right.txt
    ${python_bin} dot.py ${left_file} ${right_file}
    rm left.txt right.txt
}

tests

