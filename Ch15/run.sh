python_bin=/bin/python

function tests
{
    ${python_bin} mrMeanMapper.py < inputFile.txt | ${python_bin} mrMeanReducer.py
    ${python_bin} mrMean.py inputFile.txt 
    ${python_bin} mrMean.py --mapper inputFile.txt 
    ${python_bin} pegasos.py
    ${python_bin} mrSVMkickStart.py
    ${python_bin} mrSVM.py kickStart.txt
    ${python_bin} proximalSVM.py
}

tests

