import os
from mrjob.job import MRJob
from mrjob.step import MRStep

class MRMatrixDot(MRJob):
    def __init__(self, *args, **kwargs):
        super(MRMatrixDot, self).__init__(*args, **kwargs)
        self.M = int(os.environ.get('M'))
        self.N = int(os.environ.get('N'))
        self.P = int(os.environ.get('P'))
        self.t = 0
        if os.environ.get('map_input_file') == 'right.txt':
            self.t = 1

    #needs exactly 2 arguments
    def map(self, key, val):
        row_num, str_values = val.strip().split("\t", 2)
        row_num = int(row_num) - 1
        values = map(float, str_values.split())
        if self.t == 0:
            for i, val in enumerate(values):
                for j in xrange(self.P):
                    yield (row_num, [j, i, val])
        elif self.t == 1:
            for i, val in enumerate(values):
                for j in xrange(self.M):
                    yield (j, [i, row_num, val])

    def reduce(self, key, packedValues):
        row_num = key
        pre_column = -1
        pre_index = -1
        for valArr in packedValues:
            current_column, current_index, val = valArr

            if current_column != pre_column:
                if pre_column != -1:
                    location_sum += index_product
                    yield ((row_num,pre_column),location_sum)
                location_sum = 0
                pre_column = current_column
                pre_index = -1

            if pre_index != current_index:
                if pre_index != -1:
                    location_sum += index_product
                index_product = val
                pre_index = current_index
            else:
                index_product *= val
        location_sum += index_product
        yield ((row_num,pre_column),location_sum)
        
    def steps(self):
        return ([MRStep(mapper=self.map,\
                          reducer=self.reduce,)])

if __name__ == '__main__':
    MRMatrixDot.run()
