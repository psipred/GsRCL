sample1_error.csv - too many rows (1001) to verify
sample2_error.csv - rows do not have text labels
sample3_error.csv - data geometry is incorrect. There must be more measurements 
                    (columns) than there are samples (rows)
sample4_error.csv - The named genes (columns) can not be found in the reference
                    data
sample5_error.csv - some zeroes are listed as 0.0 rather than 0 in this counts
                    file, verify script now rewrites the 0 values depending on
                    user log request. And verifies each data point

test_file1.csv - should verify correctly as integers under -l 1 params
test_file1.csv - should verify correctly as floats under -l 0 params
