The executable "gensvmdat" randomly generates points in R4 that take values inside [0,1]. If the sum of their coordinates is more than 2, then the point belongs to the class +1; otherwise -1.

To run code:
	$> gensvmdat file p seed
E.g.:
	$> gensvmdat dataset.dat 100 12345

Points marked with "*", randomly distributed, belong to an incorrect class.

In order to convert the file that the executable generated to AMPL input data we need to delete the *.