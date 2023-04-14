# Processing-HARPS-spectra-with-wobble
This code uses the Python library wobble (Bedell et al., 2019 - The Astronomical Journal, 158, 164) to process HARPS spectra and obtain the components corresponding to the star and to the tellurics.

The file processing_wobble.py is the code used to perform the processing/corrections. The other two codes are auxiliary, used by the main code. where_array.c finds the indexes of repeated elements in arrays and DER_SNR.py is a code written by Stoeh et al. (2008) to find the sigbal-to-noise ratio of spectra.
