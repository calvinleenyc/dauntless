# dauntless

An implementation of the CDNA model in https://sites.google.com/site/robotprediction/.  Training data not included here - download it at that site (137 GB).

To begin training, run train.py.  Sample test results are in the folder sample_hallucinations.

A few technical notes:
- I didn't implement the DNA model or the STP model.  They're both very similar to the CNDA model.  Also, it looks to me like Figure 8 in the paper is exactly the same as Figure 1, despite what the text says about them.
- I didn't implement the scheduled sampling, though there is a comment in the code marking the place where one could do so.
