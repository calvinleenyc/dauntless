# dauntless

An implementation of the CDNA model in https://sites.google.com/site/robotprediction/.  Training data not included here - download it at that site (137 GB).

To begin training, run train.py.  Sample test results are in the folder sample_hallucinations.

A few technical notes:
- I didn't implement the DNA model or the STP model.  They're both very similar to the CNDA model.  Also, it looks to me like Figure 8 in the paper is exactly the same as Figure 1, despite what the text says about them.
- I didn't implement the scheduled sampling, though there is a comment in the code marking the place where one could do so.

# View Training in Progress [outdated]
To view current training results, go to http://ec2-52-23-253-9.compute-1.amazonaws.com:8080 (at least for now).

See also: http://ec2-52-23-253-9.compute-1.amazonaws.com:8080/#scalars&runSelectionState=eyJOb3YyM18wOC0zOS0xOF9pcC0xNzItMzEtODAtMTUiOmZhbHNlLCJOb3YyM18wOC01Ni00N19pcC0xNzItMzEtODAtMTUiOmZhbHNlLCJOb3YyM18wOC01OC0wN19pcC0xNzItMzEtODAtMTUiOmZhbHNlLCJOb3YyM18wOS0wMS01OF9pcC0xNzItMzEtODAtMTUiOmZhbHNlLCJOb3YyM18wOS0wMy0xNV9pcC0xNzItMzEtODAtMTUiOmZhbHNlLCJOb3YyM18wOS0wNS0xNl9pcC0xNzItMzEtODAtMTUiOmZhbHNlLCJOb3YyM18wOS0wOS0zOV9pcC0xNzItMzEtODAtMTUiOmZhbHNlLCJOb3YyM18wOS0xMi0wN19pcC0xNzItMzEtODAtMTUiOnRydWV9&_smoothingWeight=0.989

[Training has now stopped, none of those links work anymore.]
