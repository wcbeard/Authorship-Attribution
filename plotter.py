#!/usr/bin/env python
from matplotlib import pyplot as pl
from authorship_data import *


bin_num = 10
pl.clf()
# pl.subplot(3,1,1)
pl.subplot(1,3,1)
pl.hist(AC, bins=bin_num)
pl.title('Paragraphs in coauthored papers\n by Authors A and B')
pl.xlabel('Probability that paragraph\n was written by Author B')
pl.ylabel('No. of paragraphs')

# pl.subplot(3,1,2)
pl.subplot(1,3,2)
# pl.clf()
pl.hist(AC, bins=bin_num)
pl.title('Paragraphs in coauthored papers\n by Authors A and C')
pl.xlabel('Probability that paragraph\n was written by Author C')
pl.ylabel('No. of paragraphs')

# pl.subplot(3,1,3)
pl.subplot(1,3,3)
# pl.clf()
pl.hist(AD, bins=bin_num)
pl.title('Paragraphs in coauthored papers\n by Authors A and D')
pl.xlabel('Probability that paragraph\n was written by Author D')
pl.ylabel('No. of paragraphs')

pl.savefig('hist3.png')
# pl.savefig('hist3.pdf')


