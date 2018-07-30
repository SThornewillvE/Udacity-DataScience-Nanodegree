# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 13:04:38 2018

@author: SimonThornewill
"""

import argparse
import numpy as np


def cylinder_volume(height, radius):
    return np.pi*(radius**2)*height

# Create a parser instance from the ArgumentParser class
parser = argparse.ArgumentParser(description="Calculate volume of a Cylidner")

# Add arguments
parser.add_argument('-r', 
                    '--radius', 
                    type=int, 
                    help='Radius of Cylinder', 
                    required=True)

parser.add_argument('-H', 
                    '--height', 
                    type=int, 
                    help='Height of Cylinder', 
                    required=True)

# Create mutually exclusive group
group = parser.add_mutually_exclusive_group()

# Create arguments
group.add_argument('-q', '--quiet', action='store_true', help='print quiet')
group.add_argument('-v', '--verb', action='store_true', help='print verbose')

# Parse args
args = parser.parse_args()

# Calculate the volume using args
volume = cylinder_volume(height=args.height, radius=args.radius)


# Create conditions depending on quiet, verbose or none
if args.quiet: print(volume)
elif args.verb: print("Volume of a Cylinder w/ radius {} and height {} is {}".format(args.radius, 
                                                                                     args.height, 
                                                                                     volume))
else: print("Volume of a Cylinder is {}".format(volume))