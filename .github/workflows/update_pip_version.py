#!/usr/bin/env python3

import sys, getopt, fileinput

def main(argv):
	help_string = 'USAGE: update_pip_version.py -v <versionstring>'
	new_version = ''
	try:
		opts, args = getopt.getopt(argv, "hv:", ["version ="])
	except:
		print(help_string)	
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print(help_string)
			sys.exit()
		elif opt in ("-v", "--version"):
			new_version = arg
	
	if new_version == "":
		print(help_string)
		sys.exit(2)
	
	print("NEW VERSION: " + new_version)
	# Write the new version to the file
	with fileinput.FileInput("../../setup.py", inplace=True, backup='.bak') as file:
		for line in file:
			if "version =" in line:
				line = "\tversion = \"" + new_version + "\","
			print(line.rstrip())
if __name__ == "__main__":
	main(sys.argv[1:]) 
