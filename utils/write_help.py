# Write the given line to the file of the provided name
def write_to_file(filename, line):
    with open(filename, 'a') as file:
        file.write(line + '\n')
