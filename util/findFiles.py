import os


def findfiles(target):
    '''Searches a specified directory and locates all absolute file paths.

        Arguments:
            - target    String path to the target directory to be searched

        Returns:
            - list of file paths (iterable)
    '''
    data = []

    for path, subFolders, files in os.walk(target):
        files.sort()
        for file in files:
            filepath = os.path.join(os.path.abspath(path), file)
            data.append(filepath)
            print(filepath)

    return data
