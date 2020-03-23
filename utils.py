# Print iterations progress
def ProgressBar (
    iteration, total, prefix = '', 
    suffix = '', decimals = 1, 
    length = 10, fill = '█', 
    printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + ' ' * (length - filledLength)
    print('\r %s |%s| %s%% -- err => %.7s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()