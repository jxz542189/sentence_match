def print_log(arg, file):
    file.info(arg)

# print all used hyper-parameters on both SCREEN an LOG file
def print_args(args, log_file):
    """
    :Param args: all used hyper-parameters
    :Param log_f: the log life
    """
    argsDict = vars(args)
    argsList = sorted(argsDict.items())
    print_log("------------- HYPER PARAMETERS -------------", file = log_file)
    for a in argsList:
        print_log("%s: %s" % (a[0], str(a[1])), file = log_file)
    print_log("-----------------------------------------", file = log_file)
    return None


def count_parameters():
    totalParams = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variableParams = 1
        for dim in shape:
            variableParams *= dim.value
        totalParams += variableParams
    return totalParams


def get_time_diff(startTime):
    endTime = time.time()
    diff = endTime - startTime
    return timedelta(seconds = int(round(diff)))