
import os
import sys
import logging
import subprocess

def daemonize():
    '''Daemonize the current process.'''
    # Fork the first time
    if os.fork() > 0:
        sys.exit(0)

    # Detach from the parent environment
    os.setsid()

    # Fork a second time to prevent re-acquiring a controlling terminal
    if os.fork() > 0:
        sys.exit(0)

    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()
    with open("/dev/null", "w") as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())
        os.dup2(devnull.fileno(), sys.stdout.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())


def killProcessByName(processName):
    '''Copied from plico/utils'''
    proc= subprocess.Popen("pgrep -f %s" % processName,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           shell=True)

    pgrepPID= proc.pid
    pids= []
    for each in proc.stdout:
        if int(each) == pgrepPID:
            continue
        logging.debug("%s with PID %s" % (processName, each))
        pids.append(int(each))

    logging.info("number of processes '%s': %d" % (processName, len(pids)))
    for each in pids:
        cmd= "kill -KILL %d" % each
        logging.debug("Executing %s" % cmd)
        exitCode= subprocess.call(cmd, shell=True)
        assert exitCode == 0, "Terminating %s with PID %d" % (
            processName, each)