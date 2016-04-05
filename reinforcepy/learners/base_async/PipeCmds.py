from enum import Enum


class PipeCmds(Enum):
    """
    Enum that holds all pipe commands that can be passed from/to the host or client.

    # host commands

    Start, End, HostSendingGlobalParameters


    # client commands

    ClientSendingGradientsSteps, ClientSendingStats
    """
    # host commands
    Start = 1
    End = 2
    HostSendingGlobalParameters = 3

    # client commands
    ClientSendingGradientsSteps = 4
    ClientSendingStats = 5