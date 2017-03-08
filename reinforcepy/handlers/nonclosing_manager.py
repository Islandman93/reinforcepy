from multiprocessing.managers import SyncManager
import signal


# initilizer for SyncManager
def manager_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class NonclosingManager(SyncManager):
    """
    This class will keep it's sockets open when you press CTRL+C so you don't get a buffer closed error
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start(manager_init)
