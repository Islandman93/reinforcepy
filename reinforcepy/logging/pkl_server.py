import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from reinforcepy.logging.pklstats_to_json import load_stat_file, load_stats
from reinforcepy.logging.pklparms_to_json import load_parm_file, load_parms
from reinforcepy.logging.pklcheckpoints_to_json import load_checkpoint_file


class HTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """ Handle GET Request"""
        path_split = self.path.split('/')
        del path_split[0]  # remove the first '/'

        if path_split[0] == 'stats':
            if path_split[1] == 'getlist':
                # Send file content.
                self.wfile.write(json.dumps(stat_files).encode())
            elif path_split[1] == 'getstat':
                # Send file content.
                self.wfile.write(json.dumps(load_stat_file(path_split[2])).encode())
        elif path_split[0] == 'parms':
            if path_split[1] == 'getlist':
                # Send file content.
                self.wfile.write(json.dumps(parm_files).encode())
            elif path_split[1] == 'getparm':
                print('sending parm', path_split[2])
                # Send file content.
                self.wfile.write(json.dumps(load_parm_file(path_split[2])).encode())
        elif path_split[0] == 'checkpoints':
            print('sending checkpoints')
            self.wfile.write(json.dumps(load_checkpoint_file()).encode())


if __name__ == '__main__':
    # dir = 'D:\\_code\\reinforcepy\\examples\\ALE\\novelty'
    dir = 'D:\\_code\\reinforcepy\\examples\\ALE\\DQN_Async'
    os.chdir(dir + '\\saves')

    # load the stats file list
    stat_files = load_stats()

    # load the parms file list
    parm_files = load_parms()

    # Port on which server will run.
    PORT = 8080
    HTTPDeamon = HTTPServer(('', PORT), HTTPRequestHandler)

    print("Listening at port", PORT)

    try:
        HTTPDeamon.serve_forever()
    except KeyboardInterrupt:
        pass

    HTTPDeamon.server_close()
    print("Server stopped")
