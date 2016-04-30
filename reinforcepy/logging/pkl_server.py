import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from reinforcepy.logging.pklstats_to_json import load_stat_file


class HTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """ Handle GET Request"""
        path_split = self.path.split('/')
        del path_split[0]
        if path_split[0] == 'stats':
            if path_split[1] == 'getlist':
                # Send file content.
                self.wfile.write(json.dumps(stat_files).encode())
            elif path_split[1] == 'getstat':
                # Send file content.
                self.wfile.write(json.dumps(load_stat_file(path_split[2])).encode())

if __name__ == '__main__':
    dir = 'D:\\_code\\reinforcepy\\examples\\ALE\\DQN_Async'
    # load the stats file list
    os.chdir(dir + '\\saves')
    stat_files = list()
    for file in os.listdir(os.getcwd()):
        if os.path.isfile(file) and file[-9:] == 'stats.pkl':
            stat_files.append(file)

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