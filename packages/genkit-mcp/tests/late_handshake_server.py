# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""A server that answers initialize only after the client has given up on it.

Imports nothing heavy, so its reply lands inside the two seconds the SDK waits
for the child to exit after closing its stdin.
"""

import json
import os
import sys
import threading


def main() -> None:
    initialize = None
    # Reads until the client closes stdin, which it does once the handshake times out.
    for line in sys.stdin.buffer:
        message = json.loads(line)
        if message.get('method') == 'initialize':
            initialize = message
    if initialize is not None:
        reply = {
            'jsonrpc': '2.0',
            'id': initialize['id'],
            'result': {
                'protocolVersion': initialize['params']['protocolVersion'],
                'capabilities': {'tools': {}},
                'serverInfo': {'name': 'late-handshake-server', 'version': '0'},
            },
        }
        os.write(1, (json.dumps(reply) + '\n').encode())
    # Stays up so the client's stdout reader, not process exit, sees the reply.
    threading.Event().wait()


if __name__ == '__main__':
    main()
