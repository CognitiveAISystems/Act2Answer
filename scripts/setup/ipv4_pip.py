import socket, sys
_orig = socket.getaddrinfo
def _ipv4_only(host, *a, **k):
    res = _orig(host, *a, **k)
    v4 = [r for r in res if r[0] == socket.AF_INET]
    return v4 or res
socket.getaddrinfo = _ipv4_only
from pip._internal.cli.main import main
sys.exit(main(sys.argv[1:]))
